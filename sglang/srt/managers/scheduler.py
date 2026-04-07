"""Scheduler for SGLang runtime."""

import time


class Scheduler:
    """Scheduler manages batch scheduling for inference requests."""

    def __init__(self):
        self.queue = []
        self.running = []
        self.metrics = {"batches_scheduled": 0, "total_requests": 0}

    def get_next_batch_to_run(self):
        """Select the next batch of requests to run.

        Returns:
            List of requests to process, or None if no requests available.
        """
        # Step 1: Check if there are requests in queue
        if not self.queue:
            return None

        # Step 2: Sort by request phase (PREFILL before DECODE), then by arrival
        # [UPSTREAM v1.2] Added prefill/decode phase separation
        sorted_requests = sorted(
            self.queue,
            key=lambda r: (r.phase, r.arrival_time)
        )

        # Step 3: Select up to batch_size requests
        batch_size = 32
        batch = sorted_requests[:batch_size]

        # Step 4: Move selected requests to running
        for req in batch:
            self.queue.remove(req)
            self.running.append(req)

        # Step 5: Return the batch with metrics tracking
        self.metrics["batches_scheduled"] += 1
        self.metrics["total_requests"] += len(batch)
        return batch


class Request:
    """A single inference request."""

    def __init__(self, request_id, arrival_time, phase=0):
        self.request_id = request_id
        self.arrival_time = arrival_time
        self.phase = phase  # 0 = PREFILL, 1 = DECODE
