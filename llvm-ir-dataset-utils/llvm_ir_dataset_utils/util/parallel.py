"""Utilities for job distribution and execution."""


# TODO(boomanaiden154): Write some unit tests for this function.
def split_batches(individual_jobs, batch_size):
  batches = []
  current_start_index = 0
  while True:
    end_index = current_start_index + batch_size
    chunk = individual_jobs[current_start_index:end_index]
    batches.append(chunk)
    current_start_index = end_index
    if current_start_index + batch_size >= len(individual_jobs):
      last_chunk = individual_jobs[current_start_index:]
      batches.append(last_chunk)
      break
  return batches
