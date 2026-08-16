// Schedule only. Payload IR is the computation; this file is the schedule.
// Applied with --transform-preload-library + --transform-interpreter.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @row_sum_schedule(
      %payload: !transform.any_op {transform.readonly}) {
    %red = transform.structured.match ops{["linalg.generic"]} in %payload
        : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_for %red tile_sizes [64, 0]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
