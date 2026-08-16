// Schedule only. Payload IR is the computation; this file is the schedule.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @column_sum_schedule(
      %payload: !transform.any_op {transform.readonly}) {
    %red = transform.structured.match ops{["linalg.generic"]} in %payload
        : (!transform.any_op) -> !transform.any_op
    // Column reduction: iterators [reduction, parallel]. Tile the parallel dim.
    %tiled, %loop = transform.structured.tile_using_for %red tile_sizes [0, 64]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
