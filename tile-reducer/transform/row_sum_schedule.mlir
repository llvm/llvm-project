// Named schedule. Public @row_sum_schedule looks up private @tile_row_reduction
// via SymbolRefAttr (transform.include).

module attributes {transform.with_named_sequence} {
  transform.named_sequence private @tile_row_reduction(
      %payload: !transform.any_op {transform.readonly}) {
    %red = transform.structured.match ops{["linalg.generic"]} in %payload
        : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_for %red tile_sizes [64, 0]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }

  transform.named_sequence @row_sum_schedule(
      %payload: !transform.any_op {transform.readonly}) {
    transform.include @tile_row_reduction failures(propagate) (%payload)
        : (!transform.any_op) -> ()
    transform.yield
  }
}
