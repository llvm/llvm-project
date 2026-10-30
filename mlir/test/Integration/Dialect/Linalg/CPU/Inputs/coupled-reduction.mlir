module attributes {transform.with_named_sequence} {
  transform.named_sequence @split_forall(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} attributes {test.coupled_pair} in %arg0 : (!transform.any_op) -> !transform.any_op
    %fill_op:2, %split_op, %combining_op, %forall_op = transform.structured.tile_reduction_using_forall %0 by num_threads = [4, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %1 = transform.structured.match ops{["linalg.generic"]} attributes {test.coupled_mixed} in %arg0 : (!transform.any_op) -> !transform.any_op
    %fill_op_0:4, %split_op_1, %combining_op_2, %forall_op_3 = transform.structured.tile_reduction_using_forall %1 by num_threads = [4, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
  transform.named_sequence @split_for(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} attributes {test.coupled_pair} in %arg0 : (!transform.any_op) -> !transform.any_op
    %fill_op:2, %split_op, %combining_op, %for_op = transform.structured.tile_reduction_using_for %0 by tile_sizes = [4, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %1 = transform.structured.match ops{["linalg.generic"]} attributes {test.coupled_mixed} in %arg0 : (!transform.any_op) -> !transform.any_op
    %fill_op_0:4, %split_op_1, %combining_op_2, %for_op_3 = transform.structured.tile_reduction_using_for %1 by tile_sizes = [4, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
