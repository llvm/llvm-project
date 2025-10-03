// RUN: mlir-opt %s --split-input-file | mlir-opt | FileCheck %s

transform.sequence failures(propagate) {
^bb1(%arg0: !transform.any_op):
  // CHECK: %{{.*}}, %{{.*}}:2 = transform.structured.tile
  %0, %1:2 = transform.structured.tile_using_for %arg0 tile_sizes [2, 0, 3] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
}

// check that the Attributes of `tile_using_for` are preserved through printing
// and parsing with and without use of the optional `interchange` Attribute.
transform.sequence failures(propagate) {
^bb1(%arg0: !transform.any_op):
  // CHECK: %{{.*}}, %{{.*}}:2 = transform.structured.tile_using_for %arg0 tile_sizes [2, 0, 3] interchange = [2, 1] {test_attr1 = 1 : i64, test_attr2}
  %0, %1:2 = transform.structured.tile_using_for %arg0 tile_sizes [2, 0, 3] interchange = [2, 1] {test_attr1 = 1 : i64, test_attr2}: (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
  // CHECK: %{{.*}}, %{{.*}}:2 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 5, 3] {test_attr3 = 1 : i64, test_attr4}
  %2, %3:2 = transform.structured.tile_using_for %0 tile_sizes [0, 5, 3] {test_attr3 = 1 : i64, test_attr4}: (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
}

transform.sequence failures(propagate) {
^bb1(%arg0: !transform.any_op):
  %t = transform.structured.split %arg0 after 42 { dimension = 0 } : !transform.any_op
  %0:2 = transform.split_handle %t : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  transform.structured.split %0#0 after %0#1 { dimension = 1 } : !transform.any_op, !transform.any_op
}

//===----------------------------------------------------------------------===//
// Check that operations are registered correctly through the extension
// mechanism. Their syntax is generated and requires no additional testing since
// we test the generator.
//===----------------------------------------------------------------------===//

transform.sequence failures(propagate) {
^bb1(%arg0: !transform.any_op):
  // CHECK: transform.structured.pad
  %0, %1, %2 = transform.structured.pad %arg0 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
}

transform.sequence failures(propagate) {
^bb1(%arg0: !transform.any_op):
  // CHECK: transform.structured.interchange
  %0 = transform.structured.interchange %arg0 : (!transform.any_op) -> !transform.any_op
}

transform.sequence failures(propagate) {
^bb1(%arg0: !transform.any_op):
  // CHECK: transform.structured.scalarize
  %0 = transform.structured.scalarize %arg0 : (!transform.any_op) -> !transform.any_op
}

// Check that the second argument of `fuse_into_containing_op` is not consumed
// (if it had been, we would have seen a diagnostic about multiple consumers).
transform.sequence failures(propagate) {
^bb1(%arg0: !transform.any_op, %arg1: !transform.any_op, %arg2: !transform.any_op):
  %loop = transform.structured.match ops{["scf.forall"]} in %arg0
    : (!transform.any_op) -> !transform.any_op
  %0:2 = transform.structured.fuse_into_containing_op %arg1 into %loop
    : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
  %1:2 = transform.structured.fuse_into_containing_op %arg2 into %loop
    : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // CHECK: transform.structured.vectorize %arg0 : !transform.any_op
  transform.structured.vectorize %arg0 vector_sizes [] : !transform.any_op

}
