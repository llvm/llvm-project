// RUN: mlir-opt %s | mlir-opt | FileCheck %s

transform.sequence failures(propagate) {
^bb1(%arg0: !pdl.operation):
  // CHECK %{{.*}}, %{{.*}}:2 = transform.structured.tile
  %0, %1:2 = transform.structured.tile %arg0 [2, 0, 3] : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
}

transform.sequence failures(propagate) {
^bb1(%arg0: !transform.any_op):
  %0:2 = transform.structured.split %arg0 after 42 { dimension = 0 } : !transform.any_op
  transform.structured.split %0#0 after %0#1 { dimension = 1 } : !transform.any_op, !transform.any_op
}

//===----------------------------------------------------------------------===//
// Check that operations are registered correctly through the extension
// mechanism. Their syntax is generated and requires no additional testing since
// we test the generator.
//===----------------------------------------------------------------------===//

transform.sequence failures(propagate) {
^bb1(%arg0: !pdl.operation):
  // CHECK: transform.structured.pad
  %0 = transform.structured.pad %arg0
}

transform.sequence failures(propagate) {
^bb1(%arg0: !pdl.operation):
  // CHECK: transform.structured.interchange
  %0 = transform.structured.interchange %arg0
}

transform.sequence failures(propagate) {
^bb1(%arg0: !pdl.operation):
  // CHECK: transform.structured.scalarize
  %0 = transform.structured.scalarize %arg0
}
