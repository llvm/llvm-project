// RUN: mlir-opt %s | mlir-opt | FileCheck %s

transform.sequence {
^bb1(%arg0: !pdl.operation):
  // CHECK %{{.*}}, %{{.*}}:2 = transform.structured.tile
  %0, %1:2 = transform.structured.tile %arg0 [2, 0, 3]
}

//===----------------------------------------------------------------------===//
// Check that operations are registered correctly through the extension
// mechanism. Their syntax is generated and requries no additional testing since
// we test the generator.
//===----------------------------------------------------------------------===//

transform.sequence {
^bb1(%arg0: !pdl.operation):
  // CHECK: transform.structured.pad
  %0 = transform.structured.pad %arg0
}

transform.sequence {
^bb1(%arg0: !pdl.operation):
  // CHECK: transform.structured.interchange
  %0 = transform.structured.interchange %arg0
}

transform.sequence {
^bb1(%arg0: !pdl.operation):
  // CHECK: transform.structured.scalarize
  %0 = transform.structured.scalarize %arg0
}
