// RUN: mlir-opt -allow-unregistered-dialect -snapshot-op-locations='filename=%/t' -mlir-print-local-scope -mlir-print-debuginfo %s | FileCheck %s -DFILE=%/t
// RUN: mlir-opt -allow-unregistered-dialect -snapshot-op-locations='filename=%/t tag='tagged'' -mlir-print-local-scope -mlir-print-debuginfo %s | FileCheck %s --check-prefix=TAG -DFILE=%/t

// CHECK: func @function(
// CHECK-NEXT: loc("[[FILE]]":{{[0-9]+}}:{{[0-9]+}})
// CHECK-NEXT: loc("[[FILE]]":{{[0-9]+}}:{{[0-9]+}})
// CHECK-NEXT: } loc("[[FILE]]":{{[0-9]+}}:{{[0-9]+}})

// TAG: func @function(
// TAG-NEXT: loc(fused["original", "tagged"("[[FILE]]":{{[0-9]+}}:{{[0-9]+}})])
// TAG-NEXT: loc(fused["original", "tagged"("[[FILE]]":{{[0-9]+}}:{{[0-9]+}})])
// TAG-NEXT: } loc(fused["original", "tagged"("[[FILE]]":{{[0-9]+}}:{{[0-9]+}})])

func.func @function() -> i32 {
  %1 = "foo"() : () -> i32 loc("original")
  return %1 : i32 loc("original")
} loc("original")
