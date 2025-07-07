// RUN: mlir-opt -allow-unregistered-dialect -snapshot-op-locations='filename=%/t' -mlir-print-local-scope -mlir-print-debuginfo %s | FileCheck %s -DFILE=%/t
// RUN: mlir-opt -allow-unregistered-dialect -snapshot-op-locations='filename=%/t tag='tagged'' -mlir-print-local-scope -mlir-print-debuginfo %s | FileCheck %s --check-prefix=TAG -DFILE=%/t
// RUN: mlir-opt -allow-unregistered-dialect -snapshot-op-locations='filename=%/t print-debuginfo' -mlir-print-local-scope -mlir-print-debuginfo %s | FileCheck %s --check-prefix=DBG -DFILE=%/t && cat %/t | FileCheck %s --check-prefix=DBGFILE

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

// DBG: func @function2(
// DBG-NEXT: loc("[[FILE]]":{{[0-9]+}}:{{[0-9]+}})
// DBG-NEXT: loc("[[FILE]]":{{[0-9]+}}:{{[0-9]+}})
// DBG-NEXT: } loc("[[FILE]]":{{[0-9]+}}:{{[0-9]+}})

// DBGFILE: func @function2(
// DBGFILE-NEXT: loc("{{.*}}location-snapshot.mlir":{{[0-9]+}}:{{[0-9]+}})
// DBGFILE-NEXT: loc("{{.*}}location-snapshot.mlir":{{[0-9]+}}:{{[0-9]+}})
// DBGFILE-NEXT: } loc("{{.*}}location-snapshot.mlir":{{[0-9]+}}:{{[0-9]+}})

func.func @function2() -> i32 {
  %1 = "foo"() : () -> i32
  return %1 : i32
}