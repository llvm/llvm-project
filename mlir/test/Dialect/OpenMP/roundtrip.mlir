// RUN: mlir-opt -verify-diagnostics %s | mlir-opt | FileCheck %s

// CHECK: omp.private {type = private} @x.privatizer : !llvm.ptr alloc {
omp.private {type = private} @x.privatizer : !llvm.ptr alloc {
// CHECK: ^bb0(%arg0: {{.*}}):
^bb0(%arg0: !llvm.ptr):
  omp.yield(%arg0 : !llvm.ptr)
}

// CHECK: omp.private {type = firstprivate} @y.privatizer : !llvm.ptr alloc {
omp.private {type = firstprivate} @y.privatizer : !llvm.ptr alloc {
// CHECK: ^bb0(%arg0: {{.*}}):
^bb0(%arg0: !llvm.ptr):
  omp.yield(%arg0 : !llvm.ptr)
// CHECK: } copy {
} copy {
// CHECK: ^bb0(%arg0: {{.*}}, %arg1: {{.*}}):
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  omp.yield(%arg0 : !llvm.ptr)
}

