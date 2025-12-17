// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: define void @disjointflag_func
llvm.func @disjointflag_func(%arg0: i64, %arg1: i64) {
  // CHECK: %{{.*}} = or disjoint i64 %{{.*}}, %{{.*}}
  %0 = llvm.or disjoint %arg0, %arg1 : i64
  llvm.return
}
