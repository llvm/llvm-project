// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: define void @exactflag_func
llvm.func @exactflag_func(%arg0: i64, %arg1: i64) {
  // CHECK: %{{.*}} = udiv exact i64 %{{.*}}, %{{.*}}
  %0 = llvm.udiv exact %arg0, %arg1 : i64
  // CHECK: %{{.*}} = sdiv exact i64 %{{.*}}, %{{.*}}
  %1 = llvm.sdiv exact %arg0, %arg1 : i64
  // CHECK: %{{.*}} = lshr exact i64 %{{.*}}, %{{.*}}
  %2 = llvm.lshr exact %arg0, %arg1 : i64
  // CHECK: %{{.*}} = ashr exact i64 %{{.*}}, %{{.*}}
  %3 = llvm.ashr exact %arg0, %arg1 : i64
  llvm.return
}
