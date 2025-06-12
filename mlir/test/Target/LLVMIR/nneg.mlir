// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: define void @nnegflag_func
llvm.func @nnegflag_func(%arg0: i32) {
  // CHECK: %{{.*}} = zext nneg i32 %{{.*}} to i64
  %0 = llvm.zext nneg %arg0 : i32 to i64
  // CHECK: %{{.*}} = uitofp nneg i32 %{{.*}} to float
  %1 = llvm.uitofp nneg %arg0 : i32 to f32
  llvm.return
}
