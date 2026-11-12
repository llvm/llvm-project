// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: define void @intflags_func
llvm.func @intflags_func(%arg0: i64, %arg1: i64) {
  // CHECK: %{{.*}} = add nsw i64 %{{.*}}, %{{.*}}
  %0 = llvm.add %arg0, %arg1 overflow <nsw> : i64
  // CHECK: %{{.*}} = sub nuw i64 %{{.*}}, %{{.*}}
  %1 = llvm.sub %arg0, %arg1 overflow <nuw> : i64
  // CHECK: %{{.*}} = mul nuw nsw i64 %{{.*}}, %{{.*}}
  %2 = llvm.mul %arg0, %arg1 overflow <nsw, nuw> : i64
  // CHECK: %{{.*}} = shl nuw nsw i64 %{{.*}}, %{{.*}}
  %3 = llvm.shl %arg0, %arg1 overflow <nsw, nuw> : i64
  // CHECK: %{{.*}} = trunc nuw i64 %{{.*}} to i32
  %4 = llvm.trunc %arg1 overflow<nuw> : i64 to i32
  llvm.return
}
