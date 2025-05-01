; RUN: mlir-translate -import-llvm -prefer-unregistered-intrinsics %s | FileCheck %s

; CHECK-LABEL: llvm.func @lifetime
define void @lifetime(ptr %0) {
  ; CHECK: llvm.call_intrinsic "llvm.lifetime.start.p0"({{.*}}, %arg0) : (i64, !llvm.ptr {llvm.nonnull}) -> !llvm.void
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %0)
  ; CHECK: llvm.call_intrinsic "llvm.lifetime.end.p0"({{.*}}, %arg0) : (i64, !llvm.ptr {llvm.nonnull}) -> !llvm.void
  call void @llvm.lifetime.end.p0(i64 32, ptr nonnull %0)
  ret void
}
