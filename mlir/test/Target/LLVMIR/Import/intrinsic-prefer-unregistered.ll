; RUN: mlir-translate -import-llvm -prefer-unregistered-intrinsics %s | FileCheck %s

; CHECK-LABEL: llvm.func @lifetime
define void @lifetime() {
  %a = alloca [16 x i8]
  ; CHECK: llvm.call_intrinsic "llvm.lifetime.start.p0"(%[[ptr:.*]]) : (!llvm.ptr {llvm.nonnull}) -> ()
  call void @llvm.lifetime.start.p0(ptr nonnull %a)
  ; CHECK: llvm.call_intrinsic "llvm.lifetime.end.p0"(%[[ptr]]) : (!llvm.ptr {llvm.nonnull}) -> ()
  call void @llvm.lifetime.end.p0(ptr nonnull %a)
  ret void
}
