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

; CHECK-LABEL: llvm.func @assume(
; CHECK-SAME: %[[PTR:.*]]: !llvm.ptr)
define void @assume(ptr %p) {
  ; CHECK: %[[TRUE:.*]] = llvm.mlir.constant(true) : i1
  ; CHECK: %[[ALIGN:.*]] = llvm.mlir.constant(16 : i64) : i64
  ; CHECK: llvm.call_intrinsic "llvm.assume"(%[[TRUE]]) ["align"(%[[PTR]], %[[ALIGN]] : !llvm.ptr, i64)] : (i1) -> ()
  call void @llvm.assume(i1 true) ["align"(ptr %p, i64 16)]
  ret void
}
