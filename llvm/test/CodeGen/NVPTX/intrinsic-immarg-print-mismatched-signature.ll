; NOTE: This test guards AsmWriter's intrinsic immediate-argument pretty-printer
;       against a null-pointer dereference.
; RUN: llvm-as -disable-verify < %s | llvm-dis | FileCheck %s

declare void @llvm.nvvm.tensormap.replace.elemtype.p0(ptr, i32)

; A well-formed call: the elemtype immediate is pretty-printed.
define void @valid(ptr %p) {
  ; CHECK-LABEL: define void @valid(
  ; CHECK: call void @llvm.nvvm.tensormap.replace.elemtype.p0(ptr %p, /* elemtype=u8 */ i32 0)
  call void @llvm.nvvm.tensormap.replace.elemtype.p0(ptr %p, i32 0)
  ret void
}

; A call with a mismatched signature (an extra operand): getCalledFunction() is
; null, so no comment is emitted and printing must not crash.
define void @mismatched_signature(ptr %p) {
  ; CHECK-LABEL: define void @mismatched_signature(
  ; CHECK: call void @llvm.nvvm.tensormap.replace.elemtype.p0(ptr %p, i32 0, i32 0)
  call void @llvm.nvvm.tensormap.replace.elemtype.p0(ptr %p, i32 0, i32 0)
  ret void
}
