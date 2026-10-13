; RUN: llc -verify-machineinstrs < %s -mtriple=arm64-apple-macosx | FileCheck %s
; RUN: llc -verify-machineinstrs -global-isel < %s -mtriple=arm64-apple-macosx | FileCheck %s

; Tail calls in a function that takes a swifterror argument isn't supported yet.
; Check that memcpy etc. aren't incorrectly tail called.

declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg)
declare void @llvm.memmove.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)

define swiftcc void @swifterror_tailcall_memcpy(ptr swifterror %err, ptr %dst, ptr %src, i64 %n) {
; CHECK-LABEL: swifterror_tailcall_memcpy:
; CHECK: bl {{_?}}memcpy
entry:
  tail call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 %n, i1 false)
  ret void
}

define swiftcc void @swifterror_tailcall_memmove(ptr swifterror %err, ptr %dst, ptr %src, i64 %n) {
; CHECK-LABEL: swifterror_tailcall_memmove:
; CHECK: bl {{_?}}memmove
entry:
  tail call void @llvm.memmove.p0.p0.i64(ptr %dst, ptr %src, i64 %n, i1 false)
  ret void
}

define swiftcc void @swifterror_tailcall_memset(ptr swifterror %err, ptr %dst, i64 %n) {
; CHECK-LABEL: swifterror_tailcall_memset:
; CHECK: bl {{_?}}memset
entry:
  tail call void @llvm.memset.p0.i64(ptr %dst, i8 42, i64 %n, i1 false)
  ret void
}
