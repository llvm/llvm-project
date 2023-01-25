; RUN: llc -mtriple=aarch64-unknown-unknown < %s | FileCheck %s
; RUN: llc -global-isel-abort=1 -verify-machineinstrs -mtriple=aarch64-unknown-unknown -global-isel < %s | FileCheck %s

; CHECK-LABEL: tail_memcpy:
; CHECK: b memcpy
define void @tail_memcpy(ptr nocapture %p, ptr nocapture readonly %q, i32 %n) #0 {
entry:
  tail call void @llvm.memcpy.p0.p0.i32(ptr %p, ptr %q, i32 %n, i1 false)
  ret void
}

; CHECK-LABEL: tail_memmove:
; CHECK: b memmove
define void @tail_memmove(ptr nocapture %p, ptr nocapture readonly %q, i32 %n) #0 {
entry:
  tail call void @llvm.memmove.p0.p0.i32(ptr %p, ptr %q, i32 %n, i1 false)
  ret void
}

; CHECK-LABEL: tail_memset:
; CHECK: b memset
define void @tail_memset(ptr nocapture %p, i8 %c, i32 %n) #0 {
entry:
  tail call void @llvm.memset.p0.i32(ptr %p, i8 %c, i32 %n, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i32(ptr nocapture, ptr nocapture readonly, i32, i1) #0
declare void @llvm.memmove.p0.p0.i32(ptr nocapture, ptr nocapture readonly, i32, i1) #0
declare void @llvm.memset.p0.i32(ptr nocapture, i8, i32, i1) #0

attributes #0 = { nounwind }
