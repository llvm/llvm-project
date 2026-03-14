; RUN: llc -mtriple=x86_64-unknown-unknown < %s | FileCheck %s

; CHECK-LABEL: tail_memcpy
; CHECK: jmp memcpy
define void @tail_memcpy(ptr nocapture %p, ptr nocapture readonly %q, i32 %n) #0 {
entry:
  tail call void @llvm.memcpy.p0.p0.i32(ptr %p, ptr %q, i32 %n, i1 false)
  ret void
}

; CHECK-LABEL: tail_memmove
; CHECK: jmp memmove
define void @tail_memmove(ptr nocapture %p, ptr nocapture readonly %q, i32 %n) #0 {
entry:
  tail call void @llvm.memmove.p0.p0.i32(ptr %p, ptr %q, i32 %n, i1 false)
  ret void
}

; CHECK-LABEL: tail_memset
; CHECK: jmp memset
define void @tail_memset(ptr nocapture %p, i8 %c, i32 %n) #0 {
entry:
  tail call void @llvm.memset.p0.i32(ptr %p, i8 %c, i32 %n, i1 false)
  ret void
}

; CHECK-LABEL: tail_memcpy_ret
; CHECK: jmp memcpy
define ptr @tail_memcpy_ret(ptr nocapture %p, ptr nocapture readonly %q, i32 %n) #0 {
entry:
  tail call void @llvm.memcpy.p0.p0.i32(ptr %p, ptr %q, i32 %n, i1 false)
  ret ptr %p
}

; CHECK-LABEL: tail_memmove_ret
; CHECK: jmp memmove
define ptr @tail_memmove_ret(ptr nocapture %p, ptr nocapture readonly %q, i32 %n) #0 {
entry:
  tail call void @llvm.memmove.p0.p0.i32(ptr %p, ptr %q, i32 %n, i1 false)
  ret ptr %p
}

; CHECK-LABEL: tail_memset_ret
; CHECK: jmp memset
define ptr @tail_memset_ret(ptr nocapture %p, i8 %c, i32 %n) #0 {
entry:
  tail call void @llvm.memset.p0.i32(ptr %p, i8 %c, i32 %n, i1 false)
  ret ptr %p
}

declare void @llvm.memcpy.p0.p0.i32(ptr nocapture, ptr nocapture readonly, i32, i1) #0
declare void @llvm.memmove.p0.p0.i32(ptr nocapture, ptr nocapture readonly, i32, i1) #0
declare void @llvm.memset.p0.i32(ptr nocapture, i8, i32, i1) #0

attributes #0 = { nounwind }
