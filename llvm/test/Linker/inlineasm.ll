; RUN: echo > %t.ll
; RUN: llvm-link %t.ll %s -S

; ModuleID = 'bitfield-access-2.o'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"
target triple = "i386-apple-macosx10.6.8"

; rdar://9776316 - type remapping needed for inline asm blobs.

%T = type { [18 x i32], [4 x ptr] }

define void @f(ptr %x) nounwind ssp {
entry:
call void asm sideeffect "", "=*m"(ptr elementtype(%T) %x) nounwind
unreachable
}

