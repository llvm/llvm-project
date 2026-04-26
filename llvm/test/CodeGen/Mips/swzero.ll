; RUN: llc -mtriple=mipsel < %s | FileCheck %s

%struct.unaligned = type <{ i32 }>

define void @zero_u(ptr nocapture %p) nounwind {
entry:
; CHECK: swl $zero
; CHECK: swr $zero
  store i32 0, ptr %p, align 1
  ret void
}

define void @zero_a(ptr nocapture %p) nounwind {
entry:
; CHECK: sw $zero
  store i32 0, ptr %p, align 4
  ret void
}

