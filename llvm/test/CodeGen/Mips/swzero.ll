; RUN: llc -march=mipsel < %s | FileCheck %s

%struct.unaligned = type <{ i32 }>

define void @zero_u(%struct.unaligned* nocapture %p) nounwind {
entry:
; CHECK: swl $zero
; CHECK: swr $zero
  %x = getelementptr inbounds %struct.unaligned* %p, i32 0, i32 0
  store i32 0, i32* %x, align 1
  ret void
}

define void @zero_a(i32* nocapture %p) nounwind {
entry:
; CHECK: sw $zero
  store i32 0, i32* %p, align 4
  ret void
}

