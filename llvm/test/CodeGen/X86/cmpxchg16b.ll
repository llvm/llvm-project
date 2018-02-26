; RUN: llc < %s -mtriple=x86_64-- -mcpu=core2 | FileCheck %s

; Basic 128-bit cmpxchg
define void @t1(i128* nocapture %p) nounwind ssp {
entry:
; CHECK: movl	$1, %ebx
; CHECK: lock cmpxchg16b
  %r = cmpxchg i128* %p, i128 0, i128 1 seq_cst seq_cst
  ret void
}

; FIXME: Handle 128-bit atomicrmw/load atomic/store atomic
