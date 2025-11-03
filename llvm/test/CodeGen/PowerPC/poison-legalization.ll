; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu  | FileCheck %s

define void @ExpandFloat(ptr %p1 )  {
; CHECK:      stfd 0, 8(3)
; CHECK-NEXT: stfd 0, 0(3)
; CHECK-NEXT: blr
entry:
   store volatile ppc_fp128 poison, ptr %p1
   ret void
}

