; RUN: llc -verify-machineinstrs -mtriple=powerpc64-bgq-linux < %s

; Check that llc does not crash due to an illegal APInt operation

define i1 @f(i8* %ptr) {
 entry:
  %val = load i8, i8* %ptr, align 8, !range !0
  %tobool = icmp eq i8 %val, 0
  ret i1 %tobool
}

!0 = !{i8 0, i8 2}
