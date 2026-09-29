; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux < %s

; Check that llc does not crash due to an illegal APInt operation

define i1 @f(ptr %ptr) {
 entry:
  %val = load i8, ptr %ptr, align 8, !range !0
  %tobool = icmp eq i8 %val, 0
  ret i1 %tobool
}

!0 = !{i8 0, i8 2}
