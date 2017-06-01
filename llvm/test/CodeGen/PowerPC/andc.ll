; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-apple-darwin | FileCheck %s

define i1 @and_cmp1(i32 %x, i32 %y) {
; CHECK-LABEL: and_cmp1:
; CHECK: andc [[REG1:r[0-9]+]], r4, r3
; CHECK: cntlzw   [[REG2:r[0-9]+]], [[REG1]]
; CHECK: rlwinm r3, [[REG2]], 27, 31, 31
; CHECK: blr

  %and = and i32 %x, %y
  %cmp = icmp eq i32 %and, %y
  ret i1 %cmp
}

define i1 @and_cmp_const(i32 %x) {
; CHECK-LABEL: and_cmp_const:
; CHECK: li [[REG1:r[0-9]+]], 43
; CHECK: andc [[REG2:r[0-9]+]], [[REG1]], r3
; CHECK: cntlzw   [[REG3:r[0-9]+]], [[REG2]]
; CHECK: rlwinm r3, [[REG3]], 27, 31, 31
; CHECK: blr

  %and = and i32 %x, 43
  %cmp = icmp eq i32 %and, 43
  ret i1 %cmp
}

define i1 @foo(i32 %i) {
; CHECK-LABEL: foo:
; CHECK: lis [[REG1:r[0-9]+]], 4660
; CHECK: ori [[REG2:r[0-9]+]], [[REG1]], 22136
; CHECK: andc [[REG3:r[0-9]+]], [[REG2]], r3
; CHECK: cntlzw  [[REG4:r[0-9]+]], [[REG3]]
; CHECK: rlwinm r3, [[REG4]], 27, 31, 31
; CHECK: blr

  %and = and i32 %i, 305419896
  %cmp = icmp eq i32 %and, 305419896
  ret i1 %cmp
}

