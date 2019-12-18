; RUN: opt < %s -instcombine -S | FileCheck %s

; Remove an icmp by using its operand in the subsequent logic directly.

define i8 @zext_or_icmp_icmp(i8 %a, i8 %b) {
  %mask = and i8 %a, 1
  %toBool1 = icmp eq i8 %mask, 0
  %toBool2 = icmp eq i8 %b, 0
  %bothCond = or i1 %toBool1, %toBool2
  %zext = zext i1 %bothCond to i8
  ret i8 %zext

; CHECK-LABEL: zext_or_icmp_icmp(
; CHECK-NEXT:    %mask = and i8 %a, 1
; CHECK-NEXT:    %toBool2 = icmp eq i8 %b, 0
; CHECK-NEXT:    %toBool22 = zext i1 %toBool2 to i8
; CHECK-NEXT:    %1 = xor i8 %mask, 1
; CHECK-NEXT:    %zext3 = or i8 %1, %toBool22
; CHECK-NEXT:    ret i8 %zext
}

; Here, widening the or from i1 to i32 and removing one of the icmps would
; widen an undef value (created by the out-of-range shift), increasing the
; range of valid values for the return, so we can't do it.
define i32 @dont_widen_undef() {
entry:
  br label %block2

block1:
  br label %block2

block2:
  %m.011 = phi i32 [ 33, %entry ], [ 0, %block1 ]
  %cmp.i = icmp ugt i32 %m.011, 1
  %m.1.op = lshr i32 1, %m.011
  %sext.mask = and i32 %m.1.op, 65535
  %cmp115 = icmp ne i32 %sext.mask, 0
  %cmp1 = or i1 %cmp.i, %cmp115
  %conv2 = zext i1 %cmp1 to i32
  ret i32 %conv2

; CHECK-LABEL: dont_widen_undef(
; CHECK:         %m.011 = phi i32 [ 33, %entry ], [ 0, %block1 ]
; CHECK-NEXT:    %cmp.i = icmp ugt i32 %m.011, 1
; CHECK-NEXT:    %m.1.op = lshr i32 1, %m.011
; CHECK-NEXT:    %sext.mask = and i32 %m.1.op, 65535
; CHECK-NEXT:    %cmp115 = icmp ne i32 %sext.mask, 0
; CHECK-NEXT:    %cmp1 = or i1 %cmp.i, %cmp115
; CHECK-NEXT:    %conv2 = zext i1 %cmp1 to i32
; CHECK-NEXT:    ret i32 %conv2
}
