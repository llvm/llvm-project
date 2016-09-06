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
; CHECK-NEXT:    %zext = or i8 %1, %toBool22
; CHECK-NEXT:    ret i8 %zext
}

