; RUN: llc < %s -march=x86 | FileCheck %s

define signext i16 @foo(i16 signext %x) nounwind {
entry:
; CHECK-LABEL: foo:
; CHECK: movzwl 4(%esp), %eax
; CHECK-NEXT: xorl $21998, %eax
; CHECK-NEXT: retl
  %0 = xor i16 %x, 21998
  ret i16 %0
}

define signext i16 @bar(i16 signext %x) nounwind {
entry:
; CHECK-LABEL: bar:
; CHECK: movzwl 4(%esp), %eax
; CHECK-NEXT: xorl $54766, %eax
; CHECK-NEXT: retl
  %0 = xor i16 %x, 54766
  ret i16 %0
}
