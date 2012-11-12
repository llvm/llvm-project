; RUN: llc < %s -march=x86 | FileCheck %s

define signext i16 @foo(i16 signext %x) nounwind {
entry:
; CHECK: foo:
; CHECK-NOT: movzwl
; CHECK: movswl 4(%esp), %eax
; CHECK: xorl $21998, %eax
  %0 = xor i16 %x, 21998
  ret i16 %0
}

define signext i16 @bar(i16 signext %x) nounwind {
entry:
; CHECK: bar:
; CHECK-NOT: movzwl
; CHECK: movswl 4(%esp), %eax
; CHECK: xorl $-10770, %eax
  %0 = xor i16 %x, 54766
  ret i16 %0
}
