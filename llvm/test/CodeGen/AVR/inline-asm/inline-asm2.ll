; RUN: llc < %s -mtriple=avr -no-integrated-as | FileCheck %s

; CHECK-LABEL: foo
define void @foo(i16 %a) {
  call void asm sideeffect "add $0, $0", "Z"(i16 %a) nounwind
  ret void
}

