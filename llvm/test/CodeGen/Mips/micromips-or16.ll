; RUN: llc -mtriple=mipsel -mcpu=mips32r2 -mattr=+micromips \
; RUN:   -relocation-model=pic -O3 < %s | FileCheck %s
; RUN: llc -O0 -mtriple=mips -mcpu=mips32r2 -mattr=+micromips \
; RUN:   -asm-show-inst < %s | FileCheck %s

; Branch instruction added to enable FastISel::selectOperator
; to select OR instruction
define i32 @f1(i32 signext %a, i32 signext %b) {
; CHECK-LABEL: f1
; CHECK-NOT: OR16_MMR6
      %1 = or i32 %a, %b
       br label %b1
b1:
       ret i32 %1
}

define i32 @f2(i32 signext %a, i32 signext %b) {
entry:
; CHECK-LABEL: f2
; CHECK: or16
  %0 = or i32 %a, %b
  ret i32 %0
}
