; RUN: llc -mtriple thumbv7-windows-itanium -relocation-model pic -filetype asm -o - %s \
; RUN:   | FileCheck %s -check-prefix CHECK-WIN

; RUN: llc -mtriple thumbv7-windows-gnu -relocation-model pic -filetype asm -o - %s \
; RUN:   | FileCheck %s -check-prefix CHECK-GNU

@external = external global i8

define arm_aapcs_vfpcc i8 @return_external() {
entry:
  %0 = load i8, i8* @external, align 1
  ret i8 %0
}

; CHECK-WIN-LABEL: return_external
; CHECK-WIN: movw r0, :lower16:external
; CHECK-WIN: movt r0, :upper16:external
; CHECK-WIN: ldrb r0, [r0]

; CHECK-GNU-LABEL: return_external
; CHECK-GNU: movw r0, :lower16:.refptr.external
; CHECK-GNU: movt r0, :upper16:.refptr.external
; CHECK-GNU: ldr  r0, [r0]
; CHECK-GNU: ldrb r0, [r0]
