; RUN: not llc -mtriple thumbv6m-arm-none-eabi -filetype asm -o - %s -frame-pointer=all 2>&1 | FileCheck %s --check-prefix=RESERVED-R7
; RUN: not llc -mtriple thumbv6m-arm-none-eabi -filetype asm -o - %s -frame-pointer=all -mattr=+aapcs-frame-chain 2>&1 | FileCheck %s --check-prefix=RESERVED-R11
; RUN: not llc -mtriple thumbv6m-arm-none-eabi -filetype asm -o - %s -frame-pointer=all -mattr=+aapcs-frame-chain-leaf 2>&1 | FileCheck %s --check-prefix=RESERVED-R11
; RUN: llc -mtriple thumbv6m-arm-none-eabi -filetype asm -o - %s -frame-pointer=non-leaf 2>&1 | FileCheck %s --check-prefix=RESERVED-NONE
; RUN: not llc -mtriple thumbv6m-arm-none-eabi -filetype asm -o - %s -frame-pointer=non-leaf -mattr=+aapcs-frame-chain 2>&1 | FileCheck %s --check-prefix=RESERVED-R11
; RUN: not llc -mtriple thumbv6m-arm-none-eabi -filetype asm -o - %s -frame-pointer=non-leaf -mattr=+aapcs-frame-chain-leaf 2>&1 | FileCheck %s --check-prefix=RESERVED-R11
; RUN: llc -mtriple thumbv6m-arm-none-eabi -filetype asm -o - %s -frame-pointer=none 2>&1 | FileCheck %s --check-prefix=RESERVED-NONE
; RUN: not llc -mtriple thumbv6m-arm-none-eabi -filetype asm -o - %s -frame-pointer=none -mattr=+aapcs-frame-chain 2>&1 | FileCheck %s --check-prefix=RESERVED-R11
; RUN: not llc -mtriple thumbv6m-arm-none-eabi -filetype asm -o - %s -frame-pointer=none -mattr=+aapcs-frame-chain-leaf 2>&1 | FileCheck %s --check-prefix=RESERVED-R11

declare void @leaf(i32 %input)

define void @reserved_r7(i32 %input) {
; RESERVED-NONE-NOT: error: write to reserved register 'R7'
; RESERVED-R7: error: write to reserved register 'R7'
; RESERVED-R11-NOT: error: write to reserved register 'R7'
  %1 = call i32 asm sideeffect "mov $0, $1", "={r7},r"(i32 %input)
  ret void
}

define void @reserved_r11(i32 %input) {
; RESERVED-NONE-NOT: error: write to reserved register 'R11'
; RESERVED-R7-NOT: error: write to reserved register 'R11'
; RESERVED-R11: error: write to reserved register 'R11'
  %1 = call i32 asm sideeffect "mov $0, $1", "={r11},r"(i32 %input)
  ret void
}
