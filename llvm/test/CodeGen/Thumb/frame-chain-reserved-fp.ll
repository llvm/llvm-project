; RUN: not llc -mtriple thumbv6m-arm-none-eabi -filetype asm -o - %s -frame-pointer=all                                2>&1 | FileCheck %s --check-prefix=R7-RESERVED --check-prefix=R11-FREE
; RUN: not llc -mtriple thumbv6m-arm-none-eabi -filetype asm -o - %s -frame-pointer=all      -mattr=+aapcs-frame-chain 2>&1 | FileCheck %s --check-prefix=R7-FREE     --check-prefix=R11-RESERVED
; RUN: not llc -mtriple thumbv6m-arm-none-eabi -filetype asm -o - %s -frame-pointer=non-leaf                           2>&1 | FileCheck %s --check-prefix=R7-RESERVED --check-prefix=R11-FREE
; RUN: not llc -mtriple thumbv6m-arm-none-eabi -filetype asm -o - %s -frame-pointer=non-leaf -mattr=+aapcs-frame-chain 2>&1 | FileCheck %s --check-prefix=R7-FREE     --check-prefix=R11-RESERVED
; RUN:     llc -mtriple thumbv6m-arm-none-eabi -filetype asm -o - %s -frame-pointer=none                               2>&1 | FileCheck %s --check-prefix=R7-FREE     --check-prefix=R11-FREE
; RUN:     llc -mtriple thumbv6m-arm-none-eabi -filetype asm -o - %s -frame-pointer=none     -mattr=+aapcs-frame-chain 2>&1 | FileCheck %s --check-prefix=R7-FREE     --check-prefix=R11-FREE
; RUN: not llc -mtriple thumbv6m-arm-none-eabi -filetype asm -o - %s -frame-pointer=reserved                           2>&1 | FileCheck %s --check-prefix=R7-RESERVED --check-prefix=R11-FREE
; RUN: not llc -mtriple thumbv6m-arm-none-eabi -filetype asm -o - %s -frame-pointer=reserved -mattr=+aapcs-frame-chain 2>&1 | FileCheck %s --check-prefix=R7-FREE     --check-prefix=R11-RESERVED

declare void @leaf(i32 %input)

define void @reserved_r7(i32 %input) {
; R7-RESERVED: error: write to reserved register 'R7'
; R7-FREE-NOT: error: write to reserved register 'R7'
  %1 = call i32 asm sideeffect "mov $0, $1", "={r7},r"(i32 %input)
  ret void
}

define void @reserved_r11(i32 %input) {
; R11-RESERVED: error: write to reserved register 'R11'
; R11-FREE-NOT: error: write to reserved register 'R11'
  %1 = call i32 asm sideeffect "mov $0, $1", "={r11},r"(i32 %input)
  ret void
}
