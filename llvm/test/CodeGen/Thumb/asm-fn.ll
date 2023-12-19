; RUN: not llc -mtriple=thumbv6m-none-unknown-eabi < %s 2>&1 | FileCheck %s

; CHECK: error: symbol '__aeabi_uidivmod' is already defined

module asm ".global __aeabi_uidivmod"
module asm ".type __aeabi_uidivmod, %function"
module asm "__aeabi_uidivmod:"
module asm "str    r0, [r2, #0x060]"
module asm "str    r1, [r2, #0x064]"

define void @__aeabi_uidivmod() #0 {
  tail call void asm sideeffect alignstack "push {lr}\0Asub sp, sp, #4\0Amov r2, sp\0Abl __udivmodsi4\0Aldr r1, [sp]\0Aadd sp, sp, #4\0Apop {pc}", "~{cc},~{memory}"()
  unreachable
}

attributes #0 = { naked }
