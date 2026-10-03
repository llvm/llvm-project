# RUN: llvm-mc -triple riscv32 -verify < %s

mret 0x10 # expected-error:6 {{unexpected extra operand for instruction}}

sfence.vma zero, a1, a2 # expected-error:22 {{unexpected extra operand for instruction}}

sfence.vma a0, 0x10
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:16 {{unexpected extra operand for instruction}}
# expected-note@-3:16 {{register must be a GPR}}

sinval.vma zero, a1, a2 # expected-error:1 {{invalid instruction}}

sinval.vma a0, 0x10 # expected-error:1 {{invalid instruction}}

sfence.w.inval 0x10 # expected-error:1 {{invalid instruction}}

sfence.inval.ir 0x10 # expected-error:1 {{invalid instruction}}

hfence.vvma zero, a1, a2 # expected-error:1 {{instruction requires the following: 'H' (Hypervisor)}}

hfence.vvma a0, 0x10 # expected-error:1 {{instruction requires the following: 'H' (Hypervisor)}}

hfence.gvma zero, a1, a2 # expected-error:1 {{instruction requires the following: 'H' (Hypervisor)}}

hfence.gvma a0, 0x10 # expected-error:1 {{instruction requires the following: 'H' (Hypervisor)}}

hinval.vvma zero, a1, a2 # expected-error:1 {{invalid instruction}}

hinval.vvma a0, 0x10 # expected-error:1 {{invalid instruction}}

hinval.gvma zero, a1, a2 # expected-error:1 {{invalid instruction}}

hinval.gvma a0, 0x10 # expected-error:1 {{invalid instruction}}

hlv.b a0, 0x10 # expected-error:16 {{expected '(' after optional integer offset}}

hlv.b a0, a1 # expected-error:11 {{expected '(' or optional integer offset}}

hlv.b a0, 1(a1) # expected-error:11 {{optional integer offset must be 0}}

hlv.bu a0, 0x10 # expected-error:17 {{expected '(' after optional integer offset}}

hlv.bu a0, a1 # expected-error:12 {{expected '(' or optional integer offset}}

hlv.bu a0, 1(a1) # expected-error:12 {{optional integer offset must be 0}}

hlv.h a0, 0x10 # expected-error:16 {{expected '(' after optional integer offset}}

hlv.h a0, a1 # expected-error:11 {{expected '(' or optional integer offset}}

hlv.h a0, 1(a1) # expected-error:11 {{optional integer offset must be 0}}

hlv.hu a0, 0x10 # expected-error:17 {{expected '(' after optional integer offset}}

hlv.hu a0, a1 # expected-error:12 {{expected '(' or optional integer offset}}

hlv.hu a0, 1(a1) # expected-error:12 {{optional integer offset must be 0}}

hlvx.hu a0, 0x10 # expected-error:18 {{expected '(' after optional integer offset}}

hlvx.hu a0, a1 # expected-error:13 {{expected '(' or optional integer offset}}

hlvx.hu a0, 1(a1) # expected-error:13 {{optional integer offset must be 0}}

hlv.w a0, 0x10 # expected-error:16 {{expected '(' after optional integer offset}}

hlv.w a0, a1 # expected-error:11 {{expected '(' or optional integer offset}}

hlv.w a0, 1(a1) # expected-error:11 {{optional integer offset must be 0}}

hlvx.wu a0, 0x10 # expected-error:18 {{expected '(' after optional integer offset}}

hlvx.wu a0, a1 # expected-error:13 {{expected '(' or optional integer offset}}

hlvx.wu a0, 1(a1) # expected-error:13 {{optional integer offset must be 0}}

hsv.b a0, 0x10 # expected-error:16 {{expected '(' after optional integer offset}}

hsv.b a0, a1 # expected-error:11 {{expected '(' or optional integer offset}}

hsv.b a0, 1(a1) # expected-error:11 {{optional integer offset must be 0}}

hsv.h a0, 0x10 # expected-error:16 {{expected '(' after optional integer offset}}

hsv.h a0, a1 # expected-error:11 {{expected '(' or optional integer offset}}

hsv.h a0, 1(a1) # expected-error:11 {{optional integer offset must be 0}}

hsv.w a0, 0x10 # expected-error:16 {{expected '(' after optional integer offset}}

hsv.w a0, a1 # expected-error:11 {{expected '(' or optional integer offset}}

hsv.w a0, 1(a1) # expected-error:11 {{optional integer offset must be 0}}
