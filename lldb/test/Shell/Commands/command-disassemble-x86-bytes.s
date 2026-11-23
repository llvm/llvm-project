# REQUIRES: x86

# This test verifies that disassemble -b prints out the correct bytes and
# format for x86_64 instructions of various sizes, and that an unknown
# instruction shows the opcode and disassembles as "<unknown>"

# RUN: llvm-mc -filetype=obj --triple=x86_64-unknown-unknown %s -o %t
# RUN: %lldb -b %t -o "disassemble -b -n main" | FileCheck %s

main:                                   # @main
	subq   $0x18, %rsp
	movl   $0x0, 0x14(%rsp)
	movq   %rdx, 0x8(%rsp)
	movl   %ecx, 0x4(%rsp)
	movl   (%rsp), %eax
        addq   $0x18, %rsp
	retq
        .byte  0x6 

# CHECK: [0x0] <+0>:   48 83 ec 18              subq   $0x18, %rsp
# CHECK-NEXT: [0x4] <+4>:   c7 44 24 14 00 00 00 00  movl   $0x0, 0x14(%rsp)
# CHECK-NEXT: [0xc] <+12>:  48 89 54 24 08           movq   %rdx, 0x8(%rsp)
# CHECK-NEXT: [0x11] <+17>: 89 4c 24 04              movl   %ecx, 0x4(%rsp)
# CHECK-NEXT: [0x15] <+21>: 8b 04 24                 movl   (%rsp), %eax
# CHECK-NEXT: [0x18] <+24>: 48 83 c4 18              addq   $0x18, %rsp
# CHECK-NEXT: [0x1c] <+28>: c3                       retq
# CHECK-NEXT: [0x1d] <+29>: 06                       <unknown>

