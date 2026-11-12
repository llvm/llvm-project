# RUN: llvm-mc -triple=aarch64-linux-gnu -filetype=obj -o %t.o %s
# RUN: llvm-jitlink -noexec \
# RUN:   -slab-allocate 64Kb -slab-page-size 16384 \
# RUN:   -slab-address 0x10000000 -entry=call_addend \
# RUN:   -abs target=0x10080000 -check=%s %t.o
# RUN: not llvm-jitlink -noexec \
# RUN:   -slab-allocate 64Kb -slab-page-size 16384 \
# RUN:   -slab-address 0x10000000 -entry=call_addend \
# RUN:   -abs target=0x20000000 %t.o 2>&1 | FileCheck %s

# Branch addends apply to the destination, rather than an offset into a stub.
# In-range destinations use direct branches; out-of-range ones report an error.
# CHECK: (target:0x20000000 + 0x4) is out of range of Branch26PCRel fixup

        .text
        .p2align 2

# jitlink-check: decode_operand(call_addend, 0) = \
# jitlink-check:   ((target + 4) - call_addend)[27:2]
        .globl call_addend
call_addend:
        bl target + 4
        .size call_addend, .-call_addend

# jitlink-check: decode_operand(jump_addend, 0) = \
# jitlink-check:   ((target + 4) - jump_addend)[27:2]
        .globl jump_addend
jump_addend:
        b target + 4
        .size jump_addend, .-jump_addend

# jitlink-check: decode_operand(call_negative_addend, 0) = \
# jitlink-check:   ((target - 4) - call_negative_addend)[27:2]
        .globl call_negative_addend
call_negative_addend:
        bl target - 4
        .size call_negative_addend, .-call_negative_addend
