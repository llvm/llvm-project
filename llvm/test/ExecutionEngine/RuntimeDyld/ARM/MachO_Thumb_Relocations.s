# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=thumbv7s-apple-ios7.0.0 -filetype=obj -o %t/MachO_Thumb.o %s
# RUN: llvm-rtdyld -triple=thumbv7s-apple-ios7.0.0 -verify -check=%s %t/MachO_Thumb.o

        .section        __TEXT,__text,regular,pure_instructions
        .syntax unified

        .globl  thumb_caller_thumb_callee
        .p2align        1
        .code   16
        .thumb_func     thumb_caller_thumb_callee
thumb_caller_thumb_callee:
        nop

        .globl  arm_caller_thumb_callee
        .p2align        1
        .code   16
        .thumb_func     arm_caller_thumb_callee
arm_caller_thumb_callee:
        nop

        .globl  thumb_caller_arm_callee
        .p2align        1
        .code   32
thumb_caller_arm_callee:
        nop

        .globl  thumb_caller
        .p2align        1
        .code   16
        .thumb_func     thumb_caller
thumb_caller:
        nop

# Check that stubs for thumb callers use thumb code (not arm), and that thumb
# callees have the low bit set on their addresses.
#
# rtdyld-check: *{4}(stub_addr(MachO_Thumb.o, __text, thumb_caller_thumb_callee)) = 0xf000f8df
# rtdyld-check: *{4}(stub_addr(MachO_Thumb.o, __text, thumb_caller_thumb_callee) + 4) = (thumb_caller_thumb_callee | 0x1)
        bl thumb_caller_thumb_callee

# Check that arm callees do not have the low bit set on their addresses.
#
# rtdyld-check: *{4}(stub_addr(MachO_Thumb.o, __text, thumb_caller_arm_callee)) = 0xf000f8df
# rtdyld-check: *{4}(stub_addr(MachO_Thumb.o, __text, thumb_caller_arm_callee) + 4) = thumb_caller_arm_callee
        bl thumb_caller_arm_callee

        .globl  arm_caller
        .p2align        2
        .code   32
arm_caller:
        nop

# Check that stubs for arm callers use arm code (not thumb), and that thumb
# callees have the low bit set on their addresses.
# rtdyld-check: *{4}(stub_addr(MachO_Thumb.o, __text, arm_caller_thumb_callee)) = 0xe51ff004
# rtdyld-check: *{4}(stub_addr(MachO_Thumb.o, __text, arm_caller_thumb_callee) + 4) = (arm_caller_thumb_callee | 0x1)
        bl      arm_caller_thumb_callee
        nop

.subsections_via_symbols
