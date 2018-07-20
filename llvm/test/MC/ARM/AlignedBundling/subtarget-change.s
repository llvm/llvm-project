# RUN: llvm-mc -filetype=obj -triple armv7-linux-gnueabi %s -o - \
# RUN:   | llvm-objdump -no-show-raw-insn -triple armv7 -disassemble - | FileCheck %s

        # We can switch subtargets with .arch outside of a bundle
        .syntax unified
        .text
        .bundle_align_mode 4
        .arch armv4t
        bx lr
        .bundle_lock
        and r1, r1, r1
        and r1, r1, r1
        .bundle_unlock
        bx lr

        # We can switch subtargets at the start of a bundle
        bx lr
        .bundle_lock align_to_end
        .arch armv7a
        movt r0, #0xffff
        movw r0, #0xffff
        .bundle_unlock
        bx lr

# CHECK:      0: bx    lr
# CHECK-NEXT: 4: and   r1, r1, r1
# CHECK-NEXT: 8: and   r1, r1, r1
# CHECK-NEXT: c: bx    lr
# CHECK-NEXT: 10: bx    lr
# CHECK-NEXT: 14: nop
# CHECK-NEXT: 18: movt  r0, #65535
# CHECK-NEXT: 1c: movw  r0, #65535
# CHECK-NEXT: 20: bx    lr
