# RUN: llvm-mc -triple=arm64-none-linux-gnu -filetype=obj -o %t %s
# RUN: llvm-rtdyld -triple=arm64-none-linux-gnu -verify -check=%s %t

.globl _main
.weak _label1

.section .text.label1,"ax"
_label1:
        nop

.section .text.main,"ax"
_main:
        b _label1

# Branch must be to stub in .text.main, *not* back to _label1, because
# in general sections could be loaded at arbitrary addresses in target memory,
# and when initially processing locations and generating stubs we don't know
# the final layout yet, so we can't tell if the branch offset is within range.

# rtdyld-check: *{4}(_main) = 0x14000001
