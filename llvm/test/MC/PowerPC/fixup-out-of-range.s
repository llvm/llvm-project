# RUN: not llvm-mc -triple powerpc64le-unknown-unknown -filetype=obj %s 2>&1 >/dev/null | FileCheck %s

## Each diagnostic names the branch's own position in its section, and -- where
## the fixup still refers to a symbol -- the target it was trying to reach.
##
## The `a' forms below are written `sym-.', a difference of two symbols in the
## same section. That folds to an absolute constant before the fixup is applied,
## so no symbol survives in the MCValue and there is nothing to name; those
## diagnostics carry the position only. The plain forms keep their symbol and
## report both. The position is the useful half for generated code, which has no
## source location at all.

# CHECK: error: branch target out of range (32772 not between -32768 and 32764) to 'brcond14_target' at .text+0x0
brcond14_out_of_range_hi:
    beq 0, brcond14_target
    .space 0x8000

brcond14_target:
    blr

# CHECK: error: branch target out of range (-32772 not between -32768 and 32764) to 'brcond14_out_of_range_lo' at .text+0x1000c
brcond14_out_of_range_lo:
    .space 0x8004
    beq 0, brcond14_out_of_range_lo

# CHECK: error: branch target not a multiple of four (5) to 'brcond14_misaligned_target' at .text+0x10010
brcond14_misaligned:
    beq 0, brcond14_misaligned_target
    .byte 0

brcond14_misaligned_target:
    blr



# CHECK: error: branch target out of range (32772 not between -32768 and 32764) at .text+0x10019
brcond14abs_out_of_range_hi:
    beqa 0, brcond14abs_target-.
    .space 0x8000

brcond14abs_target:
    blr

# CHECK: error: branch target out of range (-32772 not between -32768 and 32764) at .text+0x20025
brcond14abs_out_of_range_lo:
    .space 0x8004
    beqa 0, brcond14abs_out_of_range_lo-.

# CHECK: error: branch target not a multiple of four (5) at .text+0x20029
brcond14abs_misaligned:
    beqa 0, brcond14abs_misaligned_target-.
    .byte 0

brcond14abs_misaligned_target:
    blr



# CHECK: error: branch target out of range (33554436 not between -33554432 and 33554428) to 'br24_target' at .text+0x20032
br24_out_of_range_hi:
    b br24_target
    .space 0x2000000

br24_target:
    blr

# CHECK: error: branch target out of range (-33554436 not between -33554432 and 33554428) to 'br24_out_of_range_lo' at .text+0x402003e
br24_out_of_range_lo:
    .space 0x2000004
    b br24_out_of_range_lo

# CHECK: error: branch target not a multiple of four (5) to 'br24_misaligned_target' at .text+0x4020042
br24_misaligned:
    b br24_misaligned_target
    .byte 0

br24_misaligned_target:
    blr



# CHECK: error: branch target out of range (33554436 not between -33554432 and 33554428) at .text+0x402004b
br24abs_out_of_range_hi:
    ba br24abs_target-.
    .space 0x2000000

br24abs_target:
    blr

# CHECK: error: branch target out of range (-33554436 not between -33554432 and 33554428) at .text+0x8020057
br24abs_out_of_range_lo:
    .space 0x2000004
    ba br24abs_out_of_range_lo-.

# CHECK: error: branch target not a multiple of four (5) at .text+0x802005b
br24abs_misaligned:
    ba br24abs_misaligned_target-.
    .byte 0

br24abs_misaligned_target:
    blr
