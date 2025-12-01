# RUN: not llvm-mc -triple powerpc64le-unknown-unknown -filetype=obj %s 2>&1 >/dev/null | FileCheck %s

# CHECK: error: branch target out of range (32772 not between -32768 and 32764)
brcond14_out_of_range_hi:
    beq 0, brcond14_target
    .space 0x8000

brcond14_target:
    blr

# CHECK: error: branch target out of range (-32772 not between -32768 and 32764)
brcond14_out_of_range_lo:
    .space 0x8004
    beq 0, brcond14_out_of_range_lo

# CHECK: error: branch target not a multiple of four (5)
brcond14_misaligned:
    beq 0, brcond14_misaligned_target
    .byte 0

brcond14_misaligned_target:
    blr



# CHECK: error: branch target out of range (32772 not between -32768 and 32764)
brcond14abs_out_of_range_hi:
    beqa 0, brcond14abs_target-.
    .space 0x8000

brcond14abs_target:
    blr

# CHECK: error: branch target out of range (-32772 not between -32768 and 32764)
brcond14abs_out_of_range_lo:
    .space 0x8004
    beqa 0, brcond14abs_out_of_range_lo-.

# CHECK: error: branch target not a multiple of four (5)
brcond14abs_misaligned:
    beqa 0, brcond14abs_misaligned_target-.
    .byte 0

brcond14abs_misaligned_target:
    blr



# CHECK: error: branch target out of range (33554436 not between -33554432 and 33554428)
br24_out_of_range_hi:
    b br24_target
    .space 0x2000000

br24_target:
    blr

# CHECK: error: branch target out of range (-33554436 not between -33554432 and 33554428)
br24_out_of_range_lo:
    .space 0x2000004
    b br24_out_of_range_lo

# CHECK: error: branch target not a multiple of four (5)
br24_misaligned:
    b br24_misaligned_target
    .byte 0

br24_misaligned_target:
    blr



# CHECK: error: branch target out of range (33554436 not between -33554432 and 33554428)
br24abs_out_of_range_hi:
    ba br24abs_target-.
    .space 0x2000000

br24abs_target:
    blr

# CHECK: error: branch target out of range (-33554436 not between -33554432 and 33554428)
br24abs_out_of_range_lo:
    .space 0x2000004
    ba br24abs_out_of_range_lo-.

# CHECK: error: branch target not a multiple of four (5)
br24abs_misaligned:
    ba br24abs_misaligned_target-.
    .byte 0

br24abs_misaligned_target:
    blr
