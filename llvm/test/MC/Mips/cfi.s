# RUN: llvm-mc %s -triple=mips-unknown-unknown -show-encoding -mcpu=mips32 | \
# RUN:     FileCheck %s
# RUN: llvm-mc %s -triple=mips64-unknown-unknown -show-encoding -mcpu=mips64 | \
# RUN:     FileCheck %s

# Check that we can accept register names in CFI directives and that they are
# canonicalised to their DWARF register numbers.

        .cfi_startproc         # CHECK: .cfi_startproc
        .cfi_register   $6, $5 # CHECK: .cfi_register 6, 5
        .cfi_def_cfa    $fp, 8 # CHECK: .cfi_def_cfa 30, 8
        .cfi_def_cfa    $2, 16 # CHECK: .cfi_def_cfa 2, 16
        .cfi_endproc           # CHECK: .cfi_endproc
