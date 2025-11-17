# RUN: llvm-mc -triple x86_64-unknown-unknown %s --validate-cfi --filetype=null 2>&1 | FileCheck %s --allow-empty --implicit-check-not warning: --implicit-check-not error:
## TODO: `--allow-empty` should be erased and replaced with a simple check for the asm output when `--filetype=asm` is implemented for `--validate-cfi`.
# RUN: llvm-mc -triple x86_64-unknown-unknown %s --filetype=asm 2>&1 | FileCheck %s -check-prefix=ASSEMBLER --implicit-check-not warning: --implicit-check-not error:
# ASSEMBLER: .text
.text
        .globl  f
        .type   f, @function
f:
        .cfi_startproc
.Lfunc_end0:
        .size   f, .Lfunc_end0-f
        .cfi_endproc
