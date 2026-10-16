# RUN: llvm-mc -triple x86_64-unknown-unknown %s --validate-cfi --filetype=null 2>&1 | FileCheck %s --allow-empty --implicit-check-not warning: --implicit-check-not error:
## TODO: `--allow-empty` should be erased and replaced with a simple check for the asm output when `--filetype=asm` is implemented for `--validate-cfi`.
# RUN: llvm-mc -triple x86_64-unknown-unknown %s --filetype=asm 2>&1 | FileCheck %s -check-prefix=ASSEMBLER --implicit-check-not warning: --implicit-check-not error:
# ASSEMBLER: .section
.pushsection A
f: 
.cfi_startproc
.pushsection B
g: 
.cfi_startproc
ret
.cfi_endproc
.popsection
ret
.cfi_endproc
.popsection
