# RUN: not llvm-mc %s -triple x86_64-linux -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

# FIXME: Push source locations into diagnostics.

.text
.cfi_def_cfa rsp, 8
# CHECK: error: this directive must appear between .cfi_startproc and .cfi_endproc directives

.cfi_startproc
nop

.cfi_startproc
# CHECK: error: starting new .cfi frame before finishing the previous one
nop
.cfi_endproc

.cfi_def_cfa rsp, 8
# CHECK: error: this directive must appear between .cfi_startproc and .cfi_endproc directives
