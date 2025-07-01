# RUN: llvm-mc %s --validate-cfi --filetype=null 2>&1 \
# RUN:   | FileCheck %s
.text
        .globl  f
        .type   f, @function
f:
        .cfi_startproc
        .cfi_escape 0
        # CHECK: warning: this directive is not supported, ignoring it
.Lfunc_end0:
        .size   f, .Lfunc_end0-f
        .cfi_endproc
