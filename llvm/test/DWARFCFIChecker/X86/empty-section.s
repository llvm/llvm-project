# RUN: llvm-mc %s --validate-cfi --filetype=null 2>&1 \
# RUN:   | FileCheck %s --allow-empty 
# CHECK-NOT: warning:
.text
        .globl  f
        .type   f, @function
f:
        .cfi_startproc
.Lfunc_end0:
        .size   f, .Lfunc_end0-f
        .cfi_endproc
