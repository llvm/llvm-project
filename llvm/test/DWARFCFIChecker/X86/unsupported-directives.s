# RUN: llvm-mc -triple x86_64-unknown-unknown %s --validate-cfi --filetype=null 2>&1 | FileCheck %s
# RUN: llvm-mc -triple x86_64-unknown-unknown %s --filetype=asm 2>&1 | FileCheck %s -check-prefix=ASSEMBLER --implicit-check-not warning: --implicit-check-not error:
# ASSEMBLER: .text
.text
        .globl  f
        .type   f, @function
f:
        .cfi_startproc
        
        .cfi_remember_state
        # CHECK: warning: this directive is not supported, ignoring it
        .cfi_escape 0
        # CHECK: warning: this directive is not supported, ignoring it
        .cfi_restore %rsp
        # CHECK: warning: this directive behavior depends on the assembler, ignoring it
        .cfi_restore_state
        # CHECK: warning: this directive is not supported, ignoring it
.Lfunc_end0:
        .size   f, .Lfunc_end0-f
        .cfi_endproc
