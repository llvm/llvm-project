# RUN: not llvm-mc %s -triple x86_64-linux -o /dev/null 2>&1 | FileCheck %s
# RUN: not llvm-mc %s -triple x86_64-linux -filetype=obj -o /dev/null 2>&1 | FileCheck %s

.text
.cfi_def_cfa rsp, 8
# CHECK: [[#@LINE-1]]:1: error: this directive must appear between .cfi_startproc and .cfi_endproc directives

.cfi_startproc
nop

## This tests source location correctness as well as the error and it not crashing.
# CHECK: [[#@LINE+2]]:1: error: starting new .cfi frame before finishing the previous one
.cfi_startproc

nop
.cfi_endproc

.cfi_def_cfa rsp, 8
# CHECK: [[#@LINE-1]]:1: error: this directive must appear between .cfi_startproc and .cfi_endproc directives

## The .cfi_startproc in .text.qux is left unfinished while a later frame is
## properly closed, so DwarfFrameInfos.back() has a valid End and the
## unfinished frame is only visible through the frame stack. Check we
## diagnose it instead of crashing.
## https://github.com/llvm/llvm-project/issues/177852
.pushsection .text.qux, "ax", @progbits
.cfi_startproc
.popsection

.cfi_startproc
.cfi_endproc
# CHECK: [[#@LINE+1]]:1: error: Unfinished frame!
