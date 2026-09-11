## FileCheck reads the directives from the split input so @LINE matches the
## line numbers llvm-mc reports (split-file restarts numbering per file).
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: not llvm-mc a.s -triple x86_64 2>&1 | FileCheck a.s
# RUN: not llvm-mc a.s -triple x86_64 -filetype=obj -o /dev/null 2>&1 | FileCheck a.s
# RUN: not llvm-mc b.s -triple x86_64 2>&1 | FileCheck b.s

#--- a.s
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

## Check we don't crash on unclosed frame scope.
.globl foo
foo:
 .cfi_startproc
# CHECK: [[#@LINE+1]]:1: error: Unfinished frame!
#--- b.s
## A .cfi_startproc in .text.qux is left unfinished while a later frame is
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
