# RUN: not llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

.section __TEXT,__text
.globl _foo
_foo:
  .cfi_startproc
  subq $8, %rsp
  .cfi_adjust_cfa_offset 8
  subq $8, %rsp
  .cfi_adjust_cfa_offset 8

tmp0: # non-private label cannot appear here
  addq $8, %rsp
# CHECK: :[[#@LINE+1]]:3: error: invalid CFI advance_loc expression
  .cfi_adjust_cfa_offset -8
.tmp1: # non-private label cannot appear here
  addq $8, %rsp
# CHECK: :[[#@LINE+1]]:3: error: invalid CFI advance_loc expression
  .cfi_adjust_cfa_offset -8
  retq
  .cfi_endproc
