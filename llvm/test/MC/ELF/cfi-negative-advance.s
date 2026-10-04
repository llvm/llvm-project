// RUN: not llvm-mc -filetype=obj -triple=x86_64-pc-linux-gnu %s -o /dev/null 2>&1 | FileCheck %s

// A CFI location cannot move backwards when text subsections are laid out.
.text 2
.cfi_startproc
.text
// CHECK: error: invalid CFI advance_loc expression
.cfi_def_cfa_offset 8
nop
.cfi_endproc