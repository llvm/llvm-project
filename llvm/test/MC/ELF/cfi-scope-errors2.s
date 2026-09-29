# Test for D51695 ensuring there is no crash when two .cfi_startproc are opened
# without the first one being closed.

# RUN: not llvm-mc %s -filetype=obj -triple=x86_64-unknown-linux -o /dev/null 2>&1 | FileCheck %s

.text
 .cfi_startproc

.text
 .cfi_startproc
# CHECK: [[#@LINE]]:1: error: starting new .cfi frame before finishing the previous one

 .cfi_endproc
