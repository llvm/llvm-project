# RUN: not llvm-mc %s -filetype=obj -triple=x86_64-unknown-linux \
# RUN:   -o /dev/null 2>&1 | FileCheck %s

## Check we don't crash on unclosed frame scope.
# CHECK: error: Unfinished frame!

.text
.globl foo
foo:
 .cfi_startproc
