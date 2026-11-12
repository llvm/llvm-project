# RUN: not llvm-mc -filetype=obj -triple x86_64 %s -o %t 2>&1 | FileCheck %s --implicit-check-not=error:

.section foo
foo:
# CHECK: [[#@LINE-1]]:1: error: symbol 'foo' is already defined

x1:
.section x1
# CHECK: <unknown>:0: error: invalid symbol redefinition

## Equated symbol followed by .section should report an error, not crash.
x2 = 0
.section x2
# CHECK: <unknown>:0: error: invalid symbol redefinition
x2 = 0
# CHECK: [[#@LINE-1]]:6: error: redefinition of 'x2'

## A symbol equated to a register is not considered defined, but it can't be
## used as a section symbol either.
.equ x3, %esp
.pushsection x3,""
# CHECK: <unknown>:0: error: invalid symbol redefinition
.popsection
