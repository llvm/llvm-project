# RUN: not llvm-mc -triple=ve -filetype=obj %s -o /dev/null 2>&1 | \
# RUN:     FileCheck %s

.data
a:
.2byte 0xff5588
.4byte 0xff5588aade
.8byte 0xff5588aadeadbeafde
.byte 0xff55
.short 0xff5588
.word 0xff5588aaff
.int 0xff5588aaff
.long 0xff5588aadeadbeafde
.quad 0xff5588aadeadbeafde
.llong 0xff5588aadeadbeafde

# CHECK:      data-size-error.s:6:8: error: out of range literal value
# CHECK-NEXT: .2byte 0xff5588
# CHECK:      data-size-error.s:7:8: error: out of range literal value
# CHECK-NEXT: .4byte 0xff5588aade
# CHECK:      data-size-error.s:8:8: error: literal value out of range for directive
# CHECK-NEXT: .8byte 0xff5588aadeadbeafde
# CHECK:      data-size-error.s:9:7: error: out of range literal value
# CHECK-NEXT: .byte 0xff55
# CHECK:      data-size-error.s:10:8: error: out of range literal value
# CHECK-NEXT: .short 0xff5588
# CHECK:      data-size-error.s:11:1: error: value evaluated as 1096651680511 is out of range.
# CHECK-NEXT: .word 0xff5588aaff
# CHECK:      data-size-error.s:12:6: error: out of range literal value
# CHECK-NEXT: .int 0xff5588aaff
# CHECK:      data-size-error.s:13:7: error: literal value out of range for directive
# CHECK-NEXT: .long 0xff5588aadeadbeafde
# CHECK:      data-size-error.s:14:7: error: literal value out of range for directive
# CHECK-NEXT: .quad 0xff5588aadeadbeafde
# CHECK:      data-size-error.s:15:8: error: literal value out of range for directive
# CHECK-NEXT: .llong 0xff5588aadeadbeafde

# CHECK: [[#@LINE+1]]:17: error: unknown token in expression
.word 0xd0bb1e +

# CHECK: [[#@LINE+1]]:16: error: unexpected token
.long 0xd0bb1e =

# CHECK: [[#@LINE+1]]:10: error: unexpected token
.llong 2 0xd0bb1e
