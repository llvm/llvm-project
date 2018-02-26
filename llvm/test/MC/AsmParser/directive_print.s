# RUN: not llvm-mc -triple i386-linux-gnu %s 2> %t.err | FileCheck %s
# RUN: FileCheck < %t.err %s --check-prefix=CHECK-ERR

T1:
# CHECK: e
# CHECK: 2.718281828459045235
.print "e"
.print "2.718281828459045235"

T2:
# CHECK-ERR: expected double quoted string after .print
.altmacro
.print <pi>
.noaltmacro

T3:
# CHECK-ERR: expected end of statement
.print "a" "misplaced-string"
