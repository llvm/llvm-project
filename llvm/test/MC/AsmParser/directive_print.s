# RUN: llvm-mc -triple i386 %s | FileCheck %s
# RUN: not llvm-mc -triple i386 --defsym ERR=1 %s 2>&1 | FileCheck %s --check-prefix=ERR

T1:
# CHECK: e
# CHECK: 2.718281828459045235
.print "e"
.print "2.718281828459045235"

.ifdef ERR
# CHECK-ERR: :[[#@LINE+2]]:8: expected double quoted string after .print
.altmacro
.print <pi>
.noaltmacro

# ERR: :[[#@LINE+1]]:12: error: expected newline
.print "a" "misplaced-string"
.endif
