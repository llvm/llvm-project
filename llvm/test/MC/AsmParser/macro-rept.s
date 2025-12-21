# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -triple=x86_64 a.s | FileCheck %s

#--- a.s
.rept 2
    .long 1
.endr
# 3 "a.s"
## Test line marker after .endr \n.

.rept 3
.rept 2
    .long 0
.endr
.endr # comment after .endr
.long 42

# CHECK:      .long 1
# CHECK-NEXT: .long 1

# CHECK:      .long 0
# CHECK-NEXT: .long 0
# CHECK-NEXT: .long 0
# CHECK-NEXT: .long 0
# CHECK-NEXT: .long 0
# CHECK-NEXT: .long 0
# CHECK-NEXT: .long 42

# RUN: not llvm-mc -triple=x86_64 err1.s 2>&1 | FileCheck %s --check-prefix=ERR1
# ERR1: .s:1:6: error: unmatched '.endr' directive
#--- err1.s
.endr

# RUN: not llvm-mc -triple=x86_64 err2.s 2>&1 | FileCheck %s --check-prefix=ERR2
# ERR2: .s:1:1: error: no matching '.endr' in definition
#--- err2.s
.rept 3
.long

# RUN: not llvm-mc -triple=x86_64 err3.s 2>&1 | FileCheck %s --check-prefix=ERR3
# ERR3: .s:3:7: error: expected newline
#--- err3.s
.rept 1
.long 0
.endr ab
