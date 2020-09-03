# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple x86_64-pc-win32 %s -o %t
# RUN: llvm-readobj %t --cg-profile | FileCheck %s

# CHECK:      CGProfile [
# CHECK-NEXT:   CGProfileEntry {
# CHECK-NEXT:     From: a (10)
# CHECK-NEXT:     To: b (11)
# CHECK-NEXT:     Weight: 32
# CHECK-NEXT:   }
# CHECK-NEXT:   CGProfileEntry {
# CHECK-NEXT:     From: c (12)
# CHECK-NEXT:     To: a (10)
# CHECK-NEXT:     Weight: 11
# CHECK-NEXT:   }
# CHECK-NEXT:   CGProfileEntry {
# CHECK-NEXT:     From: d (13)
# CHECK-NEXT:     To: e (14)
# CHECK-NEXT:     Weight: 20
# CHECK-NEXT:   }
# CHECK-NEXT: ]

.section .test
a:
b:
c:
d:
e:

.section ".llvm.call-graph-profile"
    .long 10    ## Symbol index of a.
    .long 11    ## Symbol index of b.
    .quad 32    ## Weight from a to b.

    .long 12    ## Symbol index of c.
    .long 10    ## Symbol index of a.
    .quad 11    ## Weight from c to a.

    .long 13    ## Symbol index of d.
    .long 14    ## Symbol index of e.
    .quad 20    ## Weight from d to e.
