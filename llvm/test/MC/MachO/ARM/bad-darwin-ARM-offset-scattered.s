@ RUN: not llvm-mc -n -triple armv7-apple-darwin10 %s -filetype=obj -o - 2> %t.err > %t
@ RUN: FileCheck --check-prefix=CHECK-ERROR < %t.err %s

.text
.space 0x1029eb8

fn:
    movw  r0, :lower16:(fn2-L1)
    andeq r0, r0, r0
L1:
    andeq r0, r0, r0

fn2:

@ CHECK-ERROR: error: can not encode offset '0x1029EB8' in resulting scattered relocation.
