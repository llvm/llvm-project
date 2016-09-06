; Test 8-bit atomic additions.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CHECK-SHIFT1
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CHECK-SHIFT2

; Check addition of a variable.
; - CHECK is for the main loop.
; - CHECK-SHIFT1 makes sure that the negated shift count used by the second
;   RLL is set up correctly.  The negation is independent of the NILL and L
;   tested in CHECK.
; - CHECK-SHIFT2 makes sure that %b is shifted into the high part of the word
;   before being used.  This shift is independent of the other loop prologue
;   instructions.
define i8 @f1(i8 *%src, i8 %b) {
; CHECK-LABEL: f1:
; CHECK: risbg %r1, %r2, 0, 189, 0{{$}}
; CHECK: sll [[SHIFT:%r[0-9]+]], 3
; CHECK: l [[OLD:%r[0-9]+]], 0(%r1)
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK: rll [[ROT:%r[0-9]+]], [[OLD]], 0([[SHIFT]])
; CHECK: ar [[ROT]], %r3
; CHECK: rll [[NEW:%r[0-9]+]], [[ROT]], 0({{%r[1-9]+}})
; CHECK: cs [[OLD]], [[NEW]], 0(%r1)
; CHECK: jl [[LABEL]]
; CHECK: rll %r2, [[OLD]], 8([[SHIFT]])
; CHECK: br %r14
;
; CHECK-SHIFT1-LABEL: f1:
; CHECK-SHIFT1: sll [[SHIFT:%r[1-9]+]], 3
; CHECK-SHIFT1: lcr [[NEGSHIFT:%r[1-9]+]], [[SHIFT]]
; CHECK-SHIFT1: rll
; CHECK-SHIFT1: rll {{%r[0-9]+}}, {{%r[0-9]+}}, 0([[NEGSHIFT]])
; CHECK-SHIFT1: rll
; CHECK-SHIFT1: br %r14
;
; CHECK-SHIFT2-LABEL: f1:
; CHECK-SHIFT2: sll %r3, 24
; CHECK-SHIFT2: rll
; CHECK-SHIFT2: ar {{%r[0-9]+}}, %r3
; CHECK-SHIFT2: rll
; CHECK-SHIFT2: rll
; CHECK-SHIFT2: br %r14
  %res = atomicrmw add i8 *%src, i8 %b seq_cst
  ret i8 %res
}

; Check the minimum signed value.  We add 0x80000000 to the rotated word.
define i8 @f2(i8 *%src) {
; CHECK-LABEL: f2:
; CHECK: risbg [[RISBG:%r[1-9]+]], %r2, 0, 189, 0
; CHECK: sll %r2, 3
; CHECK: l [[OLD:%r[0-9]+]], 0([[RISBG]])
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK: rll [[ROT:%r[0-9]+]], [[OLD]], 0(%r2)
; CHECK: afi [[ROT]], -2147483648
; CHECK: rll [[NEW:%r[0-9]+]], [[ROT]], 0([[NEGSHIFT:%r[1-9]+]])
; CHECK: cs [[OLD]], [[NEW]], 0([[RISBG]])
; CHECK: jl [[LABEL]]
; CHECK: rll %r2, [[OLD]], 8([[SHIFT]])
; CHECK: br %r14
;
; CHECK-SHIFT1-LABEL: f2:
; CHECK-SHIFT1: sll [[SHIFT:%r[1-9]+]], 3
; CHECK-SHIFT1: lcr [[NEGSHIFT:%r[1-9]+]], [[SHIFT]]
; CHECK-SHIFT1: rll
; CHECK-SHIFT1: rll {{%r[0-9]+}}, {{%r[0-9]+}}, 0([[NEGSHIFT]])
; CHECK-SHIFT1: rll
; CHECK-SHIFT1: br %r14
;
; CHECK-SHIFT2-LABEL: f2:
; CHECK-SHIFT2: br %r14
  %res = atomicrmw add i8 *%src, i8 -128 seq_cst
  ret i8 %res
}

; Check addition of -1.  We add 0xff000000 to the rotated word.
define i8 @f3(i8 *%src) {
; CHECK-LABEL: f3:
; CHECK: afi [[ROT]], -16777216
; CHECK: br %r14
;
; CHECK-SHIFT1-LABEL: f3:
; CHECK-SHIFT1: br %r14
; CHECK-SHIFT2-LABEL: f3:
; CHECK-SHIFT2: br %r14
  %res = atomicrmw add i8 *%src, i8 -1 seq_cst
  ret i8 %res
}

; Check addition of 1.  We add 0x01000000 to the rotated word.
define i8 @f4(i8 *%src) {
; CHECK-LABEL: f4:
; CHECK: afi [[ROT]], 16777216
; CHECK: br %r14
;
; CHECK-SHIFT1-LABEL: f4:
; CHECK-SHIFT1: br %r14
; CHECK-SHIFT2-LABEL: f4:
; CHECK-SHIFT2: br %r14
  %res = atomicrmw add i8 *%src, i8 1 seq_cst
  ret i8 %res
}

; Check the maximum signed value.  We add 0x7f000000 to the rotated word.
define i8 @f5(i8 *%src) {
; CHECK-LABEL: f5:
; CHECK: afi [[ROT]], 2130706432
; CHECK: br %r14
;
; CHECK-SHIFT1-LABEL: f5:
; CHECK-SHIFT1: br %r14
; CHECK-SHIFT2-LABEL: f5:
; CHECK-SHIFT2: br %r14
  %res = atomicrmw add i8 *%src, i8 127 seq_cst
  ret i8 %res
}

; Check addition of a large unsigned value.  We add 0xfe000000 to the
; rotated word, expressed as a negative AFI operand.
define i8 @f6(i8 *%src) {
; CHECK-LABEL: f6:
; CHECK: afi [[ROT]], -33554432
; CHECK: br %r14
;
; CHECK-SHIFT1-LABEL: f6:
; CHECK-SHIFT1: br %r14
; CHECK-SHIFT2-LABEL: f6:
; CHECK-SHIFT2: br %r14
  %res = atomicrmw add i8 *%src, i8 254 seq_cst
  ret i8 %res
}
