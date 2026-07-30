; Test 16-bit atomic subtractions.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CHECK-SHIFT1
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CHECK-SHIFT2

; Check subtraction of a variable.
; - CHECK is for the main loop.
; - CHECK-SHIFT1 makes sure that the negated shift count used by the second
;   RLL is set up correctly.  The negation is independent of the NILL and L
;   tested in CHECK.
; - CHECK-SHIFT2 makes sure that %b is shifted into the high part of the word
;   before being used.  This shift is independent of the other loop prologue
;   instructions.
define i16 @f1(ptr %src, i16 %b) {
; CHECK-LABEL: f1:
; CHECK: risbg [[RISBG:%r[1-9]+]], %r2, 0, 189, 0{{$}}
; CHECK-DAG: sll %r2, 3
; CHECK-DAG: l [[OLD:%r[0-9]+]], 0([[RISBG]])
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK: rll [[ROT:%r[0-9]+]], [[OLD]], 0(%r2)
; CHECK: sr [[ROT]], %r3
; CHECK: rll [[NEW:%r[0-9]+]], [[ROT]], 0({{%r[1-9]+}})
; CHECK: cs [[OLD]], [[NEW]], 0([[RISBG]])
; CHECK: jl [[LABEL]]
; CHECK: rll %r2, [[OLD]], 16(%r2)
; CHECK: br %r14
;
; CHECK-SHIFT1-LABEL: f1:
; CHECK-SHIFT1: sll %r2, 3
; CHECK-SHIFT1: lcr [[NEGSHIFT:%r[1-9]+]], %r2
; CHECK-SHIFT1: rll
; CHECK-SHIFT1: rll {{%r[0-9]+}}, {{%r[0-9]+}}, 0([[NEGSHIFT]])
; CHECK-SHIFT1: rll
; CHECK-SHIFT1: br %r14
;
; CHECK-SHIFT2-LABEL: f1:
; CHECK-SHIFT2: sll %r3, 16
; CHECK-SHIFT2: rll
; CHECK-SHIFT2: sr {{%r[0-9]+}}, %r3
; CHECK-SHIFT2: rll
; CHECK-SHIFT2: rll
; CHECK-SHIFT2: br %r14
  %res = atomicrmw sub ptr %src, i16 %b seq_cst
  ret i16 %res
}

; Check the minimum signed value.  We add 0x80000000 to the rotated word.
define i16 @f2(ptr %src) {
; CHECK-LABEL: f2:
; CHECK: risbg [[RISBG:%r[1-9]+]], %r2, 0, 189, 0{{$}}
; CHECK-DAG: sll %r2, 3
; CHECK-DAG: l [[OLD:%r[0-9]+]], 0([[RISBG]])
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK: rll [[ROT:%r[0-9]+]], [[OLD]], 0(%r2)
; CHECK: afi [[ROT]], -2147483648
; CHECK: rll [[NEW:%r[0-9]+]], [[ROT]], 0([[NEGSHIFT:%r[1-9]+]])
; CHECK: cs [[OLD]], [[NEW]], 0([[RISBG]])
; CHECK: jl [[LABEL]]
; CHECK: rll %r2, [[OLD]], 16(%r2)
; CHECK: br %r14
;
; CHECK-SHIFT1-LABEL: f2:
; CHECK-SHIFT1: sll %r2, 3
; CHECK-SHIFT1: lcr [[NEGSHIFT:%r[1-9]+]], %r2
; CHECK-SHIFT1: rll
; CHECK-SHIFT1: rll {{%r[0-9]+}}, {{%r[0-9]+}}, 0([[NEGSHIFT]])
; CHECK-SHIFT1: rll
; CHECK-SHIFT1: br %r14
;
; CHECK-SHIFT2-LABEL: f2:
; CHECK-SHIFT2: br %r14
  %res = atomicrmw sub ptr %src, i16 -32768 seq_cst
  ret i16 %res
}

; Check subtraction of -1.  We add 0x00010000 to the rotated word.
define i16 @f3(ptr %src) {
; CHECK-LABEL: f3:
; CHECK: afi [[ROT]], 65536
; CHECK: br %r14
;
; CHECK-SHIFT1-LABEL: f3:
; CHECK-SHIFT1: br %r14
; CHECK-SHIFT2-LABEL: f3:
; CHECK-SHIFT2: br %r14
  %res = atomicrmw sub ptr %src, i16 -1 seq_cst
  ret i16 %res
}

; Check subtraction of 1.  We add 0xffff0000 to the rotated word.
define i16 @f4(ptr %src) {
; CHECK-LABEL: f4:
; CHECK: afi [[ROT]], -65536
; CHECK: br %r14
;
; CHECK-SHIFT1-LABEL: f4:
; CHECK-SHIFT1: br %r14
; CHECK-SHIFT2-LABEL: f4:
; CHECK-SHIFT2: br %r14
  %res = atomicrmw sub ptr %src, i16 1 seq_cst
  ret i16 %res
}

; Check the maximum signed value.  We add 0x80010000 to the rotated word.
define i16 @f5(ptr %src) {
; CHECK-LABEL: f5:
; CHECK: afi [[ROT]], -2147418112
; CHECK: br %r14
;
; CHECK-SHIFT1-LABEL: f5:
; CHECK-SHIFT1: br %r14
; CHECK-SHIFT2-LABEL: f5:
; CHECK-SHIFT2: br %r14
  %res = atomicrmw sub ptr %src, i16 32767 seq_cst
  ret i16 %res
}

; Check subtraction of a large unsigned value.  We add 0x00020000 to the
; rotated word.
define i16 @f6(ptr %src) {
; CHECK-LABEL: f6:
; CHECK: afi [[ROT]], 131072
; CHECK: br %r14
;
; CHECK-SHIFT1-LABEL: f6:
; CHECK-SHIFT1: br %r14
; CHECK-SHIFT2-LABEL: f6:
; CHECK-SHIFT2: br %r14
  %res = atomicrmw sub ptr %src, i16 65534 seq_cst
  ret i16 %res
}
