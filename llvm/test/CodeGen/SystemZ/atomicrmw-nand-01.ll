; Test 8-bit atomic NANDs.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CHECK-SHIFT1
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CHECK-SHIFT2

; Check NAND of a variable.
; - CHECK is for the main loop.
; - CHECK-SHIFT1 makes sure that the negated shift count used by the second
;   RLL is set up correctly.  The negation is independent of the NILL and L
;   tested in CHECK.
; - CHECK-SHIFT2 makes sure that %b is shifted into the high part of the word
;   before being used, and that the low bits are set to 1.  This sequence is
;   independent of the other loop prologue instructions.
define i8 @f1(ptr %src, i8 %b) {
; CHECK-LABEL: f1:
; CHECK: risbg [[RISBG:%r[1-9]+]], %r2, 0, 189, 0{{$}}
; CHECK-DAG: sll %r2, 3
; CHECK-DAG: l [[OLD:%r[0-9]+]], 0([[RISBG]])
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK: rll [[ROT:%r[0-9]+]], [[OLD]], 0(%r2)
; CHECK: nr [[ROT]], %r3
; CHECK: xilf [[ROT]], 4278190080
; CHECK: rll [[NEW:%r[0-9]+]], [[ROT]], 0({{%r[1-9]+}})
; CHECK: cs [[OLD]], [[NEW]], 0([[RISBG]])
; CHECK: jl [[LABEL]]
; CHECK: rll %r2, [[OLD]], 8(%r2)
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
; CHECK-SHIFT2: sll %r3, 24
; CHECK-SHIFT2: oilf %r3, 16777215
; CHECK-SHIFT2: rll
; CHECK-SHIFT2: nr {{%r[0-9]+}}, %r3
; CHECK-SHIFT2: rll
; CHECK-SHIFT2: rll
; CHECK-SHIFT2: br %r14
  %res = atomicrmw nand ptr %src, i8 %b seq_cst
  ret i8 %res
}

; Check the minimum signed value.  We AND the rotated word with 0x80ffffff.
define i8 @f2(ptr %src) {
; CHECK-LABEL: f2:
; CHECK: risbg [[RISBG:%r[1-9]+]], %r2, 0, 189, 0{{$}}
; CHECK-DAG: sll %r2, 3
; CHECK-DAG: l [[OLD:%r[0-9]+]], 0([[RISBG]])
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK: rll [[ROT:%r[0-9]+]], [[OLD]], 0(%r2)
; CHECK: nilh [[ROT]], 33023
; CHECK: xilf [[ROT]], 4278190080
; CHECK: rll [[NEW:%r[0-9]+]], [[ROT]], 0([[NEGSHIFT:%r[1-9]+]])
; CHECK: cs [[OLD]], [[NEW]], 0([[RISBG]])
; CHECK: jl [[LABEL]]
; CHECK: rll %r2, [[OLD]], 8(%r2)
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
  %res = atomicrmw nand ptr %src, i8 -128 seq_cst
  ret i8 %res
}

; Check NANDs of -2 (-1 isn't useful).  We AND the rotated word with 0xfeffffff.
define i8 @f3(ptr %src) {
; CHECK-LABEL: f3:
; CHECK: nilh [[ROT]], 65279
; CHECK: xilf [[ROT]], 4278190080
; CHECK: br %r14
;
; CHECK-SHIFT1-LABEL: f3:
; CHECK-SHIFT1: br %r14
; CHECK-SHIFT2-LABEL: f3:
; CHECK-SHIFT2: br %r14
  %res = atomicrmw nand ptr %src, i8 -2 seq_cst
  ret i8 %res
}

; Check NANDs of 1.  We AND the rotated word with 0x01ffffff.
define i8 @f4(ptr %src) {
; CHECK-LABEL: f4:
; CHECK: nilh [[ROT]], 511
; CHECK: xilf [[ROT]], 4278190080
; CHECK: br %r14
;
; CHECK-SHIFT1-LABEL: f4:
; CHECK-SHIFT1: br %r14
; CHECK-SHIFT2-LABEL: f4:
; CHECK-SHIFT2: br %r14
  %res = atomicrmw nand ptr %src, i8 1 seq_cst
  ret i8 %res
}

; Check the maximum signed value.  We AND the rotated word with 0x7fffffff.
define i8 @f5(ptr %src) {
; CHECK-LABEL: f5:
; CHECK: nilh [[ROT]], 32767
; CHECK: xilf [[ROT]], 4278190080
; CHECK: br %r14
;
; CHECK-SHIFT1-LABEL: f5:
; CHECK-SHIFT1: br %r14
; CHECK-SHIFT2-LABEL: f5:
; CHECK-SHIFT2: br %r14
  %res = atomicrmw nand ptr %src, i8 127 seq_cst
  ret i8 %res
}

; Check NANDs of a large unsigned value.  We AND the rotated word with
; 0xfdffffff.
define i8 @f6(ptr %src) {
; CHECK-LABEL: f6:
; CHECK: nilh [[ROT]], 65023
; CHECK: xilf [[ROT]], 4278190080
; CHECK: br %r14
;
; CHECK-SHIFT1-LABEL: f6:
; CHECK-SHIFT1: br %r14
; CHECK-SHIFT2-LABEL: f6:
; CHECK-SHIFT2: br %r14
  %res = atomicrmw nand ptr %src, i8 253 seq_cst
  ret i8 %res
}
