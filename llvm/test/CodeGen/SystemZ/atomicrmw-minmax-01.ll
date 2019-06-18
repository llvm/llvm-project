; Test 8-bit atomic min/max operations.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -disable-block-placement | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -disable-block-placement | FileCheck %s -check-prefix=CHECK-SHIFT1
; RUN: llc < %s -mtriple=s390x-linux-gnu -disable-block-placement | FileCheck %s -check-prefix=CHECK-SHIFT2

; Check signed minimum.
; - CHECK is for the main loop.
; - CHECK-SHIFT1 makes sure that the negated shift count used by the second
;   RLL is set up correctly.  The negation is independent of the NILL and L
;   tested in CHECK.
; - CHECK-SHIFT2 makes sure that %b is shifted into the high part of the word
;   before being used, and that the low bits are set to 1.  This sequence is
;   independent of the other loop prologue instructions.
define i8 @f1(i8 *%src, i8 %b) {
; CHECK-LABEL: f1:
; CHECK: risbg [[RISBG:%r[1-9]+]], %r2, 0, 189, 0{{$}}
; CHECK-DAG: sll %r2, 3
; CHECK-DAG: l [[OLD:%r[0-9]+]], 0([[RISBG]])
; CHECK: [[LOOP:\.[^:]*]]:
; CHECK: rll [[ROT:%r[0-9]+]], [[OLD]], 0(%r2)
; CHECK: crjle [[ROT]], %r3, [[KEEP:\..*]]
; CHECK: risbg [[ROT]], %r3, 32, 39, 0
; CHECK: [[KEEP]]:
; CHECK: rll [[NEW:%r[0-9]+]], [[ROT]], 0({{%r[1-9]+}})
; CHECK: cs [[OLD]], [[NEW]], 0([[RISBG]])
; CHECK: jl [[LOOP]]
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
; CHECK-SHIFT2: rll
; CHECK-SHIFT2: crjle {{%r[0-9]+}}, %r3
; CHECK-SHIFT2: rll
; CHECK-SHIFT2: rll
; CHECK-SHIFT2: br %r14
  %res = atomicrmw min i8 *%src, i8 %b seq_cst
  ret i8 %res
}

; Check signed maximum.
define i8 @f2(i8 *%src, i8 %b) {
; CHECK-LABEL: f2:
; CHECK: risbg [[RISBG:%r[1-9]+]], %r2, 0, 189, 0{{$}}
; CHECK-DAG: sll %r2, 3
; CHECK-DAG: l [[OLD:%r[0-9]+]], 0([[RISBG]])
; CHECK: [[LOOP:\.[^:]*]]:
; CHECK: rll [[ROT:%r[0-9]+]], [[OLD]], 0(%r2)
; CHECK: crjhe [[ROT]], %r3, [[KEEP:\..*]]
; CHECK: risbg [[ROT]], %r3, 32, 39, 0
; CHECK: [[KEEP]]:
; CHECK: rll [[NEW:%r[0-9]+]], [[ROT]], 0({{%r[1-9]+}})
; CHECK: cs [[OLD]], [[NEW]], 0([[RISBG]])
; CHECK: jl [[LOOP]]
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
; CHECK-SHIFT2: sll %r3, 24
; CHECK-SHIFT2: rll
; CHECK-SHIFT2: crjhe {{%r[0-9]+}}, %r3
; CHECK-SHIFT2: rll
; CHECK-SHIFT2: rll
; CHECK-SHIFT2: br %r14
  %res = atomicrmw max i8 *%src, i8 %b seq_cst
  ret i8 %res
}

; Check unsigned minimum.
define i8 @f3(i8 *%src, i8 %b) {
; CHECK-LABEL: f3:
; CHECK: risbg [[RISBG:%r[1-9]+]], %r2, 0, 189, 0{{$}}
; CHECK-DAG: sll %r2, 3
; CHECK-DAG: l [[OLD:%r[0-9]+]], 0([[RISBG]])
; CHECK: [[LOOP:\.[^:]*]]:
; CHECK: rll [[ROT:%r[0-9]+]], [[OLD]], 0(%r2)
; CHECK: clrjle [[ROT]], %r3, [[KEEP:\..*]]
; CHECK: risbg [[ROT]], %r3, 32, 39, 0
; CHECK: [[KEEP]]:
; CHECK: rll [[NEW:%r[0-9]+]], [[ROT]], 0({{%r[1-9]+}})
; CHECK: cs [[OLD]], [[NEW]], 0([[RISBG]])
; CHECK: jl [[LOOP]]
; CHECK: rll %r2, [[OLD]], 8(%r2)
; CHECK: br %r14
;
; CHECK-SHIFT1-LABEL: f3:
; CHECK-SHIFT1: sll %r2, 3
; CHECK-SHIFT1: lcr [[NEGSHIFT:%r[1-9]+]], %r2
; CHECK-SHIFT1: rll
; CHECK-SHIFT1: rll {{%r[0-9]+}}, {{%r[0-9]+}}, 0([[NEGSHIFT]])
; CHECK-SHIFT1: rll
; CHECK-SHIFT1: br %r14
;
; CHECK-SHIFT2-LABEL: f3:
; CHECK-SHIFT2: sll %r3, 24
; CHECK-SHIFT2: rll
; CHECK-SHIFT2: clrjle {{%r[0-9]+}}, %r3,
; CHECK-SHIFT2: rll
; CHECK-SHIFT2: rll
; CHECK-SHIFT2: br %r14
  %res = atomicrmw umin i8 *%src, i8 %b seq_cst
  ret i8 %res
}

; Check unsigned maximum.
define i8 @f4(i8 *%src, i8 %b) {
; CHECK-LABEL: f4:
; CHECK: risbg [[RISBG:%r[1-9]+]], %r2, 0, 189, 0{{$}}
; CHECK-DAG: sll %r2, 3
; CHECK-DAG: l [[OLD:%r[0-9]+]], 0([[RISBG]])
; CHECK: [[LOOP:\.[^:]*]]:
; CHECK: rll [[ROT:%r[0-9]+]], [[OLD]], 0(%r2)
; CHECK: clrjhe [[ROT]], %r3, [[KEEP:\..*]]
; CHECK: risbg [[ROT]], %r3, 32, 39, 0
; CHECK: [[KEEP]]:
; CHECK: rll [[NEW:%r[0-9]+]], [[ROT]], 0({{%r[1-9]+}})
; CHECK: cs [[OLD]], [[NEW]], 0([[RISBG]])
; CHECK: jl [[LOOP]]
; CHECK: rll %r2, [[OLD]], 8(%r2)
; CHECK: br %r14
;
; CHECK-SHIFT1-LABEL: f4:
; CHECK-SHIFT1: sll %r2, 3
; CHECK-SHIFT1: lcr [[NEGSHIFT:%r[1-9]+]], %r2
; CHECK-SHIFT1: rll
; CHECK-SHIFT1: rll {{%r[0-9]+}}, {{%r[0-9]+}}, 0([[NEGSHIFT]])
; CHECK-SHIFT1: rll
; CHECK-SHIFT1: br %r14
;
; CHECK-SHIFT2-LABEL: f4:
; CHECK-SHIFT2: sll %r3, 24
; CHECK-SHIFT2: rll
; CHECK-SHIFT2: clrjhe {{%r[0-9]+}}, %r3,
; CHECK-SHIFT2: rll
; CHECK-SHIFT2: rll
; CHECK-SHIFT2: br %r14
  %res = atomicrmw umax i8 *%src, i8 %b seq_cst
  ret i8 %res
}

; Check the lowest useful signed minimum value.  We need to load 0x81000000
; into the source register.
define i8 @f5(i8 *%src) {
; CHECK-LABEL: f5:
; CHECK: llilh [[SRC2:%r[0-9]+]], 33024
; CHECK: crjle [[ROT:%r[0-9]+]], [[SRC2]]
; CHECK: risbg [[ROT]], [[SRC2]], 32, 39, 0
; CHECK: br %r14
;
; CHECK-SHIFT1-LABEL: f5:
; CHECK-SHIFT1: br %r14
; CHECK-SHIFT2-LABEL: f5:
; CHECK-SHIFT2: br %r14
  %res = atomicrmw min i8 *%src, i8 -127 seq_cst
  ret i8 %res
}

; Check the highest useful signed maximum value.  We need to load 0x7e000000
; into the source register.
define i8 @f6(i8 *%src) {
; CHECK-LABEL: f6:
; CHECK: llilh [[SRC2:%r[0-9]+]], 32256
; CHECK: crjhe [[ROT:%r[0-9]+]], [[SRC2]]
; CHECK: risbg [[ROT]], [[SRC2]], 32, 39, 0
; CHECK: br %r14
;
; CHECK-SHIFT1-LABEL: f6:
; CHECK-SHIFT1: br %r14
; CHECK-SHIFT2-LABEL: f6:
; CHECK-SHIFT2: br %r14
  %res = atomicrmw max i8 *%src, i8 126 seq_cst
  ret i8 %res
}

; Check the lowest useful unsigned minimum value.  We need to load 0x01000000
; into the source register.
define i8 @f7(i8 *%src) {
; CHECK-LABEL: f7:
; CHECK: llilh [[SRC2:%r[0-9]+]], 256
; CHECK: clrjle [[ROT:%r[0-9]+]], [[SRC2]],
; CHECK: risbg [[ROT]], [[SRC2]], 32, 39, 0
; CHECK: br %r14
;
; CHECK-SHIFT1-LABEL: f7:
; CHECK-SHIFT1: br %r14
; CHECK-SHIFT2-LABEL: f7:
; CHECK-SHIFT2: br %r14
  %res = atomicrmw umin i8 *%src, i8 1 seq_cst
  ret i8 %res
}

; Check the highest useful unsigned maximum value.  We need to load 0xfe000000
; into the source register.
define i8 @f8(i8 *%src) {
; CHECK-LABEL: f8:
; CHECK: llilh [[SRC2:%r[0-9]+]], 65024
; CHECK: clrjhe [[ROT:%r[0-9]+]], [[SRC2]],
; CHECK: risbg [[ROT]], [[SRC2]], 32, 39, 0
; CHECK: br %r14
;
; CHECK-SHIFT1-LABEL: f8:
; CHECK-SHIFT1: br %r14
; CHECK-SHIFT2-LABEL: f8:
; CHECK-SHIFT2: br %r14
  %res = atomicrmw umax i8 *%src, i8 254 seq_cst
  ret i8 %res
}
