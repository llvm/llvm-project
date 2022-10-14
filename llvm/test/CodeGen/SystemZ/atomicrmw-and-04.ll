; Test 64-bit atomic ANDs.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s

; Check ANDs of a variable.
define i64 @f1(i64 %dummy, ptr %src, i64 %b) {
; CHECK-LABEL: f1:
; CHECK: lg %r2, 0(%r3)
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK: lgr %r0, %r2
; CHECK: ngr %r0, %r4
; CHECK: csg %r2, %r0, 0(%r3)
; CHECK: jl [[LABEL]]
; CHECK: br %r14
  %res = atomicrmw and ptr %src, i64 %b seq_cst
  ret i64 %res
}

; Check ANDs of 1, which are done using a register.  (We could use RISBG
; instead, but that isn't implemented yet.)
define i64 @f2(i64 %dummy, ptr %src) {
; CHECK-LABEL: f2:
; CHECK: ngr
; CHECK: br %r14
  %res = atomicrmw and ptr %src, i64 1 seq_cst
  ret i64 %res
}

; Check the equivalent of NIHF with 1, which can use RISBG instead.
define i64 @f3(i64 %dummy, ptr %src) {
; CHECK-LABEL: f3:
; CHECK: lg %r2, 0(%r3)
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK: risbg %r0, %r2, 31, 191, 0
; CHECK: csg %r2, %r0, 0(%r3)
; CHECK: jl [[LABEL]]
; CHECK: br %r14
  %res = atomicrmw and ptr %src, i64 8589934591 seq_cst
  ret i64 %res
}

; Check the lowest NIHF value outside the range of RISBG.
define i64 @f4(i64 %dummy, ptr %src) {
; CHECK-LABEL: f4:
; CHECK: lg %r2, 0(%r3)
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK: lgr %r0, %r2
; CHECK: nihf %r0, 2
; CHECK: csg %r2, %r0, 0(%r3)
; CHECK: jl [[LABEL]]
; CHECK: br %r14
  %res = atomicrmw and ptr %src, i64 12884901887 seq_cst
  ret i64 %res
}

; Check the next value up, which must use a register.
define i64 @f5(i64 %dummy, ptr %src) {
; CHECK-LABEL: f5:
; CHECK: ngr
; CHECK: br %r14
  %res = atomicrmw and ptr %src, i64 12884901888 seq_cst
  ret i64 %res
}

; Check the lowest NIHH value outside the range of RISBG.
define i64 @f6(i64 %dummy, ptr %src) {
; CHECK-LABEL: f6:
; CHECK: nihh {{%r[0-5]}}, 2
; CHECK: br %r14
  %res = atomicrmw and ptr %src, i64 844424930131967 seq_cst
  ret i64 %res
}

; Check the next value up, which must use a register.
define i64 @f7(i64 %dummy, ptr %src) {
; CHECK-LABEL: f7:
; CHECK: ngr
; CHECK: br %r14
  %res = atomicrmw and ptr %src, i64 281474976710656 seq_cst
  ret i64 %res
}

; Check the highest NILL value outside the range of RISBG.
define i64 @f8(i64 %dummy, ptr %src) {
; CHECK-LABEL: f8:
; CHECK: nill {{%r[0-5]}}, 65530
; CHECK: br %r14
  %res = atomicrmw and ptr %src, i64 -6 seq_cst
  ret i64 %res
}

; Check the lowest NILL value outside the range of RISBG.
define i64 @f9(i64 %dummy, ptr %src) {
; CHECK-LABEL: f9:
; CHECK: nill {{%r[0-5]}}, 2
; CHECK: br %r14
  %res = atomicrmw and ptr %src, i64 -65534 seq_cst
  ret i64 %res
}

; Check the highest useful NILF value.
define i64 @f10(i64 %dummy, ptr %src) {
; CHECK-LABEL: f10:
; CHECK: nilf {{%r[0-5]}}, 4294901758
; CHECK: br %r14
  %res = atomicrmw and ptr %src, i64 -65538 seq_cst
  ret i64 %res
}

; Check the highest NILH value outside the range of RISBG.
define i64 @f11(i64 %dummy, ptr %src) {
; CHECK-LABEL: f11:
; CHECK: nilh {{%r[0-5]}}, 65530
; CHECK: br %r14
  %res = atomicrmw and ptr %src, i64 -327681 seq_cst
  ret i64 %res
}

; Check the lowest NILH value outside the range of RISBG.
define i64 @f12(i64 %dummy, ptr %src) {
; CHECK-LABEL: f12:
; CHECK: nilh {{%r[0-5]}}, 2
; CHECK: br %r14
  %res = atomicrmw and ptr %src, i64 -4294770689 seq_cst
  ret i64 %res
}

; Check the lowest NILF value outside the range of RISBG.
define i64 @f13(i64 %dummy, ptr %src) {
; CHECK-LABEL: f13:
; CHECK: nilf {{%r[0-5]}}, 2
; CHECK: br %r14
  %res = atomicrmw and ptr %src, i64 -4294967294 seq_cst
  ret i64 %res
}

; Check the highest NIHL value outside the range of RISBG.
define i64 @f14(i64 %dummy, ptr %src) {
; CHECK-LABEL: f14:
; CHECK: nihl {{%r[0-5]}}, 65530
; CHECK: br %r14
  %res = atomicrmw and ptr %src, i64 -21474836481 seq_cst
  ret i64 %res
}

; Check the lowest NIHL value outside the range of RISBG.
define i64 @f15(i64 %dummy, ptr %src) {
; CHECK-LABEL: f15:
; CHECK: nihl {{%r[0-5]}}, 2
; CHECK: br %r14
  %res = atomicrmw and ptr %src, i64 -281462091808769 seq_cst
  ret i64 %res
}

; Check the highest NIHH value outside the range of RISBG.
define i64 @f16(i64 %dummy, ptr %src) {
; CHECK-LABEL: f16:
; CHECK: nihh {{%r[0-5]}}, 65530
; CHECK: br %r14
  %res = atomicrmw and ptr %src, i64 -1407374883553281 seq_cst
  ret i64 %res
}

; Check the highest useful NIHF value.
define i64 @f17(i64 %dummy, ptr %src) {
; CHECK-LABEL: f17:
; CHECK: nihf {{%r[0-5]}}, 4294901758
; CHECK: br %r14
  %res = atomicrmw and ptr %src, i64 -281479271677953 seq_cst
  ret i64 %res
}
