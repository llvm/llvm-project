; Test the use of TM and TMY.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

@g = global i32 0

; Check a simple branching use of TM.
define void @f1(i8 *%src) {
; CHECK-LABEL: f1:
; CHECK: tm 0(%r2), 1
; CHECK: ber %r14
; CHECK: br %r14
entry:
  %byte = load i8 , i8 *%src
  %and = and i8 %byte, 1
  %cmp = icmp eq i8 %and, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}


; Check that we do not fold across an aliasing store.
define void @f2(i8 *%src) {
; CHECK-LABEL: f2:
; CHECK: llc [[REG:%r[0-5]]], 0(%r2)
; CHECK: mvi 0(%r2), 0
; CHECK: tmll [[REG]], 1
; CHECK: ber %r14
; CHECK: br %r14
entry:
  %byte = load i8 , i8 *%src
  store i8 0, i8 *%src
  %and = and i8 %byte, 1
  %cmp = icmp eq i8 %and, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; Check a simple select-based use of TM.
define double @f3(i8 *%src, double %a, double %b) {
; CHECK-LABEL: f3:
; CHECK: tm 0(%r2), 1
; CHECK: je {{\.L.*}}
; CHECK: br %r14
  %byte = load i8 , i8 *%src
  %and = and i8 %byte, 1
  %cmp = icmp eq i8 %and, 0
  %res = select i1 %cmp, double %b, double %a
  ret double %res
}

; Check that we do not fold across an aliasing store.
define double @f4(i8 *%src, double %a, double %b) {
; CHECK-LABEL: f4:
; CHECK: tm 0(%r2), 1
; CHECK: je {{\.L.*}}
; CHECK: mvi 0(%r2), 0
; CHECK: br %r14
  %byte = load i8 , i8 *%src
  %and = and i8 %byte, 1
  %cmp = icmp eq i8 %and, 0
  %res = select i1 %cmp, double %b, double %a
  store i8 0, i8 *%src
  ret double %res
}

; Check an inequality check.
define double @f5(i8 *%src, double %a, double %b) {
; CHECK-LABEL: f5:
; CHECK: tm 0(%r2), 1
; CHECK: jne {{\.L.*}}
; CHECK: br %r14
  %byte = load i8 , i8 *%src
  %and = and i8 %byte, 1
  %cmp = icmp ne i8 %and, 0
  %res = select i1 %cmp, double %b, double %a
  ret double %res
}

; Check that we can also use TM for equality comparisons with the mask.
define double @f6(i8 *%src, double %a, double %b) {
; CHECK-LABEL: f6:
; CHECK: tm 0(%r2), 254
; CHECK: jo {{\.L.*}}
; CHECK: br %r14
  %byte = load i8 , i8 *%src
  %and = and i8 %byte, 254
  %cmp = icmp eq i8 %and, 254
  %res = select i1 %cmp, double %b, double %a
  ret double %res
}

; Check inequality comparisons with the mask.
define double @f7(i8 *%src, double %a, double %b) {
; CHECK-LABEL: f7:
; CHECK: tm 0(%r2), 254
; CHECK: jno {{\.L.*}}
; CHECK: br %r14
  %byte = load i8 , i8 *%src
  %and = and i8 %byte, 254
  %cmp = icmp ne i8 %and, 254
  %res = select i1 %cmp, double %b, double %a
  ret double %res
}

; Check that we do not use the memory TM instruction when CC is being tested
; for 2.
define double @f8(i8 *%src, double %a, double %b) {
; CHECK-LABEL: f8:
; CHECK: llc [[REG:%r[0-5]]], 0(%r2)
; CHECK: tmll [[REG]], 3
; CHECK: jh {{\.L.*}}
; CHECK: br %r14
  %byte = load i8 , i8 *%src
  %and = and i8 %byte, 3
  %cmp = icmp eq i8 %and, 2
  %res = select i1 %cmp, double %b, double %a
  ret double %res
}

; ...likewise 1.
define double @f9(i8 *%src, double %a, double %b) {
; CHECK-LABEL: f9:
; CHECK: llc [[REG:%r[0-5]]], 0(%r2)
; CHECK: tmll [[REG]], 3
; CHECK: jl {{\.L.*}}
; CHECK: br %r14
  %byte = load i8 , i8 *%src
  %and = and i8 %byte, 3
  %cmp = icmp eq i8 %and, 1
  %res = select i1 %cmp, double %b, double %a
  ret double %res
}

; Check the high end of the TM range.
define double @f10(i8 *%src, double %a, double %b) {
; CHECK-LABEL: f10:
; CHECK: tm 4095(%r2), 1
; CHECK: je {{\.L.*}}
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%src, i64 4095
  %byte = load i8 , i8 *%ptr
  %and = and i8 %byte, 1
  %cmp = icmp eq i8 %and, 0
  %res = select i1 %cmp, double %b, double %a
  ret double %res
}

; Check the low end of the positive TMY range.
define double @f11(i8 *%src, double %a, double %b) {
; CHECK-LABEL: f11:
; CHECK: tmy 4096(%r2), 1
; CHECK: je {{\.L.*}}
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%src, i64 4096
  %byte = load i8 , i8 *%ptr
  %and = and i8 %byte, 1
  %cmp = icmp eq i8 %and, 0
  %res = select i1 %cmp, double %b, double %a
  ret double %res
}

; Check the high end of the TMY range.
define double @f12(i8 *%src, double %a, double %b) {
; CHECK-LABEL: f12:
; CHECK: tmy 524287(%r2), 1
; CHECK: je {{\.L.*}}
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%src, i64 524287
  %byte = load i8 , i8 *%ptr
  %and = and i8 %byte, 1
  %cmp = icmp eq i8 %and, 0
  %res = select i1 %cmp, double %b, double %a
  ret double %res
}

; Check the next byte up, which needs separate address logic.
define double @f13(i8 *%src, double %a, double %b) {
; CHECK-LABEL: f13:
; CHECK: agfi %r2, 524288
; CHECK: tm 0(%r2), 1
; CHECK: je {{\.L.*}}
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%src, i64 524288
  %byte = load i8 , i8 *%ptr
  %and = and i8 %byte, 1
  %cmp = icmp eq i8 %and, 0
  %res = select i1 %cmp, double %b, double %a
  ret double %res
}

; Check the low end of the TMY range.
define double @f14(i8 *%src, double %a, double %b) {
; CHECK-LABEL: f14:
; CHECK: tmy -524288(%r2), 1
; CHECK: je {{\.L.*}}
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%src, i64 -524288
  %byte = load i8 , i8 *%ptr
  %and = and i8 %byte, 1
  %cmp = icmp eq i8 %and, 0
  %res = select i1 %cmp, double %b, double %a
  ret double %res
}

; Check the next byte down, which needs separate address logic.
define double @f15(i8 *%src, double %a, double %b) {
; CHECK-LABEL: f15:
; CHECK: agfi %r2, -524289
; CHECK: tm 0(%r2), 1
; CHECK: je {{\.L.*}}
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%src, i64 -524289
  %byte = load i8 , i8 *%ptr
  %and = and i8 %byte, 1
  %cmp = icmp eq i8 %and, 0
  %res = select i1 %cmp, double %b, double %a
  ret double %res
}

; Check that TM(Y) does not allow an index
define double @f16(i8 *%src, i64 %index, double %a, double %b) {
; CHECK-LABEL: f16:
; CHECK: tm 0({{%r[1-5]}}), 1
; CHECK: je {{\.L.*}}
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%src, i64 %index
  %byte = load i8 , i8 *%ptr
  %and = and i8 %byte, 1
  %cmp = icmp eq i8 %and, 0
  %res = select i1 %cmp, double %b, double %a
  ret double %res
}
