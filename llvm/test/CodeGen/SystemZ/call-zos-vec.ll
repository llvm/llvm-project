; RUN: llc < %s -mtriple=s390x-ibm-zos -mcpu=z13 | FileCheck %s

; CHECK-LABEL: sum_vecs0
; CHECK: vag 24,24,25
define <2 x i64> @sum_vecs0(<2 x i64> %v1, <2 x i64> %v2) {
entry:
  %add0 = add <2 x i64> %v1, %v2
  ret <2 x i64> %add0
}

; CHECK-LABEL: sum_vecs1
; CHECK: vaf 1,24,25
; CHECK: vaf 1,1,26
; CHECK: vaf 1,1,27
; CHECK: vaf 1,1,28
; CHECK: vaf 1,1,29
; CHECK: vl  0,2304(4),4
; CHECK: vaf 1,1,30
; CHECK: vaf 1,1,31
; CHECK: vaf 24,1,0
define <4 x i32> @sum_vecs1(<4 x i32> %v1, <4 x i32> %v2, <4 x i32> %v3, <4 x i32> %v4, <4 x i32> %v5, <4 x i32> %v6, <4 x i32> %v7, <4 x i32> %v8, <4 x i32> %v9) {
entry:
  %add0 = add <4 x i32> %v1, %v2
  %add1 = add <4 x i32> %add0, %v3
  %add2 = add <4 x i32> %add1, %v4
  %add3 = add <4 x i32> %add2, %v5
  %add4 = add <4 x i32> %add3, %v6
  %add5 = add <4 x i32> %add4, %v7
  %add6 = add <4 x i32> %add5, %v8
  %add7 = add <4 x i32> %add6, %v9
  ret <4 x i32> %add7
}

; Verify that 3 is used for passing integral types if
; only 24 is used.
; CHECK-LABEL: call_vecs0
; CHECK: lgr 3,1
define i64 @call_vecs0(i64 %n, <2 x i64> %v1) {
entry:
  %ret = call i64 (<2 x i64>, i64) @pass_vecs0(<2 x i64> %v1, i64 %n)
  ret i64 %ret
}

; Verify that 3 is not allocated for passing integral types
; if 24 and %f0 are used.
; CHECK-LABEL: call_vecs1
; CHECK: vlr 24,25
; CHECK: stg 1,2200(4)
define i64 @call_vecs1(i64 %n, <2 x i64> %v1, double %x, <2 x i64> %v2) {
entry:
  %ret = call i64 (<2 x i64>, double, i64) @pass_vecs1(<2 x i64> %v2, double %x, i64 %n)
  ret i64 %ret
}

; Verify that 3 is not allocated for passing integral types
; if 24 and 25 are used.
; CHECK-LABEL: call_vecs2
; CHECK: mvghi 2208(4),55
define i64 @call_vecs2(<2 x i64> %v1, <2 x i64> %v2) {
  %ret = call i64 (<2 x i64>, <2 x i64>, i64) @pass_vecs2(<2 x i64> %v1, <2 x i64> %v2, i64 55)
  ret i64 %ret
}

declare i64 @pass_vecs0(<2 x i64> %v1, i64 %n)
declare i64 @pass_vecs1(<2 x i64> %v1, double %x, i64 %n)
declare i64 @pass_vecs2(<2 x i64> %v1, <2 x i64> %v2, i64 %n)
