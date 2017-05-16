; RUN: opt %loadPolly -analyze -polly-detect-fortran-arrays \
; RUN: -polly-scops -polly-allow-nonaffine -polly-invariant-load-hoisting < %s | FileCheck %s

; This testcase is the corresponding LLVM for testfunc:
; PROGRAM main
;     INTEGER, DIMENSION(1) :: xs
;
;     CALL testfunc(xs, 10)
; CONTAINS
;     SUBROUTINE func(xs, n)
;         IMPLICIT NONE
;         INTEGER, DIMENSION(:), INTENT(INOUT) :: xs
;         INTEGER, INTENT(IN) :: n
;         INTEGER :: i

;         DO i = 1, n
;             xs(i) = 1
;         END DO
;
;     END SUBROUTINE func
; END PROGRAM

target datalayout = "e-p:64:64:64-S128-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

module asm "\09.ident\09\22GCC: (GNU) 4.6.4 LLVM: 3.3.1\22"

%"struct.array1_integer(kind=4)" = type { i8*, i64, i64, [1 x %struct.descriptor_dimension] }
%struct.descriptor_dimension = type { i64, i64, i64 }

define internal void @testfunc(%"struct.array1_integer(kind=4)"* noalias %xs, i32* noalias %n) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  %tmp = getelementptr inbounds %"struct.array1_integer(kind=4)", %"struct.array1_integer(kind=4)"* %xs, i64 0, i32 3, i64 0, i32 0
  %tmp1 = load i64, i64* %tmp, align 8
  %tmp2 = icmp eq i64 %tmp1, 0
  %tmp3 = select i1 %tmp2, i64 1, i64 %tmp1
  %tmp4 = bitcast %"struct.array1_integer(kind=4)"* %xs to i32**
  %tmp5 = load i32*, i32** %tmp4, align 8
  %tmp6 = load i32, i32* %n, align 4
  %tmp7 = icmp sgt i32 %tmp6, 0
  br i1 %tmp7, label %"6.preheader", label %return

"6.preheader":                                    ; preds = %entry.split
  br label %"6"

"6":                                              ; preds = %"6", %"6.preheader"
  %tmp8 = phi i32 [ %tmp14, %"6" ], [ 1, %"6.preheader" ]
  %tmp9 = sext i32 %tmp8 to i64
  %tmp10 = mul i64 %tmp3, %tmp9
  %tmp11 = sub i64 %tmp10, %tmp3
  %tmp12 = getelementptr i32, i32* %tmp5, i64 %tmp11
  store i32 1, i32* %tmp12, align 4
  %tmp13 = icmp eq i32 %tmp8, %tmp6
  %tmp14 = add i32 %tmp8, 1
  br i1 %tmp13, label %return.loopexit, label %"6"

return.loopexit:                                  ; preds = %"6"
  br label %return

return:                                           ; preds = %return.loopexit, %entry.split
  ret void
}

; CHECK: ReadAccess :=  [Reduction Type: NONE] [Fortran array descriptor: xs] [Scalar: 0]
