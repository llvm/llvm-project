; RUN: llvm-as -o - %s | llc -march=cellspu > %t1.s
; RUN: grep shufb   %t1.s | count 10
; RUN: grep {ilhu.*1799}  %t1.s | count 1
; RUN: grep {ilhu.*771}  %t1.s | count 1
; RUN: grep {ilhu.*1543}  %t1.s | count 1
; RUN: grep {ilhu.*1029}  %t1.s | count 1
; RUN: grep {ilhu.*515}  %t1.s | count 2
; RUN: grep xsbh  %t1.s | count 2
; RUN: grep sfh  %t1.s | count 1

; ModuleID = 'trunc.bc'
target datalayout = "E-p:32:32:128-i1:8:128-i8:8:128-i16:16:128-i32:32:128-i64:32:128-f32:32:128-f64:64:128-v64:64:64-v128:128:128-a0:0:128-s0:128:128"
target triple = "spu"

; codegen for i128 arguments is not implemented yet on CellSPU
; once this changes uncomment the functions below
; and update the expected results accordingly

;define i8 @trunc_i128_i8(i128 %u) nounwind readnone {
;entry:
;	%0 = trunc i128 %u to i8
;	ret i8 %0
;}
;define i16 @trunc_i128_i16(i128 %u) nounwind readnone {
;entry:
;	%0 = trunc i128 %u to i16
;	ret i16 %0
;}
;define i32 @trunc_i128_i32(i128 %u) nounwind readnone {
;entry:
;	%0 = trunc i128 %u to i32
;	ret i32 %0
;}
;define i64 @trunc_i128_i64(i128 %u) nounwind readnone {
;entry:
;	%0 = trunc i128 %u to i64
;	ret i64 %0
;}

define <16 x i8> @trunc_i64_i8(i64 %u, <16 x i8> %v) nounwind readnone {
entry:
	%0 = trunc i64 %u to i8
        %tmp1 = insertelement <16 x i8> %v, i8 %0, i32 10
	ret <16 x i8> %tmp1
}
define <8 x i16> @trunc_i64_i16(i64 %u, <8 x i16> %v) nounwind readnone {
entry:
	%0 = trunc i64 %u to i16
        %tmp1 = insertelement <8 x i16> %v, i16 %0, i32 6
	ret <8 x i16> %tmp1
}
define i32 @trunc_i64_i32(i64 %u, i32 %v) nounwind readnone {
entry:
	%0 = trunc i64 %u to i32
	ret i32 %0
}

define i8 @trunc_i32_i8(i32 %u, i8 %v) nounwind readnone {
entry:
	%0 = trunc i32 %u to i8
	%1 = sub i8 %0, %v
	ret i8 %1
}
define <8 x i16> @trunc_i32_i16(i32 %u, <8 x i16> %v) nounwind readnone {
entry:
	%0 = trunc i32 %u to i16
        %tmp1 = insertelement <8 x i16> %v, i16 %0, i32 3
	ret <8 x i16> %tmp1
}

define <16 x i8> @trunc_i16_i8(i16 %u, <16 x i8> %v) nounwind readnone {
entry:
	%0 = trunc i16 %u to i8
        %tmp1 = insertelement <16 x i8> %v, i8 %0, i32 5
	ret <16 x i8> %tmp1
}
