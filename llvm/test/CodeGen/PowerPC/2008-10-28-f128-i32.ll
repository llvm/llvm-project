; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- -o - | not grep fixunstfsi

define i64 @__fixunstfdi(ppc_fp128 %a) nounwind readnone {
entry:
	%0 = fcmp olt ppc_fp128 %a, 0xM00000000000000000000000000000000		; <i1> [#uses=1]
	br i1 %0, label %bb5, label %bb1

bb1:		; preds = %entry
	%1 = fmul ppc_fp128 %a, 0xM3DF00000000000000000000000000000		; <ppc_fp128> [#uses=1]
	%2 = fptoui ppc_fp128 %1 to i32		; <i32> [#uses=1]
	%3 = zext i32 %2 to i64		; <i64> [#uses=1]
	%4 = shl i64 %3, 32		; <i64> [#uses=3]
	%5 = uitofp i64 %4 to ppc_fp128		; <ppc_fp128> [#uses=1]
	%6 = fsub ppc_fp128 %a, %5		; <ppc_fp128> [#uses=3]
	%7 = fcmp olt ppc_fp128 %6, 0xM00000000000000000000000000000000		; <i1> [#uses=1]
	br i1 %7, label %bb2, label %bb3

bb2:		; preds = %bb1
	%8 = fsub ppc_fp128 0xM80000000000000000000000000000000, %6		; <ppc_fp128> [#uses=1]
	%9 = fptoui ppc_fp128 %8 to i32		; <i32> [#uses=1]
	%10 = zext i32 %9 to i64		; <i64> [#uses=1]
	%11 = sub i64 %4, %10		; <i64> [#uses=1]
	ret i64 %11

bb3:		; preds = %bb1
	%12 = fptoui ppc_fp128 %6 to i32		; <i32> [#uses=1]
	%13 = zext i32 %12 to i64		; <i64> [#uses=1]
	%14 = or i64 %13, %4		; <i64> [#uses=1]
	ret i64 %14

bb5:		; preds = %entry
	ret i64 0
}
