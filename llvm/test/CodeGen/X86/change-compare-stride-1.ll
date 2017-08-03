; RUN: llc < %s -march=x86-64 -enable-lsr-nested | FileCheck %s
;
; Nested LSR is required to optimize this case.
; We do not expect to see this form of IR without -enable-iv-rewrite.

; xfailed for now because the scheduler two-address hack has been disabled.
; Now it's generating a leal -1 rather than a decq.
; XFAIL: *

define void @borf(i8* nocapture %in, i8* nocapture %out) nounwind {
; CHECK-LABEL: borf:
; CHECK-NOT: inc
; CHECK-NOT: leal 1(
; CHECK-NOT: leal -1(
; CHECK: decq
; CHECK-NEXT: cmpq $-478
; CHECK: ret

bb4.thread:
	br label %bb2.outer

bb2.outer:		; preds = %bb4, %bb4.thread
	%indvar19 = phi i64 [ 0, %bb4.thread ], [ %indvar.next29, %bb4 ]		; <i64> [#uses=3]
	%indvar31 = trunc i64 %indvar19 to i16		; <i16> [#uses=1]
	%i.0.reg2mem.0.ph = sub i16 639, %indvar31		; <i16> [#uses=1]
	%0 = zext i16 %i.0.reg2mem.0.ph to i32		; <i32> [#uses=1]
	%1 = mul i32 %0, 480		; <i32> [#uses=1]
	%tmp21 = mul i64 %indvar19, -478		; <i64> [#uses=1]
	br label %bb2

bb2:		; preds = %bb2, %bb2.outer
	%indvar = phi i64 [ 0, %bb2.outer ], [ %indvar.next, %bb2 ]		; <i64> [#uses=3]
	%indvar16 = trunc i64 %indvar to i16		; <i16> [#uses=1]
	%ctg2 = getelementptr i8, i8* %out, i64 %tmp21		; <i8*> [#uses=1]
	%tmp22 = ptrtoint i8* %ctg2 to i64		; <i64> [#uses=1]
	%tmp24 = sub i64 %tmp22, %indvar		; <i64> [#uses=1]
	%out_addr.0.reg2mem.0 = inttoptr i64 %tmp24 to i8*		; <i8*> [#uses=1]
	%j.0.reg2mem.0 = sub i16 479, %indvar16		; <i16> [#uses=1]
	%2 = zext i16 %j.0.reg2mem.0 to i32		; <i32> [#uses=1]
	%3 = add i32 %1, %2		; <i32> [#uses=9]
	%4 = add i32 %3, -481		; <i32> [#uses=1]
	%5 = zext i32 %4 to i64		; <i64> [#uses=1]
	%6 = getelementptr i8, i8* %in, i64 %5		; <i8*> [#uses=1]
	%7 = load i8, i8* %6, align 1		; <i8> [#uses=1]
	%8 = add i32 %3, -480		; <i32> [#uses=1]
	%9 = zext i32 %8 to i64		; <i64> [#uses=1]
	%10 = getelementptr i8, i8* %in, i64 %9		; <i8*> [#uses=1]
	%11 = load i8, i8* %10, align 1		; <i8> [#uses=1]
	%12 = add i32 %3, -479		; <i32> [#uses=1]
	%13 = zext i32 %12 to i64		; <i64> [#uses=1]
	%14 = getelementptr i8, i8* %in, i64 %13		; <i8*> [#uses=1]
	%15 = load i8, i8* %14, align 1		; <i8> [#uses=1]
	%16 = add i32 %3, -1		; <i32> [#uses=1]
	%17 = zext i32 %16 to i64		; <i64> [#uses=1]
	%18 = getelementptr i8, i8* %in, i64 %17		; <i8*> [#uses=1]
	%19 = load i8, i8* %18, align 1		; <i8> [#uses=1]
	%20 = zext i32 %3 to i64		; <i64> [#uses=1]
	%21 = getelementptr i8, i8* %in, i64 %20		; <i8*> [#uses=1]
	%22 = load i8, i8* %21, align 1		; <i8> [#uses=1]
	%23 = add i32 %3, 1		; <i32> [#uses=1]
	%24 = zext i32 %23 to i64		; <i64> [#uses=1]
	%25 = getelementptr i8, i8* %in, i64 %24		; <i8*> [#uses=1]
	%26 = load i8, i8* %25, align 1		; <i8> [#uses=1]
	%27 = add i32 %3, 481		; <i32> [#uses=1]
	%28 = zext i32 %27 to i64		; <i64> [#uses=1]
	%29 = getelementptr i8, i8* %in, i64 %28		; <i8*> [#uses=1]
	%30 = load i8, i8* %29, align 1		; <i8> [#uses=1]
	%31 = add i32 %3, 480		; <i32> [#uses=1]
	%32 = zext i32 %31 to i64		; <i64> [#uses=1]
	%33 = getelementptr i8, i8* %in, i64 %32		; <i8*> [#uses=1]
	%34 = load i8, i8* %33, align 1		; <i8> [#uses=1]
	%35 = add i32 %3, 479		; <i32> [#uses=1]
	%36 = zext i32 %35 to i64		; <i64> [#uses=1]
	%37 = getelementptr i8, i8* %in, i64 %36		; <i8*> [#uses=1]
	%38 = load i8, i8* %37, align 1		; <i8> [#uses=1]
	%39 = add i8 %11, %7		; <i8> [#uses=1]
	%40 = add i8 %39, %15		; <i8> [#uses=1]
	%41 = add i8 %40, %19		; <i8> [#uses=1]
	%42 = add i8 %41, %22		; <i8> [#uses=1]
	%43 = add i8 %42, %26		; <i8> [#uses=1]
	%44 = add i8 %43, %30		; <i8> [#uses=1]
	%45 = add i8 %44, %34		; <i8> [#uses=1]
	%46 = add i8 %45, %38		; <i8> [#uses=1]
	store i8 %46, i8* %out_addr.0.reg2mem.0, align 1
	%indvar.next = add i64 %indvar, 1		; <i64> [#uses=2]
	%exitcond = icmp eq i64 %indvar.next, 478		; <i1> [#uses=1]
	br i1 %exitcond, label %bb4, label %bb2

bb4:		; preds = %bb2
	%indvar.next29 = add i64 %indvar19, 1		; <i64> [#uses=2]
	%exitcond30 = icmp eq i64 %indvar.next29, 638		; <i1> [#uses=1]
	br i1 %exitcond30, label %return, label %bb2.outer

return:		; preds = %bb4
	ret void
}
