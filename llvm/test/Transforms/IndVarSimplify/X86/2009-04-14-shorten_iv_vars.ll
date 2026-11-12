; RUN: opt < %s -passes=indvars -S | FileCheck %s
; ModuleID = '<stdin>'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n32:64"
target triple = "x86_64-apple-darwin9.6"
@a = external global ptr		; <ptr> [#uses=3]
@b = external global ptr		; <ptr> [#uses=3]
@c = external global ptr		; <ptr> [#uses=3]
@d = external global ptr		; <ptr> [#uses=3]
@e = external global ptr		; <ptr> [#uses=3]
@f = external global ptr		; <ptr> [#uses=3]

define void @foo() nounwind {
; CHECK-LABEL: @foo(
; CHECK-NOT: sext
bb1.thread:
	br label %bb1

bb1:		; preds = %bb1, %bb1.thread
	%i.0.reg2mem.0 = phi i32 [ 0, %bb1.thread ], [ %84, %bb1 ]		; <i32> [#uses=19]
	%0 = load ptr, ptr @a, align 8		; <ptr> [#uses=1]
	%1 = load ptr, ptr @b, align 8		; <ptr> [#uses=1]
	%2 = sext i32 %i.0.reg2mem.0 to i64		; <i64> [#uses=1]
	%3 = getelementptr i32, ptr %1, i64 %2		; <ptr> [#uses=1]
	%4 = load i32, ptr %3, align 1		; <i32> [#uses=1]
	%5 = load ptr, ptr @c, align 8		; <ptr> [#uses=1]
	%6 = sext i32 %i.0.reg2mem.0 to i64		; <i64> [#uses=1]
	%7 = getelementptr i32, ptr %5, i64 %6		; <ptr> [#uses=1]
	%8 = load i32, ptr %7, align 1		; <i32> [#uses=1]
	%9 = add i32 %8, %4		; <i32> [#uses=1]
	%10 = sext i32 %i.0.reg2mem.0 to i64		; <i64> [#uses=1]
	%11 = getelementptr i32, ptr %0, i64 %10		; <ptr> [#uses=1]
	store i32 %9, ptr %11, align 1
	%12 = load ptr, ptr @a, align 8		; <ptr> [#uses=1]
	%13 = add i32 %i.0.reg2mem.0, 1		; <i32> [#uses=1]
	%14 = load ptr, ptr @b, align 8		; <ptr> [#uses=1]
	%15 = add i32 %i.0.reg2mem.0, 1		; <i32> [#uses=1]
	%16 = sext i32 %15 to i64		; <i64> [#uses=1]
	%17 = getelementptr i32, ptr %14, i64 %16		; <ptr> [#uses=1]
	%18 = load i32, ptr %17, align 1		; <i32> [#uses=1]
	%19 = load ptr, ptr @c, align 8		; <ptr> [#uses=1]
	%20 = add i32 %i.0.reg2mem.0, 1		; <i32> [#uses=1]
	%21 = sext i32 %20 to i64		; <i64> [#uses=1]
	%22 = getelementptr i32, ptr %19, i64 %21		; <ptr> [#uses=1]
	%23 = load i32, ptr %22, align 1		; <i32> [#uses=1]
	%24 = add i32 %23, %18		; <i32> [#uses=1]
	%25 = sext i32 %13 to i64		; <i64> [#uses=1]
	%26 = getelementptr i32, ptr %12, i64 %25		; <ptr> [#uses=1]
	store i32 %24, ptr %26, align 1
	%27 = load ptr, ptr @a, align 8		; <ptr> [#uses=1]
	%28 = add i32 %i.0.reg2mem.0, 2		; <i32> [#uses=1]
	%29 = load ptr, ptr @b, align 8		; <ptr> [#uses=1]
	%30 = add i32 %i.0.reg2mem.0, 2		; <i32> [#uses=1]
	%31 = sext i32 %30 to i64		; <i64> [#uses=1]
	%32 = getelementptr i32, ptr %29, i64 %31		; <ptr> [#uses=1]
	%33 = load i32, ptr %32, align 1		; <i32> [#uses=1]
	%34 = load ptr, ptr @c, align 8		; <ptr> [#uses=1]
	%35 = add i32 %i.0.reg2mem.0, 2		; <i32> [#uses=1]
	%36 = sext i32 %35 to i64		; <i64> [#uses=1]
	%37 = getelementptr i32, ptr %34, i64 %36		; <ptr> [#uses=1]
	%38 = load i32, ptr %37, align 1		; <i32> [#uses=1]
	%39 = add i32 %38, %33		; <i32> [#uses=1]
	%40 = sext i32 %28 to i64		; <i64> [#uses=1]
	%41 = getelementptr i32, ptr %27, i64 %40		; <ptr> [#uses=1]
	store i32 %39, ptr %41, align 1
	%42 = load ptr, ptr @d, align 8		; <ptr> [#uses=1]
	%43 = load ptr, ptr @e, align 8		; <ptr> [#uses=1]
	%44 = sext i32 %i.0.reg2mem.0 to i64		; <i64> [#uses=1]
	%45 = getelementptr i32, ptr %43, i64 %44		; <ptr> [#uses=1]
	%46 = load i32, ptr %45, align 1		; <i32> [#uses=1]
	%47 = load ptr, ptr @f, align 8		; <ptr> [#uses=1]
	%48 = sext i32 %i.0.reg2mem.0 to i64		; <i64> [#uses=1]
	%49 = getelementptr i32, ptr %47, i64 %48		; <ptr> [#uses=1]
	%50 = load i32, ptr %49, align 1		; <i32> [#uses=1]
	%51 = add i32 %50, %46		; <i32> [#uses=1]
	%52 = sext i32 %i.0.reg2mem.0 to i64		; <i64> [#uses=1]
	%53 = getelementptr i32, ptr %42, i64 %52		; <ptr> [#uses=1]
	store i32 %51, ptr %53, align 1
	%54 = load ptr, ptr @d, align 8		; <ptr> [#uses=1]
	%55 = add i32 %i.0.reg2mem.0, 1		; <i32> [#uses=1]
	%56 = load ptr, ptr @e, align 8		; <ptr> [#uses=1]
	%57 = add i32 %i.0.reg2mem.0, 1		; <i32> [#uses=1]
	%58 = sext i32 %57 to i64		; <i64> [#uses=1]
	%59 = getelementptr i32, ptr %56, i64 %58		; <ptr> [#uses=1]
	%60 = load i32, ptr %59, align 1		; <i32> [#uses=1]
	%61 = load ptr, ptr @f, align 8		; <ptr> [#uses=1]
	%62 = add i32 %i.0.reg2mem.0, 1		; <i32> [#uses=1]
	%63 = sext i32 %62 to i64		; <i64> [#uses=1]
	%64 = getelementptr i32, ptr %61, i64 %63		; <ptr> [#uses=1]
	%65 = load i32, ptr %64, align 1		; <i32> [#uses=1]
	%66 = add i32 %65, %60		; <i32> [#uses=1]
	%67 = sext i32 %55 to i64		; <i64> [#uses=1]
	%68 = getelementptr i32, ptr %54, i64 %67		; <ptr> [#uses=1]
	store i32 %66, ptr %68, align 1
	%69 = load ptr, ptr @d, align 8		; <ptr> [#uses=1]
	%70 = add i32 %i.0.reg2mem.0, 2		; <i32> [#uses=1]
	%71 = load ptr, ptr @e, align 8		; <ptr> [#uses=1]
	%72 = add i32 %i.0.reg2mem.0, 2		; <i32> [#uses=1]
	%73 = sext i32 %72 to i64		; <i64> [#uses=1]
	%74 = getelementptr i32, ptr %71, i64 %73		; <ptr> [#uses=1]
	%75 = load i32, ptr %74, align 1		; <i32> [#uses=1]
	%76 = load ptr, ptr @f, align 8		; <ptr> [#uses=1]
	%77 = add i32 %i.0.reg2mem.0, 2		; <i32> [#uses=1]
	%78 = sext i32 %77 to i64		; <i64> [#uses=1]
	%79 = getelementptr i32, ptr %76, i64 %78		; <ptr> [#uses=1]
	%80 = load i32, ptr %79, align 1		; <i32> [#uses=1]
	%81 = add i32 %80, %75		; <i32> [#uses=1]
	%82 = sext i32 %70 to i64		; <i64> [#uses=1]
	%83 = getelementptr i32, ptr %69, i64 %82		; <ptr> [#uses=1]
	store i32 %81, ptr %83, align 1
	%84 = add i32 %i.0.reg2mem.0, 1		; <i32> [#uses=2]
	%85 = icmp sgt i32 %84, 23646		; <i1> [#uses=1]
	br i1 %85, label %return, label %bb1

return:		; preds = %bb1
	ret void
}
