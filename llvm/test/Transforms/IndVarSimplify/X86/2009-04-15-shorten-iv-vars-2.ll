; RUN: opt < %s -indvars -instcombine -S | FileCheck %s
; ModuleID = '<stdin>'
;extern int *a, *b, *c, *d, *e, *f;  /* 64 bit */
;extern int K[256];
;void foo () {
;  int i;
;  for (i=0; i<23647; i++) {
;    a[(i&15)] = b[i&15]+c[i&15];
;    a[(i+1)&15] = b[(i+1)&15]+c[(i+1)&15];
;    a[(i+2)&15] = b[(i+2)&15]+c[(i+2)&15];
;    d[i&15] = e[i&15]+f[i&15] +K[i];
;    d[(i+1)&15] = e[(i+1)&15]+f[(i+1)&15]+K[i+1];
;    d[(i+2)&15] = e[(i+2)&15]+f[(i+2)&15]+K[i+2];
;  }
;}
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n32:64"
target triple = "x86_64-apple-darwin9.6"
@a = external global i32*		; <i32**> [#uses=3]
@b = external global i32*		; <i32**> [#uses=3]
@c = external global i32*		; <i32**> [#uses=3]
@d = external global i32*		; <i32**> [#uses=3]
@e = external global i32*		; <i32**> [#uses=3]
@f = external global i32*		; <i32**> [#uses=3]
@K = external global [256 x i32]		; <[256 x i32]*> [#uses=3]

define void @foo() nounwind {
; CHECK-LABEL: @foo(
; CHECK-NOT: sext
; CHECK-NOT: zext
bb1.thread:
	br label %bb1

bb1:		; preds = %bb1, %bb1.thread
	%i.0.reg2mem.0 = phi i32 [ 0, %bb1.thread ], [ %116, %bb1 ]		; <i32> [#uses=22]
	%0 = load i32*, i32** @a, align 8		; <i32*> [#uses=1]
	%1 = and i32 %i.0.reg2mem.0, 15		; <i32> [#uses=1]
	%2 = load i32*, i32** @b, align 8		; <i32*> [#uses=1]
	%3 = and i32 %i.0.reg2mem.0, 15		; <i32> [#uses=1]
	%4 = zext i32 %3 to i64		; <i64> [#uses=1]
	%5 = getelementptr i32, i32* %2, i64 %4		; <i32*> [#uses=1]
	%6 = load i32, i32* %5, align 1		; <i32> [#uses=1]
	%7 = load i32*, i32** @c, align 8		; <i32*> [#uses=1]
	%8 = and i32 %i.0.reg2mem.0, 15		; <i32> [#uses=1]
	%9 = zext i32 %8 to i64		; <i64> [#uses=1]
	%10 = getelementptr i32, i32* %7, i64 %9		; <i32*> [#uses=1]
	%11 = load i32, i32* %10, align 1		; <i32> [#uses=1]
	%12 = add i32 %11, %6		; <i32> [#uses=1]
	%13 = zext i32 %1 to i64		; <i64> [#uses=1]
	%14 = getelementptr i32, i32* %0, i64 %13		; <i32*> [#uses=1]
	store i32 %12, i32* %14, align 1
	%15 = load i32*, i32** @a, align 8		; <i32*> [#uses=1]
	%16 = add i32 %i.0.reg2mem.0, 1		; <i32> [#uses=1]
	%17 = and i32 %16, 15		; <i32> [#uses=1]
	%18 = load i32*, i32** @b, align 8		; <i32*> [#uses=1]
	%19 = add i32 %i.0.reg2mem.0, 1		; <i32> [#uses=1]
	%20 = and i32 %19, 15		; <i32> [#uses=1]
	%21 = zext i32 %20 to i64		; <i64> [#uses=1]
	%22 = getelementptr i32, i32* %18, i64 %21		; <i32*> [#uses=1]
	%23 = load i32, i32* %22, align 1		; <i32> [#uses=1]
	%24 = load i32*, i32** @c, align 8		; <i32*> [#uses=1]
	%25 = add i32 %i.0.reg2mem.0, 1		; <i32> [#uses=1]
	%26 = and i32 %25, 15		; <i32> [#uses=1]
	%27 = zext i32 %26 to i64		; <i64> [#uses=1]
	%28 = getelementptr i32, i32* %24, i64 %27		; <i32*> [#uses=1]
	%29 = load i32, i32* %28, align 1		; <i32> [#uses=1]
	%30 = add i32 %29, %23		; <i32> [#uses=1]
	%31 = zext i32 %17 to i64		; <i64> [#uses=1]
	%32 = getelementptr i32, i32* %15, i64 %31		; <i32*> [#uses=1]
	store i32 %30, i32* %32, align 1
	%33 = load i32*, i32** @a, align 8		; <i32*> [#uses=1]
	%34 = add i32 %i.0.reg2mem.0, 2		; <i32> [#uses=1]
	%35 = and i32 %34, 15		; <i32> [#uses=1]
	%36 = load i32*, i32** @b, align 8		; <i32*> [#uses=1]
	%37 = add i32 %i.0.reg2mem.0, 2		; <i32> [#uses=1]
	%38 = and i32 %37, 15		; <i32> [#uses=1]
	%39 = zext i32 %38 to i64		; <i64> [#uses=1]
	%40 = getelementptr i32, i32* %36, i64 %39		; <i32*> [#uses=1]
	%41 = load i32, i32* %40, align 1		; <i32> [#uses=1]
	%42 = load i32*, i32** @c, align 8		; <i32*> [#uses=1]
	%43 = add i32 %i.0.reg2mem.0, 2		; <i32> [#uses=1]
	%44 = and i32 %43, 15		; <i32> [#uses=1]
	%45 = zext i32 %44 to i64		; <i64> [#uses=1]
	%46 = getelementptr i32, i32* %42, i64 %45		; <i32*> [#uses=1]
	%47 = load i32, i32* %46, align 1		; <i32> [#uses=1]
	%48 = add i32 %47, %41		; <i32> [#uses=1]
	%49 = zext i32 %35 to i64		; <i64> [#uses=1]
	%50 = getelementptr i32, i32* %33, i64 %49		; <i32*> [#uses=1]
	store i32 %48, i32* %50, align 1
	%51 = load i32*, i32** @d, align 8		; <i32*> [#uses=1]
	%52 = and i32 %i.0.reg2mem.0, 15		; <i32> [#uses=1]
	%53 = load i32*, i32** @e, align 8		; <i32*> [#uses=1]
	%54 = and i32 %i.0.reg2mem.0, 15		; <i32> [#uses=1]
	%55 = zext i32 %54 to i64		; <i64> [#uses=1]
	%56 = getelementptr i32, i32* %53, i64 %55		; <i32*> [#uses=1]
	%57 = load i32, i32* %56, align 1		; <i32> [#uses=1]
	%58 = load i32*, i32** @f, align 8		; <i32*> [#uses=1]
	%59 = and i32 %i.0.reg2mem.0, 15		; <i32> [#uses=1]
	%60 = zext i32 %59 to i64		; <i64> [#uses=1]
	%61 = getelementptr i32, i32* %58, i64 %60		; <i32*> [#uses=1]
	%62 = load i32, i32* %61, align 1		; <i32> [#uses=1]
	%63 = sext i32 %i.0.reg2mem.0 to i64		; <i64> [#uses=1]
	%64 = getelementptr [256 x i32], [256 x i32]* @K, i64 0, i64 %63		; <i32*> [#uses=1]
	%65 = load i32, i32* %64, align 4		; <i32> [#uses=1]
	%66 = add i32 %62, %57		; <i32> [#uses=1]
	%67 = add i32 %66, %65		; <i32> [#uses=1]
	%68 = zext i32 %52 to i64		; <i64> [#uses=1]
	%69 = getelementptr i32, i32* %51, i64 %68		; <i32*> [#uses=1]
	store i32 %67, i32* %69, align 1
	%70 = load i32*, i32** @d, align 8		; <i32*> [#uses=1]
	%71 = add i32 %i.0.reg2mem.0, 1		; <i32> [#uses=1]
	%72 = and i32 %71, 15		; <i32> [#uses=1]
	%73 = load i32*, i32** @e, align 8		; <i32*> [#uses=1]
	%74 = add i32 %i.0.reg2mem.0, 1		; <i32> [#uses=1]
	%75 = and i32 %74, 15		; <i32> [#uses=1]
	%76 = zext i32 %75 to i64		; <i64> [#uses=1]
	%77 = getelementptr i32, i32* %73, i64 %76		; <i32*> [#uses=1]
	%78 = load i32, i32* %77, align 1		; <i32> [#uses=1]
	%79 = load i32*, i32** @f, align 8		; <i32*> [#uses=1]
	%80 = add i32 %i.0.reg2mem.0, 1		; <i32> [#uses=1]
	%81 = and i32 %80, 15		; <i32> [#uses=1]
	%82 = zext i32 %81 to i64		; <i64> [#uses=1]
	%83 = getelementptr i32, i32* %79, i64 %82		; <i32*> [#uses=1]
	%84 = load i32, i32* %83, align 1		; <i32> [#uses=1]
	%85 = add i32 %i.0.reg2mem.0, 1		; <i32> [#uses=1]
	%86 = sext i32 %85 to i64		; <i64> [#uses=1]
	%87 = getelementptr [256 x i32], [256 x i32]* @K, i64 0, i64 %86		; <i32*> [#uses=1]
	%88 = load i32, i32* %87, align 4		; <i32> [#uses=1]
	%89 = add i32 %84, %78		; <i32> [#uses=1]
	%90 = add i32 %89, %88		; <i32> [#uses=1]
	%91 = zext i32 %72 to i64		; <i64> [#uses=1]
	%92 = getelementptr i32, i32* %70, i64 %91		; <i32*> [#uses=1]
	store i32 %90, i32* %92, align 1
	%93 = load i32*, i32** @d, align 8		; <i32*> [#uses=1]
	%94 = add i32 %i.0.reg2mem.0, 2		; <i32> [#uses=1]
	%95 = and i32 %94, 15		; <i32> [#uses=1]
	%96 = load i32*, i32** @e, align 8		; <i32*> [#uses=1]
	%97 = add i32 %i.0.reg2mem.0, 2		; <i32> [#uses=1]
	%98 = and i32 %97, 15		; <i32> [#uses=1]
	%99 = zext i32 %98 to i64		; <i64> [#uses=1]
	%100 = getelementptr i32, i32* %96, i64 %99		; <i32*> [#uses=1]
	%101 = load i32, i32* %100, align 1		; <i32> [#uses=1]
	%102 = load i32*, i32** @f, align 8		; <i32*> [#uses=1]
	%103 = add i32 %i.0.reg2mem.0, 2		; <i32> [#uses=1]
	%104 = and i32 %103, 15		; <i32> [#uses=1]
	%105 = zext i32 %104 to i64		; <i64> [#uses=1]
	%106 = getelementptr i32, i32* %102, i64 %105		; <i32*> [#uses=1]
	%107 = load i32, i32* %106, align 1		; <i32> [#uses=1]
	%108 = add i32 %i.0.reg2mem.0, 2		; <i32> [#uses=1]
	%109 = sext i32 %108 to i64		; <i64> [#uses=1]
	%110 = getelementptr [256 x i32], [256 x i32]* @K, i64 0, i64 %109		; <i32*> [#uses=1]
	%111 = load i32, i32* %110, align 4		; <i32> [#uses=1]
	%112 = add i32 %107, %101		; <i32> [#uses=1]
	%113 = add i32 %112, %111		; <i32> [#uses=1]
	%114 = zext i32 %95 to i64		; <i64> [#uses=1]
	%115 = getelementptr i32, i32* %93, i64 %114		; <i32*> [#uses=1]
	store i32 %113, i32* %115, align 1
	%116 = add i32 %i.0.reg2mem.0, 1		; <i32> [#uses=2]
	%117 = icmp sgt i32 %116, 23646		; <i1> [#uses=1]
	br i1 %117, label %return, label %bb1

return:		; preds = %bb1
	ret void
}
