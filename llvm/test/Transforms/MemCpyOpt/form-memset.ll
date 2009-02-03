; RUN: llvm-as < %s | opt -memcpyopt | llvm-dis | not grep store
; RUN: llvm-as < %s | opt -memcpyopt | llvm-dis | grep {call.*llvm.memset}

; All the stores in this example should be merged into a single memset.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin8"

define void @foo(i8 signext  %c) nounwind  {
entry:
	%x = alloca [19 x i8]		; <[19 x i8]*> [#uses=20]
	%tmp = getelementptr [19 x i8]* %x, i32 0, i32 0		; <i8*> [#uses=1]
	store i8 %c, i8* %tmp, align 1
	%tmp5 = getelementptr [19 x i8]* %x, i32 0, i32 1		; <i8*> [#uses=1]
	store i8 %c, i8* %tmp5, align 1
	%tmp9 = getelementptr [19 x i8]* %x, i32 0, i32 2		; <i8*> [#uses=1]
	store i8 %c, i8* %tmp9, align 1
	%tmp13 = getelementptr [19 x i8]* %x, i32 0, i32 3		; <i8*> [#uses=1]
	store i8 %c, i8* %tmp13, align 1
	%tmp17 = getelementptr [19 x i8]* %x, i32 0, i32 4		; <i8*> [#uses=1]
	store i8 %c, i8* %tmp17, align 1
	%tmp21 = getelementptr [19 x i8]* %x, i32 0, i32 5		; <i8*> [#uses=1]
	store i8 %c, i8* %tmp21, align 1
	%tmp25 = getelementptr [19 x i8]* %x, i32 0, i32 6		; <i8*> [#uses=1]
	store i8 %c, i8* %tmp25, align 1
	%tmp29 = getelementptr [19 x i8]* %x, i32 0, i32 7		; <i8*> [#uses=1]
	store i8 %c, i8* %tmp29, align 1
	%tmp33 = getelementptr [19 x i8]* %x, i32 0, i32 8		; <i8*> [#uses=1]
	store i8 %c, i8* %tmp33, align 1
	%tmp37 = getelementptr [19 x i8]* %x, i32 0, i32 9		; <i8*> [#uses=1]
	store i8 %c, i8* %tmp37, align 1
	%tmp41 = getelementptr [19 x i8]* %x, i32 0, i32 10		; <i8*> [#uses=1]
	store i8 %c, i8* %tmp41, align 1
	%tmp45 = getelementptr [19 x i8]* %x, i32 0, i32 11		; <i8*> [#uses=1]
	store i8 %c, i8* %tmp45, align 1
	%tmp49 = getelementptr [19 x i8]* %x, i32 0, i32 12		; <i8*> [#uses=1]
	store i8 %c, i8* %tmp49, align 1
	%tmp53 = getelementptr [19 x i8]* %x, i32 0, i32 13		; <i8*> [#uses=1]
	store i8 %c, i8* %tmp53, align 1
	%tmp57 = getelementptr [19 x i8]* %x, i32 0, i32 14		; <i8*> [#uses=1]
	store i8 %c, i8* %tmp57, align 1
	%tmp61 = getelementptr [19 x i8]* %x, i32 0, i32 15		; <i8*> [#uses=1]
	store i8 %c, i8* %tmp61, align 1
	%tmp65 = getelementptr [19 x i8]* %x, i32 0, i32 16		; <i8*> [#uses=1]
	store i8 %c, i8* %tmp65, align 1
	%tmp69 = getelementptr [19 x i8]* %x, i32 0, i32 17		; <i8*> [#uses=1]
	store i8 %c, i8* %tmp69, align 1
	%tmp73 = getelementptr [19 x i8]* %x, i32 0, i32 18		; <i8*> [#uses=1]
	store i8 %c, i8* %tmp73, align 1
	%tmp76 = call i32 (...)* @bar( [19 x i8]* %x ) nounwind 		; <i32> [#uses=0]
	ret void
}

declare i32 @bar(...)

