; RUN: opt < %s -inline -S | FileCheck %s

; Inlining a byval struct should cause an explicit copy into an alloca.

	%struct.ss = type { i32, i64 }
@.str = internal constant [10 x i8] c"%d, %lld\0A\00"		; <[10 x i8]*> [#uses=1]

define internal void @f(%struct.ss* byval  %b) nounwind  {
entry:
	%tmp = getelementptr %struct.ss* %b, i32 0, i32 0		; <i32*> [#uses=2]
	%tmp1 = load i32* %tmp, align 4		; <i32> [#uses=1]
	%tmp2 = add i32 %tmp1, 1		; <i32> [#uses=1]
	store i32 %tmp2, i32* %tmp, align 4
	ret void
}

declare i32 @printf(i8*, ...) nounwind 

define i32 @test1() nounwind  {
entry:
	%S = alloca %struct.ss		; <%struct.ss*> [#uses=4]
	%tmp1 = getelementptr %struct.ss* %S, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 1, i32* %tmp1, align 8
	%tmp4 = getelementptr %struct.ss* %S, i32 0, i32 1		; <i64*> [#uses=1]
	store i64 2, i64* %tmp4, align 4
	call void @f( %struct.ss* byval  %S ) nounwind 
	ret i32 0
; CHECK: @test1()
; CHECK: %S1 = alloca %struct.ss
; CHECK: %S = alloca %struct.ss
; CHECK: call void @llvm.memcpy
; CHECK: ret i32 0
}

; Inlining a byval struct should NOT cause an explicit copy 
; into an alloca if the function is readonly

define internal i32 @f2(%struct.ss* byval  %b) nounwind readonly {
entry:
	%tmp = getelementptr %struct.ss* %b, i32 0, i32 0		; <i32*> [#uses=2]
	%tmp1 = load i32* %tmp, align 4		; <i32> [#uses=1]
	%tmp2 = add i32 %tmp1, 1		; <i32> [#uses=1]
	ret i32 %tmp2
}

define i32 @test2() nounwind  {
entry:
	%S = alloca %struct.ss		; <%struct.ss*> [#uses=4]
	%tmp1 = getelementptr %struct.ss* %S, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 1, i32* %tmp1, align 8
	%tmp4 = getelementptr %struct.ss* %S, i32 0, i32 1		; <i64*> [#uses=1]
	store i64 2, i64* %tmp4, align 4
	%X = call i32 @f2( %struct.ss* byval  %S ) nounwind 
	ret i32 %X
; CHECK: @test2()
; CHECK: %S = alloca %struct.ss
; CHECK-NOT: call void @llvm.memcpy
; CHECK: ret i32
}


; Inlining a byval with an explicit alignment needs to use *at least* that
; alignment on the generated alloca.
; PR8769
declare void @g3(%struct.ss* %p)

define internal void @f3(%struct.ss* byval align 64 %b) nounwind {
   call void @g3(%struct.ss* %b)  ;; Could make alignment assumptions!
   ret void
}

define void @test3() nounwind  {
entry:
	%S = alloca %struct.ss, align 1  ;; May not be aligned.
	call void @f3( %struct.ss* byval align 64 %S) nounwind 
	ret void
; CHECK: @test3()
; CHECK: %S1 = alloca %struct.ss, align 64
; CHECK: %S = alloca %struct.ss
; CHECK: call void @llvm.memcpy
; CHECK: call void @g3(%struct.ss* %S1)
; CHECK: ret void
}


; Inlining a byval struct should NOT cause an explicit copy 
; into an alloca if the function is readonly, but should increase an alloca's
; alignment to satisfy an explicit alignment request.

define internal i32 @f4(%struct.ss* byval align 64 %b) nounwind readonly {
        call void @g3(%struct.ss* %b)
	ret i32 4
}

define i32 @test4() nounwind  {
entry:
	%S = alloca %struct.ss, align 2		; <%struct.ss*> [#uses=4]
	%X = call i32 @f4( %struct.ss* byval align 64 %S ) nounwind 
	ret i32 %X
; CHECK: @test4()
; CHECK: %S = alloca %struct.ss, align 64
; CHECK-NOT: call void @llvm.memcpy
; CHECK: call void @g3
; CHECK: ret i32 4
}

