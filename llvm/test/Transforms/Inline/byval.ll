; RUN: opt < %s -inline -S | FileCheck %s
; RUN: opt < %s -passes='cgscc(inline)' -S | FileCheck %s

target datalayout = "p:32:32-p1:64:64-p2:16:16-n16:32:64"

; Inlining a byval struct should cause an explicit copy into an alloca.

	%struct.ss = type { i32, i64 }
@.str = internal constant [10 x i8] c"%d, %lld\0A\00"		; <[10 x i8]*> [#uses=1]

define internal void @f(%struct.ss* byval  %b) nounwind  {
entry:
	%tmp = getelementptr %struct.ss, %struct.ss* %b, i32 0, i32 0		; <i32*> [#uses=2]
	%tmp1 = load i32, i32* %tmp, align 4		; <i32> [#uses=1]
	%tmp2 = add i32 %tmp1, 1		; <i32> [#uses=1]
	store i32 %tmp2, i32* %tmp, align 4
	ret void
}

declare i32 @printf(i8*, ...) nounwind 

define i32 @test1() nounwind  {
entry:
	%S = alloca %struct.ss		; <%struct.ss*> [#uses=4]
	%tmp1 = getelementptr %struct.ss, %struct.ss* %S, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 1, i32* %tmp1, align 8
	%tmp4 = getelementptr %struct.ss, %struct.ss* %S, i32 0, i32 1		; <i64*> [#uses=1]
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
	%tmp = getelementptr %struct.ss, %struct.ss* %b, i32 0, i32 0		; <i32*> [#uses=2]
	%tmp1 = load i32, i32* %tmp, align 4		; <i32> [#uses=1]
	%tmp2 = add i32 %tmp1, 1		; <i32> [#uses=1]
	ret i32 %tmp2
}

define i32 @test2() nounwind  {
entry:
	%S = alloca %struct.ss		; <%struct.ss*> [#uses=4]
	%tmp1 = getelementptr %struct.ss, %struct.ss* %S, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 1, i32* %tmp1, align 8
	%tmp4 = getelementptr %struct.ss, %struct.ss* %S, i32 0, i32 1		; <i64*> [#uses=1]
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

%struct.S0 = type { i32 }

@b = global %struct.S0 { i32 1 }, align 4
@a = common global i32 0, align 4

define internal void @f5(%struct.S0* byval nocapture readonly align 4 %p) {
entry:
	store i32 0, i32* getelementptr inbounds (%struct.S0, %struct.S0* @b, i64 0, i32 0), align 4
	%f2 = getelementptr inbounds %struct.S0, %struct.S0* %p, i64 0, i32 0
	%0 = load i32, i32* %f2, align 4
	store i32 %0, i32* @a, align 4
	ret void
}

define i32 @test5() {
entry:
	tail call void @f5(%struct.S0* byval align 4 @b)
	%0 = load i32, i32* @a, align 4
	ret i32 %0
; CHECK: @test5()
; CHECK: store i32 0, i32* getelementptr inbounds (%struct.S0, %struct.S0* @b, i64 0, i32 0), align 4
; CHECK-NOT: load i32, i32* getelementptr inbounds (%struct.S0, %struct.S0* @b, i64 0, i32 0), align 4
}

%struct.S1 = type { i32 }

@d = addrspace(1) global %struct.S1 { i32 1 }, align 4
@c = common addrspace(1) global i32 0, align 4

define internal void @f5_as1(%struct.S1 addrspace(1)* byval nocapture readonly align 4 %p) {
entry:
	store i32 0, i32 addrspace(1)* getelementptr inbounds (%struct.S1, %struct.S1 addrspace(1)* @d, i64 0, i32 0), align 4
	%f2 = getelementptr inbounds %struct.S1, %struct.S1 addrspace(1)* %p, i64 0, i32 0
	%0 = load i32, i32 addrspace(1)* %f2, align 4
	store i32 %0, i32 addrspace(1)* @c, align 4
	ret void
}

define i32 @test5_as1() {
entry:
	tail call void @f5_as1(%struct.S1 addrspace(1)* byval align 4 @d)
	%0 = load i32, i32 addrspace(1)* @c, align 4
	ret i32 %0
; CHECK: @test5_as1()
; CHECK: store i32 0, i32 addrspace(1)* getelementptr inbounds (%struct.S1, %struct.S1 addrspace(1)* @d, i64 0, i32 0), align 4
; CHECK-NOT: load i32, i32 addrspace(1)* getelementptr inbounds (%struct.S1, %struct.S1 addrspace(1)* @d, i64 0, i32 0), align 4
}
