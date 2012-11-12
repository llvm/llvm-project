; RUN: opt < %s -instcombine -S | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"
@C.0.1248 = internal constant [128 x float] [ float -1.000000e+00, float -1.000000e+00, float -1.000000e+00, float 0.000000e+00, float -1.000000e+00, float -1.000000e+00, float 0.000000e+00, float -1.000000e+00, float -1.000000e+00, float -1.000000e+00, float 0.000000e+00, float 1.000000e+00, float -1.000000e+00, float -1.000000e+00, float 1.000000e+00, float 0.000000e+00, float -1.000000e+00, float 0.000000e+00, float -1.000000e+00, float -1.000000e+00, float -1.000000e+00, float 0.000000e+00, float -1.000000e+00, float 1.000000e+00, float -1.000000e+00, float 0.000000e+00, float 1.000000e+00, float -1.000000e+00, float -1.000000e+00, float 0.000000e+00, float 1.000000e+00, float 1.000000e+00, float -1.000000e+00, float 1.000000e+00, float -1.000000e+00, float 0.000000e+00, float -1.000000e+00, float 1.000000e+00, float 0.000000e+00, float -1.000000e+00, float -1.000000e+00, float 1.000000e+00, float 0.000000e+00, float 1.000000e+00, float -1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 0.000000e+00, float 0.000000e+00, float -1.000000e+00, float -1.000000e+00, float -1.000000e+00, float 0.000000e+00, float -1.000000e+00, float -1.000000e+00, float 1.000000e+00, float 0.000000e+00, float -1.000000e+00, float 1.000000e+00, float -1.000000e+00, float 0.000000e+00, float -1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float -1.000000e+00, float -1.000000e+00, float 0.000000e+00, float 1.000000e+00, float -1.000000e+00, float 0.000000e+00, float -1.000000e+00, float 1.000000e+00, float -1.000000e+00, float 0.000000e+00, float 1.000000e+00, float 1.000000e+00, float -1.000000e+00, float 1.000000e+00, float 0.000000e+00, float 1.000000e+00, float 0.000000e+00, float -1.000000e+00, float -1.000000e+00, float 1.000000e+00, float 0.000000e+00, float -1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 0.000000e+00, float 1.000000e+00, float -1.000000e+00, float 1.000000e+00, float 0.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float -1.000000e+00, float 0.000000e+00, float 1.000000e+00, float 1.000000e+00, float 0.000000e+00, float -1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 0.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 0.000000e+00, float 0.000000e+00, float 1.000000e+00, float -1.000000e+00, float -1.000000e+00, float 0.000000e+00, float 1.000000e+00, float -1.000000e+00, float 1.000000e+00, float 0.000000e+00, float 1.000000e+00, float 1.000000e+00, float -1.000000e+00, float 0.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00 ], align 32		; <[128 x float]*> [#uses=1]

define float @test1(i32 %hash, float %x, float %y, float %z, float %w) {
entry:
	%lookupTable = alloca [128 x float], align 16		; <[128 x float]*> [#uses=5]
	%lookupTable1 = bitcast [128 x float]* %lookupTable to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.p0i8.p0i8.i64(i8* %lookupTable1, i8* bitcast ([128 x float]* @C.0.1248 to i8*), i64 512, i32 16, i1 false)
        
; CHECK: @test1
; CHECK-NOT: alloca
; CHECK-NOT: call{{.*}}@llvm.memcpy
        
	%tmp3 = shl i32 %hash, 2		; <i32> [#uses=1]
	%tmp5 = and i32 %tmp3, 124		; <i32> [#uses=4]
	%tmp753 = getelementptr [128 x float]* %lookupTable, i32 0, i32 %tmp5		; <float*> [#uses=1]
	%tmp9 = load float* %tmp753		; <float> [#uses=1]
	%tmp11 = fmul float %tmp9, %x		; <float> [#uses=1]
	%tmp13 = fadd float %tmp11, 0.000000e+00		; <float> [#uses=1]
	%tmp17.sum52 = or i32 %tmp5, 1		; <i32> [#uses=1]
	%tmp1851 = getelementptr [128 x float]* %lookupTable, i32 0, i32 %tmp17.sum52		; <float*> [#uses=1]
	%tmp19 = load float* %tmp1851		; <float> [#uses=1]
	%tmp21 = fmul float %tmp19, %y		; <float> [#uses=1]
	%tmp23 = fadd float %tmp21, %tmp13		; <float> [#uses=1]
	%tmp27.sum50 = or i32 %tmp5, 2		; <i32> [#uses=1]
	%tmp2849 = getelementptr [128 x float]* %lookupTable, i32 0, i32 %tmp27.sum50		; <float*> [#uses=1]
	%tmp29 = load float* %tmp2849		; <float> [#uses=1]
	%tmp31 = fmul float %tmp29, %z		; <float> [#uses=1]
	%tmp33 = fadd float %tmp31, %tmp23		; <float> [#uses=1]
	%tmp37.sum48 = or i32 %tmp5, 3		; <i32> [#uses=1]
	%tmp3847 = getelementptr [128 x float]* %lookupTable, i32 0, i32 %tmp37.sum48		; <float*> [#uses=1]
	%tmp39 = load float* %tmp3847		; <float> [#uses=1]
	%tmp41 = fmul float %tmp39, %w		; <float> [#uses=1]
	%tmp43 = fadd float %tmp41, %tmp33		; <float> [#uses=1]
	ret float %tmp43
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1) nounwind

%T = type { i8, [123 x i8] }
%U = type { i32, i32, i32, i32, i32 }

@G = constant %T {i8 1, [123 x i8] zeroinitializer }
@H = constant [2 x %U] zeroinitializer, align 16

define void @test2() {
  %A = alloca %T
  %B = alloca %T
  %a = bitcast %T* %A to i8*
  %b = bitcast %T* %B to i8*

; CHECK: @test2

; %A alloca is deleted
; CHECK-NEXT: alloca [124 x i8]
; CHECK-NEXT: getelementptr inbounds [124 x i8]*

; use @G instead of %A
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* %{{.*}}, i8* getelementptr inbounds (%T* @G, i64 0, i32 0)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %a, i8* bitcast (%T* @G to i8*), i64 124, i32 4, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %b, i8* %a, i64 124, i32 4, i1 false)
  call void @bar(i8* %b)
  ret void
}

declare void @bar(i8*)


;; Should be able to eliminate the alloca.
define void @test3() {
  %A = alloca %T
  %a = bitcast %T* %A to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %a, i8* bitcast (%T* @G to i8*), i64 124, i32 4, i1 false)
  call void @bar(i8* %a) readonly
; CHECK: @test3
; CHECK-NEXT: call void @bar(i8* getelementptr inbounds (%T* @G, i64 0, i32 0))
  ret void
}

define void @test4() {
  %A = alloca %T
  %a = bitcast %T* %A to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %a, i8* bitcast (%T* @G to i8*), i64 124, i32 4, i1 false)
  call void @baz(i8* byval %a) 
; CHECK: @test4
; CHECK-NEXT: call void @baz(i8* byval getelementptr inbounds (%T* @G, i64 0, i32 0))
  ret void
}

declare void @llvm.lifetime.start(i64, i8*)
define void @test5() {
  %A = alloca %T
  %a = bitcast %T* %A to i8*
  call void @llvm.lifetime.start(i64 -1, i8* %a)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %a, i8* bitcast (%T* @G to i8*), i64 124, i32 4, i1 false)
  call void @baz(i8* byval %a) 
; CHECK: @test5
; CHECK-NEXT: call void @baz(i8* byval getelementptr inbounds (%T* @G, i64 0, i32 0))
  ret void
}


declare void @baz(i8* byval)


define void @test6() {
  %A = alloca %U, align 16
  %a = bitcast %U* %A to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %a, i8* bitcast ([2 x %U]* @H to i8*), i64 20, i32 16, i1 false)
  call void @bar(i8* %a) readonly
; CHECK: @test6
; CHECK-NEXT: call void @bar(i8* bitcast ([2 x %U]* @H to i8*))
  ret void
}

define void @test7() {
  %A = alloca %U, align 16
  %a = bitcast %U* %A to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %a, i8* bitcast (%U* getelementptr ([2 x %U]* @H, i64 0, i32 0) to i8*), i64 20, i32 4, i1 false)
  call void @bar(i8* %a) readonly
; CHECK: @test7
; CHECK-NEXT: call void @bar(i8* bitcast ([2 x %U]* @H to i8*))
  ret void
}

define void @test8() {
  %A = alloca %U, align 16
  %a = bitcast %U* %A to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %a, i8* bitcast (%U* getelementptr ([2 x %U]* @H, i64 0, i32 1) to i8*), i64 20, i32 4, i1 false)
  call void @bar(i8* %a) readonly
; CHECK: @test8
; CHECK: llvm.memcpy
; CHECK: bar
  ret void
}
