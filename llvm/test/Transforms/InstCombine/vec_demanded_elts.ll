; RUN: opt < %s -instcombine -S | FileCheck %s

define i16 @test1(float %f) {
entry:
; CHECK: @test1
; CHECK: fmul float
; CHECK-NOT: insertelement {{.*}} 0.00
; CHECK-NOT: call {{.*}} @llvm.x86.sse.mul
; CHECK-NOT: call {{.*}} @llvm.x86.sse.sub
; CHECK: ret
	%tmp = insertelement <4 x float> undef, float %f, i32 0		; <<4 x float>> [#uses=1]
	%tmp10 = insertelement <4 x float> %tmp, float 0.000000e+00, i32 1		; <<4 x float>> [#uses=1]
	%tmp11 = insertelement <4 x float> %tmp10, float 0.000000e+00, i32 2		; <<4 x float>> [#uses=1]
	%tmp12 = insertelement <4 x float> %tmp11, float 0.000000e+00, i32 3		; <<4 x float>> [#uses=1]
	%tmp28 = tail call <4 x float> @llvm.x86.sse.sub.ss( <4 x float> %tmp12, <4 x float> < float 1.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00 > )		; <<4 x float>> [#uses=1]
	%tmp37 = tail call <4 x float> @llvm.x86.sse.mul.ss( <4 x float> %tmp28, <4 x float> < float 5.000000e-01, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00 > )		; <<4 x float>> [#uses=1]
	%tmp48 = tail call <4 x float> @llvm.x86.sse.min.ss( <4 x float> %tmp37, <4 x float> < float 6.553500e+04, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00 > )		; <<4 x float>> [#uses=1]
	%tmp59 = tail call <4 x float> @llvm.x86.sse.max.ss( <4 x float> %tmp48, <4 x float> zeroinitializer )		; <<4 x float>> [#uses=1]
	%tmp.upgrd.1 = tail call i32 @llvm.x86.sse.cvttss2si( <4 x float> %tmp59 )		; <i32> [#uses=1]
	%tmp69 = trunc i32 %tmp.upgrd.1 to i16		; <i16> [#uses=1]
	ret i16 %tmp69
}

define i32 @test2(float %f) {
; CHECK: @test2
; CHECK-NOT: insertelement
; CHECK-NOT: extractelement
; CHECK: ret
  %tmp5 = fmul float %f, %f
  %tmp9 = insertelement <4 x float> undef, float %tmp5, i32 0
  %tmp10 = insertelement <4 x float> %tmp9, float 0.000000e+00, i32 1
  %tmp11 = insertelement <4 x float> %tmp10, float 0.000000e+00, i32 2
  %tmp12 = insertelement <4 x float> %tmp11, float 0.000000e+00, i32 3
  %tmp19 = bitcast <4 x float> %tmp12 to <4 x i32>
  %tmp21 = extractelement <4 x i32> %tmp19, i32 0
  ret i32 %tmp21
}

define i64 @test3(float %f, double %d) {
; CHECK: @test3
; CHECK-NOT: insertelement {{.*}} 0.00
; CHECK: ret
entry:
  %v00 = insertelement <4 x float> undef, float %f, i32 0
  %v01 = insertelement <4 x float> %v00, float 0.000000e+00, i32 1
  %v02 = insertelement <4 x float> %v01, float 0.000000e+00, i32 2
  %v03 = insertelement <4 x float> %v02, float 0.000000e+00, i32 3
  %tmp0 = tail call i32 @llvm.x86.sse.cvtss2si(<4 x float> %v03)
  %v10 = insertelement <4 x float> undef, float %f, i32 0
  %v11 = insertelement <4 x float> %v10, float 0.000000e+00, i32 1
  %v12 = insertelement <4 x float> %v11, float 0.000000e+00, i32 2
  %v13 = insertelement <4 x float> %v12, float 0.000000e+00, i32 3
  %tmp1 = tail call i64 @llvm.x86.sse.cvtss2si64(<4 x float> %v13)
  %v20 = insertelement <4 x float> undef, float %f, i32 0
  %v21 = insertelement <4 x float> %v20, float 0.000000e+00, i32 1
  %v22 = insertelement <4 x float> %v21, float 0.000000e+00, i32 2
  %v23 = insertelement <4 x float> %v22, float 0.000000e+00, i32 3
  %tmp2 = tail call i32 @llvm.x86.sse.cvttss2si(<4 x float> %v23)
  %v30 = insertelement <4 x float> undef, float %f, i32 0
  %v31 = insertelement <4 x float> %v30, float 0.000000e+00, i32 1
  %v32 = insertelement <4 x float> %v31, float 0.000000e+00, i32 2
  %v33 = insertelement <4 x float> %v32, float 0.000000e+00, i32 3
  %tmp3 = tail call i64 @llvm.x86.sse.cvttss2si64(<4 x float> %v33)
  %v40 = insertelement <2 x double> undef, double %d, i32 0
  %v41 = insertelement <2 x double> %v40, double 0.000000e+00, i32 1
  %tmp4 = tail call i32 @llvm.x86.sse2.cvtsd2si(<2 x double> %v41)
  %v50 = insertelement <2 x double> undef, double %d, i32 0
  %v51 = insertelement <2 x double> %v50, double 0.000000e+00, i32 1
  %tmp5 = tail call i64 @llvm.x86.sse2.cvtsd2si64(<2 x double> %v51)
  %v60 = insertelement <2 x double> undef, double %d, i32 0
  %v61 = insertelement <2 x double> %v60, double 0.000000e+00, i32 1
  %tmp6 = tail call i32 @llvm.x86.sse2.cvttsd2si(<2 x double> %v61)
  %v70 = insertelement <2 x double> undef, double %d, i32 0
  %v71 = insertelement <2 x double> %v70, double 0.000000e+00, i32 1
  %tmp7 = tail call i64 @llvm.x86.sse2.cvttsd2si64(<2 x double> %v71)
  %tmp8 = add i32 %tmp0, %tmp2
  %tmp9 = add i32 %tmp4, %tmp6
  %tmp10 = add i32 %tmp8, %tmp9
  %tmp11 = sext i32 %tmp10 to i64
  %tmp12 = add i64 %tmp1, %tmp3
  %tmp13 = add i64 %tmp5, %tmp7
  %tmp14 = add i64 %tmp12, %tmp13
  %tmp15 = add i64 %tmp11, %tmp14
  ret i64 %tmp15
}

define void @get_image() nounwind {
; CHECK: @get_image
; CHECK-NOT: extractelement
; CHECK: unreachable
entry:
  %0 = call i32 @fgetc(i8* null) nounwind               ; <i32> [#uses=1]
  %1 = trunc i32 %0 to i8         ; <i8> [#uses=1]
  %tmp2 = insertelement <100 x i8> zeroinitializer, i8 %1, i32 1          ; <<100 x i8>> [#uses=1]
  %tmp1 = extractelement <100 x i8> %tmp2, i32 0          ; <i8> [#uses=1]
  %2 = icmp eq i8 %tmp1, 80               ; <i1> [#uses=1]
  br i1 %2, label %bb2, label %bb3

bb2:            ; preds = %entry
  br label %bb3

bb3:            ; preds = %bb2, %entry
  unreachable
}

; PR4340
define void @vac(<4 x float>* nocapture %a) nounwind {
; CHECK: @vac
; CHECK-NOT: load
; CHECK: ret
entry:
	%tmp1 = load <4 x float>* %a		; <<4 x float>> [#uses=1]
	%vecins = insertelement <4 x float> %tmp1, float 0.000000e+00, i32 0	; <<4 x float>> [#uses=1]
	%vecins4 = insertelement <4 x float> %vecins, float 0.000000e+00, i32 1; <<4 x float>> [#uses=1]
	%vecins6 = insertelement <4 x float> %vecins4, float 0.000000e+00, i32 2; <<4 x float>> [#uses=1]
	%vecins8 = insertelement <4 x float> %vecins6, float 0.000000e+00, i32 3; <<4 x float>> [#uses=1]
	store <4 x float> %vecins8, <4 x float>* %a
	ret void
}

declare i32 @fgetc(i8*)

declare <4 x float> @llvm.x86.sse.sub.ss(<4 x float>, <4 x float>)

declare <4 x float> @llvm.x86.sse.mul.ss(<4 x float>, <4 x float>)

declare <4 x float> @llvm.x86.sse.min.ss(<4 x float>, <4 x float>)

declare <4 x float> @llvm.x86.sse.max.ss(<4 x float>, <4 x float>)

declare i32 @llvm.x86.sse.cvtss2si(<4 x float>)
declare i64 @llvm.x86.sse.cvtss2si64(<4 x float>)
declare i32 @llvm.x86.sse.cvttss2si(<4 x float>)
declare i64 @llvm.x86.sse.cvttss2si64(<4 x float>)
declare i32 @llvm.x86.sse2.cvtsd2si(<2 x double>)
declare i64 @llvm.x86.sse2.cvtsd2si64(<2 x double>)
declare i32 @llvm.x86.sse2.cvttsd2si(<2 x double>)
declare i64 @llvm.x86.sse2.cvttsd2si64(<2 x double>)

; <rdar://problem/6945110>
define <4 x i32> @kernel3_vertical(<4 x i16> * %src, <8 x i16> * %foo) nounwind {
entry:
	%tmp = load <4 x i16>* %src
	%tmp1 = load <8 x i16>* %foo
; CHECK: %tmp2 = shufflevector
	%tmp2 = shufflevector <4 x i16> %tmp, <4 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
; pmovzxwd ignores the upper 64-bits of its input; -instcombine should remove this shuffle:
; CHECK-NOT: shufflevector
	%tmp3 = shufflevector <8 x i16> %tmp1, <8 x i16> %tmp2, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 4, i32 5, i32 6, i32 7>
; CHECK-NEXT: pmovzxwd
	%0 = call <4 x i32> @llvm.x86.sse41.pmovzxwd(<8 x i16> %tmp3)
	ret <4 x i32> %0
}
declare <4 x i32> @llvm.x86.sse41.pmovzxwd(<8 x i16>) nounwind readnone

define <4 x float> @dead_shuffle_elt(<4 x float> %x, <2 x float> %y) nounwind {
entry:
; CHECK: define <4 x float> @dead_shuffle_elt
; CHECK: shufflevector <2 x float> %y, <2 x float> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
  %shuffle.i = shufflevector <2 x float> %y, <2 x float> %y, <4 x i32> <i32 0, i32 1, i32 0, i32 1>
  %shuffle9.i = shufflevector <4 x float> %x, <4 x float> %shuffle.i, <4 x i32> <i32 4, i32 5, i32 2, i32 3>
  ret <4 x float> %shuffle9.i
}

define <2 x float> @test_fptrunc(double %f) {
; CHECK: @test_fptrunc
; CHECK: insertelement
; CHECK: insertelement
; CHECK-NOT: insertelement
  %tmp9 = insertelement <4 x double> undef, double %f, i32 0
  %tmp10 = insertelement <4 x double> %tmp9, double 0.000000e+00, i32 1
  %tmp11 = insertelement <4 x double> %tmp10, double 0.000000e+00, i32 2
  %tmp12 = insertelement <4 x double> %tmp11, double 0.000000e+00, i32 3
  %tmp5 = fptrunc <4 x double> %tmp12 to <4 x float>
  %ret = shufflevector <4 x float> %tmp5, <4 x float> undef, <2 x i32> <i32 0, i32 1>
  ret <2 x float> %ret
}

define <2 x double> @test_fpext(float %f) {
; CHECK: @test_fpext
; CHECK: insertelement
; CHECK: insertelement
; CHECK-NOT: insertelement
  %tmp9 = insertelement <4 x float> undef, float %f, i32 0
  %tmp10 = insertelement <4 x float> %tmp9, float 0.000000e+00, i32 1
  %tmp11 = insertelement <4 x float> %tmp10, float 0.000000e+00, i32 2
  %tmp12 = insertelement <4 x float> %tmp11, float 0.000000e+00, i32 3
  %tmp5 = fpext <4 x float> %tmp12 to <4 x double>
  %ret = shufflevector <4 x double> %tmp5, <4 x double> undef, <2 x i32> <i32 0, i32 1>
  ret <2 x double> %ret
}

define <4 x float> @test_select(float %f, float %g) {
; CHECK: @test_select
; CHECK: %a0 = insertelement <4 x float> undef, float %f, i32 0
; CHECK-NOT: insertelement
; CHECK: %a3 = insertelement <4 x float> %a0, float 3.000000e+00, i32 3
; CHECK-NOT: insertelement
; CHECK: shufflevector <4 x float> %a3, <4 x float> <float undef, float 4.000000e+00, float 5.000000e+00, float undef>, <4 x i32> <i32 0, i32 5, i32 6, i32 3>
  %a0 = insertelement <4 x float> undef, float %f, i32 0
  %a1 = insertelement <4 x float> %a0, float 1.000000e+00, i32 1
  %a2 = insertelement <4 x float> %a1, float 2.000000e+00, i32 2
  %a3 = insertelement <4 x float> %a2, float 3.000000e+00, i32 3
  %b0 = insertelement <4 x float> undef, float %g, i32 0
  %b1 = insertelement <4 x float> %b0, float 4.000000e+00, i32 1
  %b2 = insertelement <4 x float> %b1, float 5.000000e+00, i32 2
  %b3 = insertelement <4 x float> %b2, float 6.000000e+00, i32 3
  %ret = select <4 x i1> <i1 true, i1 false, i1 false, i1 true>, <4 x float> %a3, <4 x float> %b3
  ret <4 x float> %ret
}


