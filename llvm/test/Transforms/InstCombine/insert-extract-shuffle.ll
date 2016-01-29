; RUN: opt -S -instcombine %s | FileCheck %s

define <1 x i8> @test1(<8 x i8> %in) {
; CHECK-LABEL: @test1
; CHECK: shufflevector <8 x i8> %in, <8 x i8> undef, <1 x i32> <i32 5>
  %val = extractelement <8 x i8> %in, i32 5
  %vec = insertelement <1 x i8> undef, i8 %val, i32 0
  ret <1 x i8> %vec
}

define <4 x i16> @test2(<8 x i16> %in, <8 x i16> %in2) {
; CHECK-LABEL: @test2
; CHECK: shufflevector <8 x i16> %in2, <8 x i16> %in, <4 x i32> <i32 11, i32 9, i32 0, i32 10>
  %elt0 = extractelement <8 x i16> %in, i32 3
  %elt1 = extractelement <8 x i16> %in, i32 1
  %elt2 = extractelement <8 x i16> %in2, i32 0
  %elt3 = extractelement <8 x i16> %in, i32 2

  %vec.0 = insertelement <4 x i16> undef, i16 %elt0, i32 0
  %vec.1 = insertelement <4 x i16> %vec.0, i16 %elt1, i32 1
  %vec.2 = insertelement <4 x i16> %vec.1, i16 %elt2, i32 2
  %vec.3 = insertelement <4 x i16> %vec.2, i16 %elt3, i32 3

  ret <4 x i16> %vec.3
}

define <2 x i64> @test_vcopyq_lane_p64(<2 x i64> %a, <1 x i64> %b) {
; CHECK-LABEL: @test_vcopyq_lane_p64
; CHECK-NEXT: %[[WIDEVEC:.*]] = shufflevector <1 x i64> %b, <1 x i64> undef, <2 x i32> <i32 0, i32 undef>
; CHECK-NEXT: shufflevector <2 x i64> %a, <2 x i64> %[[WIDEVEC]], <2 x i32> <i32 0, i32 2>
; CHECK-NEXT: ret <2 x i64> %res
  %elt = extractelement <1 x i64> %b, i32 0
  %res = insertelement <2 x i64> %a, i64 %elt, i32 1
  ret <2 x i64> %res
}

; PR2109: https://llvm.org/bugs/show_bug.cgi?id=2109

define <4 x float> @widen_extract2(<4 x float> %ins, <2 x float> %ext) {
; CHECK-LABEL: @widen_extract2(
; CHECK-NEXT: %[[WIDEVEC:.*]] = shufflevector <2 x float> %ext, <2 x float> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
; CHECK-NEXT: shufflevector <4 x float> %ins, <4 x float> %[[WIDEVEC]], <4 x i32> <i32 0, i32 4, i32 2, i32 5>
; CHECK-NEXT: ret <4 x float> %i2
  %e1 = extractelement <2 x float> %ext, i32 0
  %e2 = extractelement <2 x float> %ext, i32 1
  %i1 = insertelement <4 x float> %ins, float %e1, i32 1
  %i2 = insertelement <4 x float> %i1, float %e2, i32 3
  ret <4 x float> %i2
}

define <4 x float> @widen_extract3(<4 x float> %ins, <3 x float> %ext) {
; CHECK-LABEL: @widen_extract3(
; CHECK-NEXT: %[[WIDEVEC:.*]] = shufflevector <3 x float> %ext, <3 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 undef>
; CHECK-NEXT: shufflevector <4 x float> %ins, <4 x float> %[[WIDEVEC]], <4 x i32> <i32 6, i32 5, i32 4, i32 3>
; CHECK-NEXT: ret <4 x float> %i3
  %e1 = extractelement <3 x float> %ext, i32 0
  %e2 = extractelement <3 x float> %ext, i32 1
  %e3 = extractelement <3 x float> %ext, i32 2
  %i1 = insertelement <4 x float> %ins, float %e1, i32 2
  %i2 = insertelement <4 x float> %i1, float %e2, i32 1
  %i3 = insertelement <4 x float> %i2, float %e3, i32 0
  ret <4 x float> %i3
}

define <8 x float> @widen_extract4(<8 x float> %ins, <2 x float> %ext) {
; CHECK-LABEL: @widen_extract4(
; CHECK-NEXT: %[[WIDEVEC:.*]] = shufflevector <2 x float> %ext, <2 x float> undef, <8 x i32> <i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
; CHECK-NEXT: shufflevector <8 x float> %ins, <8 x float> %[[WIDEVEC]], <8 x i32> <i32 0, i32 1, i32 8, i32 3, i32 4, i32 5, i32 6, i32 7>
; CHECK-NEXT: ret <8 x float> %i1
  %e1 = extractelement <2 x float> %ext, i32 0
  %i1 = insertelement <8 x float> %ins, float %e1, i32 2
  ret <8 x float> %i1
}

; PR26015: https://llvm.org/bugs/show_bug.cgi?id=26015
; The widening shuffle must be inserted before any uses.

define <8 x i16> @pr26015(<4 x i16> %t0) {
; CHECK-LABEL: @pr26015(
; CHECK-NEXT:  %[[WIDEVEC:.*]] = shufflevector <4 x i16> %t0, <4 x i16> undef, <8 x i32> <i32 undef, i32 undef, i32 undef, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
; CHECK-NEXT:  %[[EXT:.*]] = extractelement <4 x i16> %t0, i32 2
; CHECK-NEXT:  %t2 = insertelement <8 x i16> <i16 0, i16 0, i16 0, i16 undef, i16 0, i16 0, i16 undef, i16 undef>, i16 %[[EXT]], i32 3
; CHECK-NEXT:  %t3 = insertelement <8 x i16> %t2, i16 0, i32 6
; CHECK-NEXT:  %t5 = shufflevector <8 x i16> %t3, <8 x i16> %[[WIDEVEC]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 11>
; CHECK-NEXT:  ret <8 x i16> %t5
  %t1 = extractelement <4 x i16> %t0, i32 2
  %t2 = insertelement <8 x i16> zeroinitializer, i16 %t1, i32 3
  %t3 = insertelement <8 x i16> %t2, i16 0, i32 6
  %t4 = extractelement <4 x i16> %t0, i32 3
  %t5 = insertelement <8 x i16> %t3, i16 %t4, i32 7
  ret <8 x i16> %t5
}

; PR25999: https://llvm.org/bugs/show_bug.cgi?id=25999
; TODO: The widening shuffle could be inserted at the start of the function to allow the first extract to use it.

define <8 x i16> @pr25999(<4 x i16> %t0, i1 %b) {
; CHECK-LABEL: @pr25999(
; CHECK-NEXT:  %t1 = extractelement <4 x i16> %t0, i32 2
; CHECK-NEXT:  br i1 %b, label %if, label %end
; CHECK:       if:
; CHECK-NEXT:  %[[WIDEVEC:.*]] = shufflevector <4 x i16> %t0, <4 x i16> undef, <8 x i32> <i32 undef, i32 undef, i32 undef, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
; CHECK-NEXT:  %t2 = insertelement <8 x i16> <i16 0, i16 0, i16 0, i16 undef, i16 0, i16 0, i16 undef, i16 undef>, i16 %t1, i32 3
; CHECK-NEXT:  %t3 = insertelement <8 x i16> %t2, i16 0, i32 6
; CHECK-NEXT:  %t5 = shufflevector <8 x i16> %t3, <8 x i16> %[[WIDEVEC]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 11>
; CHECK-NEXT:  ret <8 x i16> %t5
; CHECK:       end:
; CHECK-NEXT:  %a1 = add i16 %t1, 4
; CHECK-NEXT:  %t6 = insertelement <8 x i16> <i16 undef, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, i16 %a1, i32 0
; CHECK-NEXT:  ret <8 x i16> %t6

  %t1 = extractelement <4 x i16> %t0, i32 2
  br i1 %b, label %if, label %end

if:
  %t2 = insertelement <8 x i16> zeroinitializer, i16 %t1, i32 3
  %t3 = insertelement <8 x i16> %t2, i16 0, i32 6
  %t4 = extractelement <4 x i16> %t0, i32 3
  %t5 = insertelement <8 x i16> %t3, i16 %t4, i32 7
  ret <8 x i16> %t5

end:
  %a1 = add i16 %t1, 4
  %t6 = insertelement <8 x i16> zeroinitializer, i16 %a1, i32 0
  ret <8 x i16> %t6
}

; The widening shuffle must be inserted at a valid point (after the PHIs). 

define <4 x double> @pr25999_phis1(i1 %c, <2 x double> %a, <4 x double> %b) {
; CHECK-LABEL: @pr25999_phis1(
; CHECK:       %tmp1 = phi <2 x double> [ %a, %bb1 ], [ %r, %bb2 ]
; CHECK-NEXT:  %tmp2 = phi <4 x double> [ %b, %bb1 ], [ zeroinitializer, %bb2 ]
; CHECK-NEXT:  %[[WIDEVEC:.*]] = shufflevector <2 x double> %tmp1, <2 x double> undef, <4 x i32> <i32 0, i32 undef, i32 undef, i32 undef>
; CHECK-NEXT:  %tmp4 = shufflevector <4 x double> %tmp2, <4 x double> %[[WIDEVEC]], <4 x i32> <i32 0, i32 1, i32 4, i32 3>
; CHECK-NEXT:  ret <4 x double> %tmp4
bb1:
  br i1 %c, label %bb2, label %bb3

bb2:
  %r = call <2 x double> @dummy(<2 x double> %a)
  br label %bb3

bb3:
  %tmp1 = phi <2 x double> [ %a, %bb1 ], [ %r, %bb2 ]
  %tmp2 = phi <4 x double> [ %b, %bb1 ], [ zeroinitializer, %bb2 ]
  %tmp3 = extractelement <2 x double> %tmp1, i32 0
  %tmp4 = insertelement <4 x double> %tmp2, double %tmp3, i32 2
  ret <4 x double> %tmp4
}

declare <2 x double> @dummy(<2 x double>)

define <4 x double> @pr25999_phis2(i1 %c, <2 x double> %a, <4 x double> %b) {
; CHECK-LABEL: @pr25999_phis2(
; CHECK:       %tmp1 = phi <2 x double> [ %a, %bb1 ], [ %r, %bb2 ]
; CHECK-NEXT:  %tmp2 = phi <4 x double> [ %b, %bb1 ], [ zeroinitializer, %bb2 ]
; CHECK-NEXT:  %d = fadd <2 x double> %tmp1, %tmp1
; CHECK-NEXT:  %[[WIDEVEC:.*]] = shufflevector <2 x double> %d, <2 x double> undef, <4 x i32> <i32 0, i32 undef, i32 undef, i32 undef>
; CHECK-NEXT:  %tmp4 = shufflevector <4 x double> %tmp2, <4 x double> %[[WIDEVEC]], <4 x i32> <i32 0, i32 1, i32 4, i32 3>
; CHECK-NEXT:  ret <4 x double> %tmp4
bb1:
  br i1 %c, label %bb2, label %bb3

bb2:
  %r = call <2 x double> @dummy(<2 x double> %a)
  br label %bb3

bb3:
  %tmp1 = phi <2 x double> [ %a, %bb1 ], [ %r, %bb2 ]
  %tmp2 = phi <4 x double> [ %b, %bb1 ], [ zeroinitializer, %bb2 ]
  %d = fadd <2 x double> %tmp1, %tmp1
  %tmp3 = extractelement <2 x double> %d, i32 0
  %tmp4 = insertelement <4 x double> %tmp2, double %tmp3, i32 2
  ret <4 x double> %tmp4
}

; PR26354: https://llvm.org/bugs/show_bug.cgi?id=26354
; Don't create a shufflevector if we know that we're not going to replace the insertelement.

define double @pr26354(<2 x double>* %tmp, i1 %B) {
; CHECK-LABEL: @pr26354(
; CHECK:       %ld = load <2 x double>, <2 x double>* %tmp
; CHECK-NEXT:  %e1 = extractelement <2 x double> %ld, i32 0
; CHECK-NEXT:  br i1 %B, label %if, label %end
; CHECK:       if:
; CHECK-NEXT:  %e2 = extractelement <2 x double> %ld, i32 1
; CHECK-NEXT:  %i1 = insertelement <4 x double>
; CHECK-NEXT:  br label %end

entry:
  %ld = load <2 x double>, <2 x double>* %tmp
  %e1 = extractelement <2 x double> %ld, i32 0
  %e2 = extractelement <2 x double> %ld, i32 1
  br i1 %B, label %if, label %end

if:
  %i1 = insertelement <4 x double> zeroinitializer, double %e2, i32 3
  br label %end

end:
  %ph = phi <4 x double> [ undef, %entry ], [ %i1, %if ]
  %e3 = extractelement <4 x double> %ph, i32 1
  %mu = fmul double %e1, %e3
  ret double %mu
}

