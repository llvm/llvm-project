; Test that vectors are scalarized/lowered correctly.
; RUN: llc < %s

%d8 = type <8 x double>
%f1 = type <1 x float>
%f2 = type <2 x float>
%f4 = type <4 x float>
%f8 = type <8 x float>
%i4 = type <4 x i32>

;;; TEST HANDLING OF VARIOUS VECTOR SIZES

define void @test_f1(%f1* %P, %f1* %Q, %f1* %S) {
	%p = load %f1, %f1* %P		; <%f1> [#uses=1]
	%q = load %f1, %f1* %Q		; <%f1> [#uses=1]
	%R = fadd %f1 %p, %q		; <%f1> [#uses=1]
	store %f1 %R, %f1* %S
	ret void
}

define void @test_f2(%f2* %P, %f2* %Q, %f2* %S) {
	%p = load %f2, %f2* %P		; <%f2> [#uses=1]
	%q = load %f2, %f2* %Q		; <%f2> [#uses=1]
	%R = fadd %f2 %p, %q		; <%f2> [#uses=1]
	store %f2 %R, %f2* %S
	ret void
}

define void @test_f4(%f4* %P, %f4* %Q, %f4* %S) {
	%p = load %f4, %f4* %P		; <%f4> [#uses=1]
	%q = load %f4, %f4* %Q		; <%f4> [#uses=1]
	%R = fadd %f4 %p, %q		; <%f4> [#uses=1]
	store %f4 %R, %f4* %S
	ret void
}

define void @test_f8(%f8* %P, %f8* %Q, %f8* %S) {
	%p = load %f8, %f8* %P		; <%f8> [#uses=1]
	%q = load %f8, %f8* %Q		; <%f8> [#uses=1]
	%R = fadd %f8 %p, %q		; <%f8> [#uses=1]
	store %f8 %R, %f8* %S
	ret void
}

define void @test_fmul(%f8* %P, %f8* %Q, %f8* %S) {
	%p = load %f8, %f8* %P		; <%f8> [#uses=1]
	%q = load %f8, %f8* %Q		; <%f8> [#uses=1]
	%R = fmul %f8 %p, %q		; <%f8> [#uses=1]
	store %f8 %R, %f8* %S
	ret void
}

define void @test_div(%f8* %P, %f8* %Q, %f8* %S) {
	%p = load %f8, %f8* %P		; <%f8> [#uses=1]
	%q = load %f8, %f8* %Q		; <%f8> [#uses=1]
	%R = fdiv %f8 %p, %q		; <%f8> [#uses=1]
	store %f8 %R, %f8* %S
	ret void
}

;;; TEST VECTOR CONSTRUCTS


define void @test_cst(%f4* %P, %f4* %S) {
	%p = load %f4, %f4* %P		; <%f4> [#uses=1]
	%R = fadd %f4 %p, < float 0x3FB99999A0000000, float 1.000000e+00, float 2.000000e+00, float 4.500000e+00 >		; <%f4> [#uses=1]
	store %f4 %R, %f4* %S
	ret void
}

define void @test_zero(%f4* %P, %f4* %S) {
	%p = load %f4, %f4* %P		; <%f4> [#uses=1]
	%R = fadd %f4 %p, zeroinitializer		; <%f4> [#uses=1]
	store %f4 %R, %f4* %S
	ret void
}

define void @test_undef(%f4* %P, %f4* %S) {
	%p = load %f4, %f4* %P		; <%f4> [#uses=1]
	%R = fadd %f4 %p, undef		; <%f4> [#uses=1]
	store %f4 %R, %f4* %S
	ret void
}

define void @test_constant_insert(%f4* %S) {
	%R = insertelement %f4 zeroinitializer, float 1.000000e+01, i32 0		; <%f4> [#uses=1]
	store %f4 %R, %f4* %S
	ret void
}

define void @test_variable_buildvector(float %F, %f4* %S) {
	%R = insertelement %f4 zeroinitializer, float %F, i32 0		; <%f4> [#uses=1]
	store %f4 %R, %f4* %S
	ret void
}

define void @test_scalar_to_vector(float %F, %f4* %S) {
	%R = insertelement %f4 undef, float %F, i32 0		; <%f4> [#uses=1]
	store %f4 %R, %f4* %S
	ret void
}

define float @test_extract_elt(%f8* %P) {
	%p = load %f8, %f8* %P		; <%f8> [#uses=1]
	%R = extractelement %f8 %p, i32 3		; <float> [#uses=1]
	ret float %R
}

define double @test_extract_elt2(%d8* %P) {
	%p = load %d8, %d8* %P		; <%d8> [#uses=1]
	%R = extractelement %d8 %p, i32 3		; <double> [#uses=1]
	ret double %R
}

define void @test_cast_1(%f4* %b, %i4* %a) {
	%tmp = load %f4, %f4* %b		; <%f4> [#uses=1]
	%tmp2 = fadd %f4 %tmp, < float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00 >		; <%f4> [#uses=1]
	%tmp3 = bitcast %f4 %tmp2 to %i4		; <%i4> [#uses=1]
	%tmp4 = add %i4 %tmp3, < i32 1, i32 2, i32 3, i32 4 >		; <%i4> [#uses=1]
	store %i4 %tmp4, %i4* %a
	ret void
}

define void @test_cast_2(%f8* %a, <8 x i32>* %b) {
	%T = load %f8, %f8* %a		; <%f8> [#uses=1]
	%T2 = bitcast %f8 %T to <8 x i32>		; <<8 x i32>> [#uses=1]
	store <8 x i32> %T2, <8 x i32>* %b
	ret void
}

;;; TEST IMPORTANT IDIOMS

define void @splat(%f4* %P, %f4* %Q, float %X) {
	%tmp = insertelement %f4 undef, float %X, i32 0		; <%f4> [#uses=1]
	%tmp2 = insertelement %f4 %tmp, float %X, i32 1		; <%f4> [#uses=1]
	%tmp4 = insertelement %f4 %tmp2, float %X, i32 2		; <%f4> [#uses=1]
	%tmp6 = insertelement %f4 %tmp4, float %X, i32 3		; <%f4> [#uses=1]
	%q = load %f4, %f4* %Q		; <%f4> [#uses=1]
	%R = fadd %f4 %q, %tmp6		; <%f4> [#uses=1]
	store %f4 %R, %f4* %P
	ret void
}

define void @splat_i4(%i4* %P, %i4* %Q, i32 %X) {
	%tmp = insertelement %i4 undef, i32 %X, i32 0		; <%i4> [#uses=1]
	%tmp2 = insertelement %i4 %tmp, i32 %X, i32 1		; <%i4> [#uses=1]
	%tmp4 = insertelement %i4 %tmp2, i32 %X, i32 2		; <%i4> [#uses=1]
	%tmp6 = insertelement %i4 %tmp4, i32 %X, i32 3		; <%i4> [#uses=1]
	%q = load %i4, %i4* %Q		; <%i4> [#uses=1]
	%R = add %i4 %q, %tmp6		; <%i4> [#uses=1]
	store %i4 %R, %i4* %P
	ret void
}

define <2 x i32*> @vector_gep(<2 x [3 x {i32, i32}]*> %a) {
    %w = getelementptr [3 x {i32, i32}], <2 x [3 x {i32, i32}]*> %a, <2 x i32> <i32 1, i32 2>, <2 x i32> <i32 2, i32 3>, <2 x i32> <i32 1, i32 1>
      ret <2 x i32*> %w
}

define i32 @extractelt_constant_bitcast() {
  %1 = bitcast i64 4 to <2 x i32>
  %2 = extractelement <2 x i32> %1, i32 0
  ret i32 %2
}
