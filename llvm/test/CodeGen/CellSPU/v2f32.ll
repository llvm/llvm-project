;RUN: llc --march=cellspu %s -o - | FileCheck %s
%vec = type <2 x float>

define %vec @test_ret(%vec %param)
{
;CHECK: bi $lr
 ret %vec %param
}

define %vec @test_add(%vec %param)
{
;CHECK: fa {{\$.}}, $3, $3
 %1 = fadd %vec %param, %param
;CHECK: bi $lr
 ret %vec %1
}

define %vec @test_sub(%vec %param)
{
;CHECK: fs {{\$.}}, $3, $3
 %1 = fsub %vec %param, %param

;CHECK: bi $lr
 ret %vec %1
}

define %vec @test_mul(%vec %param)
{
;CHECK: fm {{\$.}}, $3, $3
 %1 = fmul %vec %param, %param

;CHECK: bi $lr
 ret %vec %1
}

; CHECK: test_splat:
define %vec @test_splat(float %param ) {
;CHECK: lqa
;CHECK: shufb
  %sv = insertelement <1 x float> undef, float %param, i32 0 
  %rv = shufflevector <1 x float> %sv, <1 x float> undef, <2 x i32> zeroinitializer 
;CHECK: bi $lr
  ret %vec %rv
}

define void @test_store(%vec %val, %vec* %ptr){
; CHECK: test_store:
;CHECK: stqd 
  store %vec zeroinitializer, %vec* null

;CHECK: stqd $3, 0(${{.*}})
;CHECK: bi $lr
  store %vec %val, %vec* %ptr
  ret void
}

; CHECK: test_insert:
define %vec @test_insert(){
;CHECK: cwd
;CHECK: shufb $3
  %rv = insertelement %vec undef, float 0.0e+00, i32 undef
;CHECK: bi $lr
  ret %vec %rv
}

; CHECK: test_unaligned_store:

define void @test_unaligned_store()  {
;CHECK:	cdd
;CHECK:	shufb
;CHECK:	stqd
  %data = alloca [4 x float], align 16         ; <[4 x float]*> [#uses=1]
  %ptr = getelementptr [4 x float]* %data, i32 0, i32 2 ; <float*> [#uses=1]
  %vptr = bitcast float* %ptr to  <2 x float>* ; <[1 x <2 x float>]*> [#uses=1]
  store <2 x float> zeroinitializer, <2 x float>* %vptr
  ret void
}

