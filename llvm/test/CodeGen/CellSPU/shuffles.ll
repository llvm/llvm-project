; RUN: llc -O1  --march=cellspu < %s | FileCheck %s

;CHECK: shuffle
define <4 x float> @shuffle(<4 x float> %param1, <4 x float> %param2) {
  ; CHECK: cwd {{\$.}}, 0($sp)
  ; CHECK: shufb {{\$., \$4, \$3, \$.}}
  %val= shufflevector <4 x float> %param1, <4 x float> %param2, <4 x i32> <i32 4,i32 1,i32 2,i32 3>
  ret <4 x float> %val
}

;CHECK: splat
define <4 x float> @splat(float %param1) {
  ; CHECK: lqa
  ; CHECK: shufb $3
  ; CHECK: bi
  %vec = insertelement <1 x float> undef, float %param1, i32 0
  %val= shufflevector <1 x float> %vec, <1 x float> undef, <4 x i32> <i32 0,i32 0,i32 0,i32 0>
  ret <4 x float> %val  
}

;CHECK: test_insert
define void @test_insert( <2 x float>* %ptr, float %val1, float %val2 ) {
  %sl2_17_tmp1 = insertelement <2 x float> zeroinitializer, float %val1, i32 0
;CHECK:	lqa	$6,
;CHECK:	shufb	$4, $4, $5, $6
  %sl2_17 = insertelement <2 x float> %sl2_17_tmp1, float %val2, i32 1

;CHECK: cdd	$5, 0($3)
;CHECK: lqd	$6, 0($3)
;CHECK: shufb	$4, $4, $6, $5
;CHECK: stqd	$4, 0($3)
;CHECK:	bi	$lr
  store <2 x float> %sl2_17, <2 x float>* %ptr
  ret void 
}

;CHECK: test_insert_1
define <4 x float>  @test_insert_1(<4 x float> %vparam, float %eltparam) {
;CHECK: cwd     $5, 4($sp)
;CHECK: shufb   $3, $4, $3, $5
;CHECK: bi      $lr
  %rv = insertelement <4 x float> %vparam, float %eltparam, i32 1
  ret <4 x float> %rv
}

;CHECK: test_v2i32
define <2 x i32> @test_v2i32(<4 x i32>%vec)
{
;CHECK: rotqbyi $3, $3, 4
;CHECK: bi $lr
  %rv = shufflevector <4 x i32> %vec, <4 x i32> undef, <2 x i32><i32 1,i32 2>
  ret <2 x i32> %rv
}

define <4 x i32> @test_v4i32_rot8(<4 x i32>%vec)
{
  %rv = shufflevector <4 x i32> %vec, <4 x i32> undef, 
        <4 x i32> <i32 2,i32 3,i32 0, i32 1>
  ret <4 x i32> %rv
}

;CHECK: test_v4i32_rot4
define <4 x i32> @test_v4i32_rot4(<4 x i32>%vec)
{
  %rv = shufflevector <4 x i32> %vec, <4 x i32> undef, 
        <4 x i32> <i32 1,i32 2,i32 3, i32 0>
  ret <4 x i32> %rv
}

