; RUN: opt -S -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Make sure dxil operation function calls for firstbitlow are generated for all integer types.

define noundef i32 @test_firstbitlow_short(i16 noundef %a) {
entry:
; CHECK: call i32 @dx.op.unaryBits.i16(i32 32, i16 %{{.*}})
  %elt.firstbitlow = call i32 @llvm.dx.firstbitlow.i16(i16 %a)
  ret i32 %elt.firstbitlow
}

define noundef i32 @test_firstbitlow_int(i32 noundef %a) {
entry:
; CHECK: call i32 @dx.op.unaryBits.i32(i32 32, i32 %{{.*}})
  %elt.firstbitlow = call i32 @llvm.dx.firstbitlow.i32(i32 %a)
  ret i32 %elt.firstbitlow
}

define noundef i32 @test_firstbitlow_long(i64 noundef %a) {
entry:
; CHECK: call i32 @dx.op.unaryBits.i64(i32 32, i64 %{{.*}})
  %elt.firstbitlow = call i32 @llvm.dx.firstbitlow.i64(i64 %a)
  ret i32 %elt.firstbitlow
}

define noundef <4 x i32> @test_firstbitlow_vec4_i32(<4 x i32> noundef %a)  {
entry:
  ; CHECK: [[ee0:%.*]] = extractelement <4 x i32> %a, i64 0
  ; CHECK: [[ie0:%.*]] = call i32 @dx.op.unaryBits.i32(i32 32, i32 [[ee0]])
  ; CHECK: [[ee1:%.*]] = extractelement <4 x i32> %a, i64 1
  ; CHECK: [[ie1:%.*]] = call i32 @dx.op.unaryBits.i32(i32 32, i32 [[ee1]])
  ; CHECK: [[ee2:%.*]] = extractelement <4 x i32> %a, i64 2
  ; CHECK: [[ie2:%.*]] = call i32 @dx.op.unaryBits.i32(i32 32, i32 [[ee2]])
  ; CHECK: [[ee3:%.*]] = extractelement <4 x i32> %a, i64 3
  ; CHECK: [[ie3:%.*]] = call i32 @dx.op.unaryBits.i32(i32 32, i32 [[ee3]])
  ; CHECK: insertelement <4 x i32> poison, i32 [[ie0]], i64 0
  ; CHECK: insertelement <4 x i32> %{{.*}}, i32 [[ie1]], i64 1
  ; CHECK: insertelement <4 x i32> %{{.*}}, i32 [[ie2]], i64 2
  ; CHECK: insertelement <4 x i32> %{{.*}}, i32 [[ie3]], i64 3
  %2 = call <4 x i32> @llvm.dx.firstbitlow.v4i32(<4 x i32> %a)
  ret <4 x i32> %2
}

declare i32 @llvm.dx.firstbitlow.i16(i16)
declare i32 @llvm.dx.firstbitlow.i32(i32)
declare i32 @llvm.dx.firstbitlow.i64(i64)
declare <4 x i32> @llvm.dx.firstbitlow.v4i32(<4 x i32>)
