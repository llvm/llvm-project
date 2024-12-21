; RUN: opt -S -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Make sure dxil operation function calls for countbits are generated for all integer types.

define noundef i16 @test_countbits_short(i16 noundef %a) {
entry:
; CHECK: [[A:%.*]] = call i32 @dx.op.unaryBits.i16(i32 31, i16 %{{.*}})
; CHECK-NEXT: [[B:%.*]] = trunc i32 [[A]] to i16
; CHECK-NEXT ret i16 [[B]]
  %elt.ctpop = call i16 @llvm.ctpop.i16(i16 %a)
  ret i16 %elt.ctpop
}

define noundef i32 @test_countbits_short2(i16 noundef %a) {
entry:
; CHECK: [[A:%.*]] = call i32 @dx.op.unaryBits.i16(i32 31, i16 %{{.*}})
; CHECK-NEXT: ret i32 [[A]]
  %elt.ctpop = call i16 @llvm.ctpop.i16(i16 %a)
  %elt.zext = zext i16 %elt.ctpop to i32
  ret i32 %elt.zext
}

define noundef i32 @test_countbits_short3(i16 noundef %a) {
entry:
; CHECK: [[A:%.*]] = call i32 @dx.op.unaryBits.i16(i32 31, i16 %{{.*}})
; CHECK-NEXT: ret i32 [[A]]
  %elt.ctpop = call i16 @llvm.ctpop.i16(i16 %a)
  %elt.sext = sext i16 %elt.ctpop to i32
  ret i32 %elt.sext
}

define noundef i32 @test_countbits_int(i32 noundef %a) {
entry:
; CHECK: [[A:%.*]] = call i32 @dx.op.unaryBits.i32(i32 31, i32 %{{.*}})
; CHECK-NEXT: ret i32 [[A]]
  %elt.ctpop = call i32 @llvm.ctpop.i32(i32 %a)
  ret i32 %elt.ctpop
}

define noundef i64 @test_countbits_long(i64 noundef %a) {
entry:
; CHECK: [[A:%.*]] = call i32 @dx.op.unaryBits.i64(i32 31, i64 %{{.*}})
; CHECK-NEXT: [[B:%.*]] = zext i32 [[A]] to i64
; CHECK-NEXT ret i64 [[B]]
  %elt.ctpop = call i64 @llvm.ctpop.i64(i64 %a)
  ret i64 %elt.ctpop
}

define noundef i32 @test_countbits_long2(i64 noundef %a) {
entry:
; CHECK: [[A:%.*]] = call i32 @dx.op.unaryBits.i64(i32 31, i64 %{{.*}})
; CHECK-NEXT: ret i32 [[A]]
  %elt.ctpop = call i64 @llvm.ctpop.i64(i64 %a)
  %elt.trunc = trunc i64 %elt.ctpop to i32
  ret i32 %elt.trunc
}

define noundef <4 x i32> @countbits_vec4_i32(<4 x i32> noundef %a)  {
entry:
  ; CHECK: [[ee0:%.*]] = extractelement <4 x i32> %a, i64 0
  ; CHECK: [[ie0:%.*]] = call i32 @dx.op.unaryBits.i32(i32 31, i32 [[ee0]])
  ; CHECK: [[ee1:%.*]] = extractelement <4 x i32> %a, i64 1
  ; CHECK: [[ie1:%.*]] = call i32 @dx.op.unaryBits.i32(i32 31, i32 [[ee1]])
  ; CHECK: [[ee2:%.*]] = extractelement <4 x i32> %a, i64 2
  ; CHECK: [[ie2:%.*]] = call i32 @dx.op.unaryBits.i32(i32 31, i32 [[ee2]])
  ; CHECK: [[ee3:%.*]] = extractelement <4 x i32> %a, i64 3
  ; CHECK: [[ie3:%.*]] = call i32 @dx.op.unaryBits.i32(i32 31, i32 [[ee3]])
  ; CHECK: insertelement <4 x i32> poison, i32 [[ie0]], i64 0
  ; CHECK: insertelement <4 x i32> %{{.*}}, i32 [[ie1]], i64 1
  ; CHECK: insertelement <4 x i32> %{{.*}}, i32 [[ie2]], i64 2
  ; CHECK: insertelement <4 x i32> %{{.*}}, i32 [[ie3]], i64 3
  %2 = call <4 x i32> @llvm.ctpop.v4i32(<4 x i32> %a)
  ret <4 x i32> %2
}

declare i16 @llvm.ctpop.i16(i16)
declare i32 @llvm.ctpop.i32(i32)
declare i64 @llvm.ctpop.i64(i64)
declare <4 x i32> @llvm.ctpop.v4i32(<4 x i32>)
