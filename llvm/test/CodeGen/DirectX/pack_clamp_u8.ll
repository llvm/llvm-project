; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.6-library %s | FileCheck %s

define noundef i32 @pack_clamp_u8_16(<4 x i16> noundef %a) {
; CHECK: [[one:%.*]] = extractelement <4 x i16> %a, i64 0
; CHECK: [[two:%.*]] = extractelement <4 x i16> %a, i64 1
; CHECK: [[three:%.*]] = extractelement <4 x i16> %a, i64 2
; CHECK: [[four:%.*]] = extractelement <4 x i16> %a, i64 3
; CHECK: [[packed:%.*]] = call i32 @dx.op.pack4x8.i16(i32 220, i8 1, i16 [[one]], i16 [[two]], i16 [[three]], i16 [[four]])
; ret i32 [[packed]]
  %packed = call i32 @llvm.dx.pack.clamp.u8.v4i16(<4 x i16> %a)
  ret i32 %packed
}

define noundef i32 @pack_clamp_u8_32(<4 x i32> noundef %a) {
; CHECK: [[one:%.*]] = extractelement <4 x i32> %a, i64 0
; CHECK: [[two:%.*]] = extractelement <4 x i32> %a, i64 1
; CHECK: [[three:%.*]] = extractelement <4 x i32> %a, i64 2
; CHECK: [[four:%.*]] = extractelement <4 x i32> %a, i64 3
; CHECK: [[packed:%.*]] = call i32 @dx.op.pack4x8.i32(i32 220, i8 1, i32 [[one]], i32 [[two]], i32 [[three]], i32 [[four]])
; ret i32 [[packed]]
  %packed = call i32 @llvm.dx.pack.clamp.u8.v4i32(<4 x i32> %a)
  ret i32 %packed
}

declare i32 @llvm.dx.pack.clamp.u8.v4i16(<4 x i16>)
declare i32 @llvm.dx.pack.clamp.u8.v4i32(<4 x i32>)
