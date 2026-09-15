; RUN: opt -S -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-compute %s | FileCheck %s

define noundef i32 @wave_bitor_simple(i32 noundef %p1) {
entry:
; CHECK: call i32 @dx.op.waveActiveBit.i32(i32 120, i32 %p1, i8 1){{$}}
  %ret = call i32 @llvm.dx.wave.reduce.or.i32(i32 %p1)
  ret i32 %ret
}

declare i32 @llvm.dx.wave.reduce.or.i32(i32)

define noundef i64 @wave_bitor_simple64(i64 noundef %p1) {
entry:
; CHECK: call i64 @dx.op.waveActiveBit.i64(i32 120, i64 %p1, i8 1){{$}}
  %ret = call i64 @llvm.dx.wave.reduce.or.i64(i64 %p1)
  ret i64 %ret
}

declare i64 @llvm.dx.wave.reduce.or.i64(i64)
