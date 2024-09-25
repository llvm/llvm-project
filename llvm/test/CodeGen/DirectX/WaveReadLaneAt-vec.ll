; RUN: opt -S  -dxil-op-lower  -mtriple=dxil-pc-shadermodel6.3-compute %s | FileCheck %s

define noundef <4 x half> @wave_rla_halfv4(<4 x half> noundef %expr, i32 noundef %idx) #0 {
entry:
; CHECK: call <4 x half> @dx.op.waveReadLaneAt.v4f16(i32 117, <4 x half> %expr, i32 %idx)
  %ret = call <4 x half> @llvm.dx.wave.read.lane.at.v4f16(<4 x half> %expr, i32 %idx)
  ret <4 x half> %ret
}

define noundef <4 x float> @wave_rla_floatv4(<4 x float> noundef %expr, i32 noundef %idx) #0 {
entry:
; CHECK: call <4 x float> @dx.op.waveReadLaneAt.v4f32(i32 117, <4 x float> %expr, i32 %idx)
  %ret = call <4 x float> @llvm.dx.wave.read.lane.at.v4f32(<4 x float> %expr, i32 %idx)
  ret <4 x float> %ret
}

define noundef <4 x double> @wave_rla_doublev4(<4 x double> noundef %expr, i32 noundef %idx) #0 {
entry:
; CHECK: call <4 x double> @dx.op.waveReadLaneAt.v4f64(i32 117, <4 x double> %expr, i32 %idx)
  %ret = call <4 x double> @llvm.dx.wave.read.lane.at.v4f64(<4 x double> %expr, i32 %idx)
  ret <4 x double> %ret
}

define noundef <4 x i1> @wave_rla_v4i1(<4 x i1> noundef %expr, i32 noundef %idx) #0 {
entry:
; CHECK: call <4 x i1> @dx.op.waveReadLaneAt.v4i1(i32 117, <4 x i1> %expr, i32 %idx)
  %ret = call <4 x i1> @llvm.dx.wave.read.lane.at.v4i1(<4 x i1> %expr, i32 %idx)
  ret <4 x i1> %ret
}

define noundef <4 x i16> @wave_rla_v4i16(<4 x i16> noundef %expr, i32 noundef %idx) #0 {
entry:
; CHECK: call <4 x i16> @dx.op.waveReadLaneAt.v4i16(i32 117, <4 x i16> %expr, i32 %idx)
  %ret = call <4 x i16> @llvm.dx.wave.read.lane.at.v4i16(<4 x i16> %expr, i32 %idx)
  ret <4 x i16> %ret
}

define noundef <4 x i32> @wave_rla_v4i32(<4 x i32> noundef %expr, i32 noundef %idx) #0 {
entry:
; CHECK: call <4 x i32> @dx.op.waveReadLaneAt.v4i32(i32 117, <4 x i32> %expr, i32 %idx)
  %ret = call <4 x i32> @llvm.dx.wave.read.lane.at.v4i32(<4 x i32> %expr, i32 %idx)
  ret <4 x i32> %ret
}

declare <4 x half> @llvm.dx.wave.read.lane.at.v4f16(<4 x half>, i32) #1
declare <4 x float> @llvm.dx.wave.read.lane.at.v4f32(<4 x float>, i32) #1
declare <4 x double> @llvm.dx.wave.read.lane.at.v4f64(<4 x double>, i32) #1

declare <4 x i1> @llvm.dx.wave.read.lane.at.v4i1(<4 x i1>, i32) #1
declare <4 x i16> @llvm.dx.wave.read.lane.at.v4i16(<4 x i16>, i32) #1
declare <4 x i32> @llvm.dx.wave.read.lane.at.v4i32(<4 x i32>, i32) #1

attributes #0 = { convergent norecurse "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent nocallback nofree nosync nounwind willreturn }
