; RUN: lli -O0 -force-interpreter < %s

; libffi does not support fp128 so we don’t test it
declare float  @llvm.sin.f32(float)
declare double @llvm.sin.f64(double)
declare float  @llvm.cos.f32(float)
declare double @llvm.cos.f64(double)
declare float  @llvm.floor.f32(float)
declare double @llvm.floor.f64(double)
declare float  @llvm.ceil.f32(float)
declare double @llvm.ceil.f64(double)
declare float  @llvm.trunc.f32(float)
declare double @llvm.trunc.f64(double)
declare float  @llvm.round.f32(float)
declare double @llvm.round.f64(double)
declare float  @llvm.roundeven.f32(float)
declare double @llvm.roundeven.f64(double)
declare float  @llvm.copysign.f32(float, float)
declare double @llvm.copysign.f64(double, double)

define i32 @main() {
  %sin32 = call float @llvm.sin.f32(float 0.000000e+00)
  %sin64 = call double @llvm.sin.f64(double 0.000000e+00)
  %cos32 = call float @llvm.cos.f32(float 0.000000e+00)
  %cos64 = call double @llvm.cos.f64(double 0.000000e+00)
  %floor32 = call float @llvm.floor.f32(float 0.000000e+00)
  %floor64 = call double @llvm.floor.f64(double 0.000000e+00)
  %ceil32 = call float @llvm.ceil.f32(float 0.000000e+00)
  %ceil64 = call double @llvm.ceil.f64(double 0.000000e+00)
  %trunc32 = call float @llvm.trunc.f32(float 0.000000e+00)
  %trunc64 = call double @llvm.trunc.f64(double 0.000000e+00)
  %round32 = call float @llvm.round.f32(float 0.000000e+00)
  %round64 = call double @llvm.round.f64(double 0.000000e+00)
  %roundeven32 = call float @llvm.roundeven.f32(float 0.000000e+00)
  %roundeven64 = call double @llvm.roundeven.f64(double 0.000000e+00)
  %copysign32 = call float @llvm.copysign.f32(float 0.000000e+00, float 0.000000e+00)
  %copysign64 = call double @llvm.copysign.f64(double 0.000000e+00, double 0.000000e+00)
  ret i32 0
}
