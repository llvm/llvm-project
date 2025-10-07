; RUN: opt -S -dxil-op-lower %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-compute"

declare void @f32_user(float)
declare void @f64_user(double)
declare void @f16_user(half)

; CHECK-LABEL: define void @loadf32
define void @loadf32() {
  %buffer = call target("dx.CBuffer", target("dx.Layout", {float}, 4, 0))
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: [[DATA:%.*]] = call %dx.types.CBufRet.f32 @dx.op.cbufferLoadLegacy.f32(i32 59, %dx.types.Handle %{{.*}}, i32 0)
  %load = call {float, float, float, float} @llvm.dx.resource.load.cbufferrow.4(
      target("dx.CBuffer", target("dx.Layout", {float}, 4, 0)) %buffer,
      i32 0)
  %data = extractvalue {float, float, float, float} %load, 0

  ; CHECK: [[VAL:%.*]] = extractvalue %dx.types.CBufRet.f32 [[DATA]], 0
  ; CHECK: call void @f32_user(float [[VAL]])
  call void @f32_user(float %data)

  ret void
}

; CHECK-LABEL: define void @loadf64
define void @loadf64() {
  %buffer = call
      target("dx.CBuffer", target("dx.Layout", {double, double, double, double}, 64, 0, 8, 16, 24))
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: [[DATA:%.*]] = call %dx.types.CBufRet.f64 @dx.op.cbufferLoadLegacy.f64(i32 59, %dx.types.Handle %{{.*}}, i32 1)
  %load = call {double, double} @llvm.dx.resource.load.cbufferrow.2(
      target("dx.CBuffer", target("dx.Layout", {double, double, double, double}, 64, 0, 8, 16, 24)) %buffer,
      i32 1)
  %data = extractvalue {double, double} %load, 1

  ; CHECK: [[VAL:%.*]] = extractvalue %dx.types.CBufRet.f64 [[DATA]], 1
  ; CHECK: call void @f64_user(double [[VAL]])
  call void @f64_user(double %data)

  ret void
}

; CHECK-LABEL: define void @loadf16
define void @loadf16() {
  %buffer = call
      target("dx.CBuffer", target("dx.Layout", {half}, 2, 0))
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: [[DATA:%.*]] = call %dx.types.CBufRet.f16.8 @dx.op.cbufferLoadLegacy.f16(i32 59, %dx.types.Handle %{{.*}}, i32 0)
  %load = call {half, half, half, half, half, half, half, half} @llvm.dx.resource.load.cbufferrow.8(
      target("dx.CBuffer", target("dx.Layout", {half}, 2, 0)) %buffer,
      i32 0)
  %data = extractvalue {half, half, half, half, half, half, half, half} %load, 0

  ; CHECK: [[VAL:%.*]] = extractvalue %dx.types.CBufRet.f16.8 [[DATA]], 0
  ; CHECK: call void @f16_user(half [[VAL]])
  call void @f16_user(half %data)

  ret void
}
