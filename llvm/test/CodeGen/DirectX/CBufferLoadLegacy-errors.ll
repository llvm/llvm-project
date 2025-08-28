; We use llc for this test so that we don't abort after the first error.
; RUN: not llc %s -o /dev/null 2>&1 | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-compute"

declare void @f32_user(float)
declare void @f64_user(double)
declare void @f16_user(half)

; CHECK: error:
; CHECK-SAME: in function four64
; CHECK-SAME: Type mismatch between intrinsic and DXIL op
define void @four64() "hlsl.export" {
  %buffer = call target("dx.CBuffer", target("dx.Layout", {double}, 8, 0))
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)

  %load = call {double, double, double, double} @llvm.dx.resource.load.cbufferrow.4(
      target("dx.CBuffer", target("dx.Layout", {double}, 8, 0)) %buffer,
      i32 0)
  %data = extractvalue {double, double, double, double} %load, 0

  call void @f64_user(double %data)

  ret void
}

; CHECK: error:
; CHECK-SAME: in function two32
; CHECK-SAME: Type mismatch between intrinsic and DXIL op
define void @two32() "hlsl.export" {
  %buffer = call target("dx.CBuffer", target("dx.Layout", {float}, 4, 0))
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)

  %load = call {float, float} @llvm.dx.resource.load.cbufferrow.2(
      target("dx.CBuffer", target("dx.Layout", {float}, 4, 0)) %buffer,
      i32 0)
  %data = extractvalue {float, float} %load, 0

  call void @f32_user(float %data)

  ret void
}

declare { double, double, double, double } @llvm.dx.resource.load.cbufferrow.4.f64.f64.f64.f64.tdx.CBuffer_tdx.Layout_sl_f64s_8_0tt(target("dx.CBuffer", target("dx.Layout", { double }, 8, 0)), i32)
declare { float, float } @llvm.dx.resource.load.cbufferrow.2.f32.f32.tdx.CBuffer_tdx.Layout_sl_f32s_4_0tt(target("dx.CBuffer", target("dx.Layout", { float }, 4, 0)), i32)
