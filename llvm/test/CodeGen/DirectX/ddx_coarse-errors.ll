; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck --check-prefixes=CHECK-TYPE %s
; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-compute %s 2>&1 | FileCheck --check-prefixes=CHECK-STAGE %s

; DXIL operation ddx.coarse does not support double overload type
; CHECK-TYPE: in function ddx.coarse
; CHECK-TYPE-SAME: Cannot create DerivCoarseX operation: Invalid overload type

; Function Attrs: noinline nounwind optnone
define noundef double @ddx.coarse_double(double noundef %a) #0 {
entry:
  %a.addr = alloca double, align 8
  store double %a, ptr %a.addr, align 8
  %0 = load double, ptr %a.addr, align 8
  %dx.ddx.coarse = call double @llvm.dx.ddx.coarse.f64(double %0)
  ret double %dx.ddx.coarse
}

; DXIL operation ddx.coarse does not support compute shader stage
; CHECK-STAGE: in function ddx.coarse
; CHECK-STAGE-SAME: Cannot create DerivCoarseX operation: Invalid stage
; Function Attrs: noinline nounwind optnone
define noundef float @ddx.coarse_float(float noundef %a) #0 {
entry:
  %a.addr = alloca float, align 8
  store float %a, ptr %a.addr, align 8
  %0 = load float, ptr %a.addr, align 8
  %dx.ddx.coarse = call float @llvm.dx.ddx.coarse.f32(float %0)
  ret float %dx.ddx.coarse
}
