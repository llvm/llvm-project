; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

; DXIL operation deriv_coarse_x does not support double overload type
; CHECK: in function deriv_coarse_x
; CHECK-SAME: Cannot create DerivCoarseX operation: Invalid overload type

; Function Attrs: noinline nounwind optnone
define noundef double @deriv_coarse_x_double(double noundef %a) #0 {
entry:
  %a.addr = alloca double, align 8
  store double %a, ptr %a.addr, align 8
  %0 = load double, ptr %a.addr, align 8
  %dx.deriv_coarse_x = call double @llvm.dx.deriv.coarse.x.f64(double %0)
  ret double %dx.deriv_coarse_x
}
