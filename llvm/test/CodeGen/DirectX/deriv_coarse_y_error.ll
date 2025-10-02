; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

; DXIL operation deriv_coarse_y does not support double overload type
; CHECK: in function deriv_coarse_y
; CHECK-SAME: Cannot create DerivCoarseY operation: Invalid overload type

; Function Attrs: noinline nounwind optnone
define noundef double @deriv_coarse_y_double(double noundef %a) #0 {
entry:
  %a.addr = alloca double, align 8
  store double %a, ptr %a.addr, align 8
  %0 = load double, ptr %a.addr, align 8
  %dx.deriv_coarse_y = call double @llvm.dx.deriv.coarse.y.f64(double %0)
  ret double %dx.deriv_coarse_y
}
