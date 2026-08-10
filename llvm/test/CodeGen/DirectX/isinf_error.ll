; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

; DXIL operation isinf does not support double overload type
; CHECK: in function isinf_double
; CHECK-SAME: Cannot create IsInf operation: Invalid overload type

define noundef i1 @isinf_double(double noundef %a) #0 {
entry:
  %a.addr = alloca double, align 8
  store double %a, ptr %a.addr, align 8
  %0 = load double, ptr %a.addr, align 8
  %dx.isinf = call i1 @llvm.dx.isinf.f64(double %0)
  ret i1 %dx.isinf
}
