; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.0-library %s 2>&1 | FileCheck %s --check-prefix=SM6_0_DOUBLE
; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s --check-prefix=SM6_3_DOUBLE

; Double is not valid in any Shader Model version
; SM6_0_DOUBLE: LLVM ERROR: Invalid Overload
; SM6_3_DOUBLE: LLVM ERROR: Invalid Overload

define noundef double @sin_double(double noundef %a) #0 {
entry:
  %a.addr = alloca double, align 8
  store double %a, ptr %a.addr, align 8
  %0 = load double, ptr %a.addr, align 8
  %1 = call double @llvm.sin.f64(double %0)
  ret double %1
}

