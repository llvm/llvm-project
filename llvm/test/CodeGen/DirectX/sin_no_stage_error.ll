; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.0 %s 2>&1 | FileCheck %s

; Shader Stage is required to ensure the operation is supported.
; CHECK: LLVM ERROR: 1.0: Unknown Compilation Target Shader Stage specified

define noundef float @sin_float(float noundef %a) #0 {
entry:
  %a.addr = alloca float, align 4
  store float %a, ptr %a.addr, align 4
  %0 = load float, ptr %a.addr, align 4
  %1 = call float @llvm.sin.f32(float %0)
  ret float %1
}
