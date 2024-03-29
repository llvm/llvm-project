; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.0-library %s 2>&1 | FileCheck %s

; DXIL operation sin does not support half overload type in SM6.0
; CHECK: LLVM ERROR: Invalid Overload

; Function Attrs: noinline nounwind optnone
define noundef half @sin_half(half noundef %a) #0 {
entry:
  %a.addr = alloca half, align 2
  store half %a, ptr %a.addr, align 2
  %0 = load half, ptr %a.addr, align 2
  ; SM6_3: call half @dx.op.unary.f16(i32 13, half %{{.*}})
  %1 = call half @llvm.sin.f16(half %0)
  ret half %1
}
