; RUN: opt -S -dxil-op-lower < %s | FileCheck %s

; Make sure dxil operation function calls for isinf are generated for float and half.
; CHECK: call i1 @dx.op.isSpecialFloat.f32(i32 9, float %{{.*}})
; CHECK: call i1 @dx.op.isSpecialFloat.f16(i32 9, half %{{.*}})

; Function Attrs: noinline nounwind optnone
define noundef i1 @isinf_float(float noundef %a) #0 {
entry:
  %a.addr = alloca float, align 4
  store float %a, ptr %a.addr, align 4
  %0 = load float, ptr %a.addr, align 4
  %dx.isinf = call i1 @llvm.dx.isinf.f32(float %0)
  ret i1 %dx.isinf
}

; Function Attrs: noinline nounwind optnone
define noundef i1 @isinf_half(half noundef %p0) #0 {
entry:
  %p0.addr = alloca half, align 2
  store half %p0, ptr %p0.addr, align 2
  %0 = load half, ptr %p0.addr, align 2
  %dx.isinf = call i1 @llvm.dx.isinf.f16(half %0)
  ret i1 %dx.isinf
}
