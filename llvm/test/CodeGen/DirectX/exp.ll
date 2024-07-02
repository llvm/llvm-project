; RUN: opt -S -dxil-op-lower < %s | FileCheck %s

; Make sure dxil operation function calls for exp are generated for float and half.

; CHECK-LABEL: exp_float
; CHECK: fmul float 0x3FF7154760000000, %{{.*}}
; CHECK: call float @dx.op.unary.f32(i32 21, float %{{.*}})
define noundef float @exp_float(float noundef %a) {
entry:
  %a.addr = alloca float, align 4
  store float %a, ptr %a.addr, align 4
  %0 = load float, ptr %a.addr, align 4
  %elt.exp = call float @llvm.exp.f32(float %0)
  ret float %elt.exp
}

; CHECK-LABEL: exp_half
; CHECK: fmul half 0xH3DC5, %{{.*}}
; CHECK: call half @dx.op.unary.f16(i32 21, half %{{.*}})
; Function Attrs: noinline nounwind optnone
define noundef half @exp_half(half noundef %a) {
entry:
  %a.addr = alloca half, align 2
  store half %a, ptr %a.addr, align 2
  %0 = load half, ptr %a.addr, align 2
  %elt.exp = call half @llvm.exp.f16(half %0)
  ret half %elt.exp
}

declare half @llvm.exp.f16(half)
declare float @llvm.exp.f32(float)
