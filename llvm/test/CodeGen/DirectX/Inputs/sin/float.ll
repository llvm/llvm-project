; Function Attrs: noinline nounwind optnone
define noundef float @sin_float(float noundef %a) #0 {
entry:
  %a.addr = alloca float, align 4
  store float %a, ptr %a.addr, align 4
  %0 = load float, ptr %a.addr, align 4
  %1 = call float @llvm.sin.f32(float %0)
  ret float %1
}
