; Function Attrs: noinline nounwind optnone
define noundef half @sin_half(half noundef %a) #0 {
entry:
  %a.addr = alloca half, align 2
  store half %a, ptr %a.addr, align 2
  %0 = load half, ptr %a.addr, align 2
  %1 = call half @llvm.sin.f16(half %0)
  ret half %1
}
