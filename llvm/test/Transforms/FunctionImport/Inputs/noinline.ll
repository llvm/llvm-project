define void @foo(ptr %v) #0 {
entry:
  %v.addr = alloca ptr, align 8
  store ptr %v, ptr %v.addr, align 8
  ret void
}

attributes #0 = { noinline }