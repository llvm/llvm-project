define void @take(ptr %p) noinline {
  ret void
}

define void @large() {
  %a = alloca [32776 x i8], align 8
  call void @take(ptr %a)
  ret void
}
