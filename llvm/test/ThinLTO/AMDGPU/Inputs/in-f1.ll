define void @f1(ptr %v) #0 {
entry:
  call void @weak_common(ptr %v)
  ret void
}

define weak hidden void @weak_common(ptr %v) #0 {
entry:
  store i32 12345, ptr %v
  ret void
}

attributes #0 = { noinline }
