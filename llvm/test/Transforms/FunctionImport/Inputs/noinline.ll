define void @foo(i64* %v) #0 {
entry:
  %v.addr = alloca i64*, align 8
  store i64* %v, i64** %v.addr, align 8
  ret void
}

attributes #0 = { noinline }