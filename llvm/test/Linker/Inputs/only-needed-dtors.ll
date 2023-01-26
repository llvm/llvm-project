define internal void @dtor1() {
  call void @func1()
  ret void
}

define internal void @dtor2() {
  ret void
}

define void @func1() {
  ret void
}

define void @unused() {
  ret void
}

@llvm.global_dtors = appending global[2 x{i32, ptr, ptr }] [
    {i32, ptr, ptr } { i32 2, ptr @dtor1, ptr null},
    {i32, ptr, ptr } { i32 7, ptr @dtor2, ptr null}]
