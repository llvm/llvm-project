define internal void @ctor1() {
  call void @func1()
  ret void
}

define internal void @ctor2() {
  ret void
}

define void @func1() {
  ret void
}

define void @unused() {
  ret void
}

@llvm.global_ctors = appending global[2 x{i32, ptr, ptr }] [
    {i32, ptr, ptr } { i32 2, ptr @ctor1, ptr null},
    {i32, ptr, ptr } { i32 7, ptr @ctor2, ptr null}]
