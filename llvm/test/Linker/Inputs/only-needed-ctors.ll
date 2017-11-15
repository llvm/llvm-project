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

@llvm.global_ctors = appending global[2 x{i32, void() *, i8 * }] [
    {i32, void() *, i8 * } { i32 2, void() *@ctor1, i8 *null},
    {i32, void() *, i8 * } { i32 7, void() *@ctor2, i8 *null}]
