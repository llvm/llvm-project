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

@llvm.global_dtors = appending global[2 x{i32, void() *, i8 * }] [
    {i32, void() *, i8 * } { i32 2, void() *@dtor1, i8 *null},
    {i32, void() *, i8 * } { i32 7, void() *@dtor2, i8 *null}]
