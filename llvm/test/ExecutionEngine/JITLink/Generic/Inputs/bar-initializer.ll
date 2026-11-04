@initializer_bar_ran = global i32 0
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @initializer_bar, ptr null }]

define internal void @initializer_bar() {
  store i32 1, ptr @initializer_bar_ran
  ret void
}
