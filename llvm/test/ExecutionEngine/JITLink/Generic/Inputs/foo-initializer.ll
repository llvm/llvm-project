@initializer_foo_ran = global i32 0
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @initializer_foo, ptr null }]

define internal void @initializer_foo() {
  store i32 1, ptr @initializer_foo_ran
  ret void
}
