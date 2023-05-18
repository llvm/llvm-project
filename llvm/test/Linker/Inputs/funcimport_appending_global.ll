@v = weak global i8 1
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @foo, ptr @v}]

define void @foo() {
  ret void
}
