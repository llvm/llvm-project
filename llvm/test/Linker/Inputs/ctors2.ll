$foo = comdat any
@foo = global i8 1, comdat
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @bar, ptr @foo }]
define void @bar() comdat($foo) {
  ret void
}
