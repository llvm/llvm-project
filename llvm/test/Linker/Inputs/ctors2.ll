$foo = comdat any
@foo = global i8 1, comdat
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @bar, i8* @foo }]
define void @bar() comdat($foo) {
  ret void
}
