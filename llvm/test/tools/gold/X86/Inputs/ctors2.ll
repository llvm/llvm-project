@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @bar, i8* null }]

define void @bar() {
  ret void
}
