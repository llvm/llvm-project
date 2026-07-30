$foo = comdat any
%t = type { i8 }
@foo = global %t zeroinitializer, comdat
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @bar, ptr @foo }]
define internal void @bar() comdat($foo) {
  ret void
}
