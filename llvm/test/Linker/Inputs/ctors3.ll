$foo = comdat any
%t = type { i8 }
@foo = global %t zeroinitializer, comdat
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @bar, i8* getelementptr (%t, %t* @foo, i32 0, i32 0) }]
define internal void @bar() comdat($foo) {
  ret void
}
