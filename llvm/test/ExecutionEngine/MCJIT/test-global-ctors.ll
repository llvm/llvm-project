; RUN: %lli -jit-kind=mcjit %s > /dev/null
; RUN: %lli %s > /dev/null
; XFAIL: darwin
@var = global i32 1, align 4
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @ctor_func, i8* null }]
@llvm.global_dtors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @dtor_func, i8* null }]

define i32 @main() nounwind {
entry:
  %0 = load i32, i32* @var, align 4
  ret i32 %0
}

define internal void @ctor_func() section ".text.startup" {
entry:
  store i32 0, i32* @var, align 4
  ret void
}

define internal void @dtor_func() section ".text.startup" {
entry:
  ret void
}
