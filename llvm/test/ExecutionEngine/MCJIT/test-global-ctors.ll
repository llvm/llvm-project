; RUN: %lli -jit-kind=mcjit %s > /dev/null
; RUN: %lli %s > /dev/null
; UNSUPPORTED: target={{.*}}-darwin{{.*}}
@var = global i32 1, align 4
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @ctor_func, ptr null }]
@llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @dtor_func, ptr null }]

define i32 @main() nounwind {
entry:
  %0 = load i32, ptr @var, align 4
  ret i32 %0
}

define internal void @ctor_func() section ".text.startup" {
entry:
  store i32 0, ptr @var, align 4
  ret void
}

define internal void @dtor_func() section ".text.startup" {
entry:
  ret void
}
