; RUN: llc %s -mtriple=x86_64-windows-msvc -o /dev/null 2>&1
; REQUIRES: asserts
; XFAIL: *

declare token @llvm.call.preallocated.setup(i32)
declare ptr @llvm.call.preallocated.arg(token, i32)

%Foo = type { i32, i32 }

declare x86_thiscallcc void @f(i32, ptr preallocated(%Foo))

define void @g() {
  %t = call token @llvm.call.preallocated.setup(i32 1)
  %a = call ptr @llvm.call.preallocated.arg(token %t, i32 0) preallocated(%Foo)
  call void @f(i32 0, ptr preallocated(%Foo) %a) ["preallocated"(token %t)]
  ret void
}
