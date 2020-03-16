; RUN: llc %s -mtriple=x86_64-windows-msvc -o /dev/null 2>&1
; REQUIRES: asserts
; XFAIL: *

declare token @llvm.call.preallocated.setup(i32)
declare i8* @llvm.call.preallocated.arg(token, i32)

%Foo = type { i32, i32 }

declare x86_thiscallcc void @f(i32, %Foo* preallocated(%Foo))

define void @g() {
  %t = call token @llvm.call.preallocated.setup(i32 1)
  %a = call i8* @llvm.call.preallocated.arg(token %t, i32 0) preallocated(%Foo)
  %b = bitcast i8* %a to %Foo*
  call void @f(i32 0, %Foo* preallocated(%Foo) %b) ["preallocated"(token %t)]
  ret void
}
