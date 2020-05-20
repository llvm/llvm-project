; RUN: llc < %s -mtriple=i686-pc-win32 | FileCheck %s
; XFAIL: *

declare token @llvm.call.preallocated.setup(i32)
declare i8* @llvm.call.preallocated.arg(token, i32)

%Foo = type { i32, i32 }

declare void @init(%Foo*)



declare void @foo_p(%Foo* preallocated(%Foo))

define void @no_call() {
; CHECK-LABEL: _no_call:
  %t = call token @llvm.call.preallocated.setup(i32 1)
  %a = call i8* @llvm.call.preallocated.arg(token %t, i32 0) preallocated(%Foo)
  %b = bitcast i8* %a to %Foo*
  call void @init(%Foo* %b)
  ret void
}
