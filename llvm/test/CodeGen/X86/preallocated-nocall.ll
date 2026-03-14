; RUN: llc < %s -mtriple=i686-pc-win32 | FileCheck %s
; REQUIRES: asserts
; XFAIL: *

declare token @llvm.call.preallocated.setup(i32)
declare ptr @llvm.call.preallocated.arg(token, i32)

%Foo = type { i32, i32 }

declare void @init(ptr)



declare void @foo_p(ptr preallocated(%Foo))

define void @no_call() {
; CHECK-LABEL: _no_call:
  %t = call token @llvm.call.preallocated.setup(i32 1)
  %a = call ptr @llvm.call.preallocated.arg(token %t, i32 0) preallocated(%Foo)
  call void @init(ptr %a)
  ret void
}
