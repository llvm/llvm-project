; RUN: llc < %s -mtriple=i686-pc-win32 | FileCheck %s

%Foo = type { i32, i32 }

declare x86_stdcallcc void @foo_byref_stdcall_p(%Foo* byref(%Foo))
declare x86_stdcallcc void @i(i32)

; byref does not imply a stack copy, so this should append 4 bytes,
; not 8.
define void @stdcall(%Foo* %value) {
; CHECK-LABEL: _stdcall:
; CHECK: pushl 4(%esp)
; CHECK: calll _foo_byref_stdcall_p@4
  call x86_stdcallcc void @foo_byref_stdcall_p(%Foo* byref(%Foo) %value)
; CHECK-NOT: %esp
; CHECK: pushl
; CHECK: calll _i@4
  call x86_stdcallcc void @i(i32 0)
  ret void
}
