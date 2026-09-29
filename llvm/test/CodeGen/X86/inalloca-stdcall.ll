; RUN: llc < %s -mtriple=i686-pc-win32 | FileCheck %s

%Foo = type { i32, i32 }

declare x86_stdcallcc void @f(ptr inalloca(%Foo) %a)
declare x86_stdcallcc void @i(i32 %a)

define void @g() {
; CHECK-LABEL: _g:
  %b = alloca inalloca %Foo
; CHECK: pushl   %eax
; CHECK: pushl   %eax
  %f2 = getelementptr %Foo, ptr %b, i32 0, i32 1
  store i32 13, ptr %b
  store i32 42, ptr %f2
; CHECK: movl %esp, %eax
; CHECK: movl    $13, (%eax)
; CHECK: movl    $42, 4(%eax)
  call x86_stdcallcc void @f(ptr inalloca(%Foo) %b)
; CHECK: calll   _f@8
; CHECK-NOT: %esp
; CHECK: pushl
; CHECK: calll   _i@4
  call x86_stdcallcc void @i(i32 0)
  ret void
}
