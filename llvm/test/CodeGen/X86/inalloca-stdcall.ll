; RUN: llc < %s -mtriple=i686-pc-win32 | FileCheck %s

%Foo = type { i32, i32 }

declare x86_stdcallcc void @f(%Foo* inalloca %a)
declare x86_stdcallcc void @i(i32 %a)

define void @g() {
; CHECK-LABEL: _g:
; CHECK: movl    %esp, %ebp
  %b = alloca inalloca %Foo
; CHECK: movl    %esp, %[[tmp_sp:.*]]
; CHECK: leal    -8(%[[tmp_sp]]), %esp
  %f1 = getelementptr %Foo, %Foo* %b, i32 0, i32 0
  %f2 = getelementptr %Foo, %Foo* %b, i32 0, i32 1
  store i32 13, i32* %f1
  store i32 42, i32* %f2
; CHECK: movl    $13, -8(%[[tmp_sp]])
; CHECK: movl    $42, -4(%[[tmp_sp]])
  call x86_stdcallcc void @f(%Foo* inalloca %b)
; CHECK: calll   _f@8
; CHECK-NOT: %esp
; CHECK: pushl
; CHECK: calll   _i@4
  call x86_stdcallcc void @i(i32 0)
  ret void
}
