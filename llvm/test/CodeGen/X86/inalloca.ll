; RUN: llc < %s -mtriple=i686-pc-win32 | FileCheck %s

%Foo = type { i32, i32 }

declare void @f(%Foo* inalloca %b)

define void @a() {
; CHECK-LABEL: _a:
entry:
  %b = alloca inalloca %Foo
; CHECK: pushl   %eax
; CHECK: pushl   %eax
  %f1 = getelementptr %Foo, %Foo* %b, i32 0, i32 0
  %f2 = getelementptr %Foo, %Foo* %b, i32 0, i32 1
  store i32 13, i32* %f1
  store i32 42, i32* %f2
; CHECK: movl %esp, %eax
; CHECK: movl    $13, (%eax)
; CHECK: movl    $42, 4(%eax)
  call void @f(%Foo* inalloca %b)
; CHECK: calll   _f
  ret void
}

declare void @inreg_with_inalloca(i32 inreg %a, %Foo* inalloca %b)

define void @b() {
; CHECK-LABEL: _b:
entry:
  %b = alloca inalloca %Foo
; CHECK: pushl   %eax
; CHECK: pushl   %eax
  %f1 = getelementptr %Foo, %Foo* %b, i32 0, i32 0
  %f2 = getelementptr %Foo, %Foo* %b, i32 0, i32 1
  store i32 13, i32* %f1
  store i32 42, i32* %f2
; CHECK: movl %esp, %eax
; CHECK: movl    $13, (%eax)
; CHECK: movl    $42, 4(%eax)
  call void @inreg_with_inalloca(i32 inreg 1, %Foo* inalloca %b)
; CHECK: movl    $1, %eax
; CHECK: calll   _inreg_with_inalloca
  ret void
}

declare x86_thiscallcc void @thiscall_with_inalloca(i8* %a, %Foo* inalloca %b)

define void @c() {
; CHECK-LABEL: _c:
entry:
  %b = alloca inalloca %Foo
; CHECK: pushl   %eax
; CHECK: pushl   %eax
  %f1 = getelementptr %Foo, %Foo* %b, i32 0, i32 0
  %f2 = getelementptr %Foo, %Foo* %b, i32 0, i32 1
  store i32 13, i32* %f1
  store i32 42, i32* %f2
; CHECK: movl %esp, %eax
; CHECK-DAG: movl    $13, (%eax)
; CHECK-DAG: movl    $42, 4(%eax)
  call x86_thiscallcc void @thiscall_with_inalloca(i8* null, %Foo* inalloca %b)
; CHECK-DAG: xorl    %ecx, %ecx
; CHECK: calll   _thiscall_with_inalloca
  ret void
}
