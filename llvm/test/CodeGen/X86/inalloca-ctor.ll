; RUN: llc < %s -mtriple=i686-pc-win32 | FileCheck %s

%Foo = type { i32, i32 }

%frame = type { %Foo, i32, %Foo }

declare void @f(ptr inalloca(%frame) %a)

declare void @Foo_ctor(ptr %this)

define void @g() {
entry:
  %args = alloca inalloca %frame
  %c = getelementptr %frame, ptr %args, i32 0, i32 2
; CHECK: pushl   %eax
; CHECK: subl    $16, %esp
; CHECK: movl %esp,
  call void @Foo_ctor(ptr %c)
; CHECK: leal 12(%{{.*}}),
; CHECK-NEXT: pushl
; CHECK-NEXT: calll _Foo_ctor
; CHECK: addl $4, %esp
  %b = getelementptr %frame, ptr %args, i32 0, i32 1
  store i32 42, ptr %b
; CHECK: movl $42,
  call void @Foo_ctor(ptr %args)
; CHECK-NEXT: pushl
; CHECK-NEXT: calll _Foo_ctor
; CHECK: addl $4, %esp
  call void @f(ptr inalloca(%frame) %args)
; CHECK: calll   _f
  ret void
}
