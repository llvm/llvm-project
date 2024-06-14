; RUN: llc < %s -mtriple=i686-pc-win32 | FileCheck %s

declare token @llvm.call.preallocated.setup(i32)
declare ptr @llvm.call.preallocated.arg(token, i32)

%Foo = type { i32, i32 }

declare void @init(ptr)



declare void @foo_p(ptr preallocated(%Foo))

define void @one_preallocated() {
; CHECK-LABEL: _one_preallocated:
  %t = call token @llvm.call.preallocated.setup(i32 1)
  %a = call ptr @llvm.call.preallocated.arg(token %t, i32 0) preallocated(%Foo)
; CHECK: subl $8, %esp
; CHECK: calll _foo_p
  call void @foo_p(ptr preallocated(%Foo) %a) ["preallocated"(token %t)]
  ret void
}

define void @one_preallocated_two_blocks() {
; CHECK-LABEL: _one_preallocated_two_blocks:
  %t = call token @llvm.call.preallocated.setup(i32 1)
  br label %second
second:
  %a = call ptr @llvm.call.preallocated.arg(token %t, i32 0) preallocated(%Foo)
; CHECK: subl $8, %esp
; CHECK: calll _foo_p
  call void @foo_p(ptr preallocated(%Foo) %a) ["preallocated"(token %t)]
  ret void
}

define void @preallocated_with_store() {
; CHECK-LABEL: _preallocated_with_store:
; CHECK: subl $8, %esp
  %t = call token @llvm.call.preallocated.setup(i32 1)
; CHECK: leal (%esp), [[REGISTER:%[a-z]+]]
  %a = call ptr @llvm.call.preallocated.arg(token %t, i32 0) preallocated(%Foo)
  %p1 = getelementptr %Foo, ptr %a, i32 0, i32 1
  store i32 13, ptr %a
  store i32 42, ptr %p1
; CHECK-DAG: movl $13, ([[REGISTER]])
; CHECK-DAG: movl $42, 4([[REGISTER]])
; CHECK-NOT: subl {{\$[0-9]+}}, %esp
; CHECK-NOT: pushl
; CHECK: calll _foo_p
  call void @foo_p(ptr preallocated(%Foo) %a) ["preallocated"(token %t)]
  ret void
}

define void @preallocated_with_init() {
; CHECK-LABEL: _preallocated_with_init:
; CHECK: subl $8, %esp
  %t = call token @llvm.call.preallocated.setup(i32 1)
; CHECK: leal (%esp), [[REGISTER:%[a-z]+]]
  %a = call ptr @llvm.call.preallocated.arg(token %t, i32 0) preallocated(%Foo)
; CHECK: pushl [[REGISTER]]
; CHECK: calll _init
  call void @init(ptr %a)
; CHECK-NOT: subl {{\$[0-9]+}}, %esp
; CHECK-NOT: pushl
; CHECK: calll _foo_p
  call void @foo_p(ptr preallocated(%Foo) %a) ["preallocated"(token %t)]
  ret void
}

declare void @foo_p_p(ptr preallocated(%Foo), ptr preallocated(%Foo))

define void @two_preallocated() {
; CHECK-LABEL: _two_preallocated:
  %t = call token @llvm.call.preallocated.setup(i32 2)
  %a1 = call ptr @llvm.call.preallocated.arg(token %t, i32 0) preallocated(%Foo)
  %a2 = call ptr @llvm.call.preallocated.arg(token %t, i32 1) preallocated(%Foo)
; CHECK: subl $16, %esp
; CHECK: calll _foo_p_p
  call void @foo_p_p(ptr preallocated(%Foo) %a1, ptr preallocated(%Foo) %a2) ["preallocated"(token %t)]
  ret void
}

declare void @foo_p_int(ptr preallocated(%Foo), i32)

define void @one_preallocated_one_normal() {
; CHECK-LABEL: _one_preallocated_one_normal:
; CHECK: subl $12, %esp
  %t = call token @llvm.call.preallocated.setup(i32 1)
; CHECK: leal (%esp), [[REGISTER:%[a-z]+]]
  %a = call ptr @llvm.call.preallocated.arg(token %t, i32 0) preallocated(%Foo)
; CHECK: pushl [[REGISTER]]
; CHECK: calll _init
  call void @init(ptr %a)
; CHECK-NOT: subl {{\$[0-9]+}}, %esp
; CHECK-NOT: pushl
; CHECK: movl $2, 8(%esp)
; CHECK: calll _foo_p_int
  call void @foo_p_int(ptr preallocated(%Foo) %a, i32 2) ["preallocated"(token %t)]
  ret void
}

declare void @foo_ret_p(ptr sret(%Foo), ptr preallocated(%Foo))

define void @nested_with_init() {
; CHECK-LABEL: _nested_with_init:
  %tmp = alloca %Foo

  %t1 = call token @llvm.call.preallocated.setup(i32 1)
; CHECK: subl $12, %esp
  %a1 = call ptr @llvm.call.preallocated.arg(token %t1, i32 0) preallocated(%Foo)
; CHECK: leal 4(%esp), [[REGISTER1:%[a-z]+]]

  %t2 = call token @llvm.call.preallocated.setup(i32 1)
; CHECK: subl $12, %esp
  %a2 = call ptr @llvm.call.preallocated.arg(token %t2, i32 0) preallocated(%Foo)
; CHECK: leal 4(%esp), [[REGISTER2:%[a-z]+]]

  call void @init(ptr %a2)
; CHECK: pushl [[REGISTER2]]
; CHECK: calll _init

  call void @foo_ret_p(ptr sret(%Foo) %a1, ptr preallocated(%Foo) %a2) ["preallocated"(token %t2)]
; CHECK-NOT: subl {{\$[0-9]+}}, %esp
; CHECK-NOT: pushl
; CHECK: calll _foo_ret_p
  call void @foo_ret_p(ptr sret(%Foo) %tmp, ptr preallocated(%Foo) %a1) ["preallocated"(token %t1)]
; CHECK-NOT: subl {{\$[0-9]+}}, %esp
; CHECK-NOT: pushl
; CHECK: calll _foo_ret_p
  ret void
}

declare void @foo_inreg_p(i32 inreg, ptr preallocated(%Foo))

define void @inreg() {
; CHECK-LABEL: _inreg:
  %t = call token @llvm.call.preallocated.setup(i32 1)
  %a = call ptr @llvm.call.preallocated.arg(token %t, i32 0) preallocated(%Foo)
; CHECK: subl $8, %esp
; CHECK: movl $9, %eax
; CHECK: calll _foo_inreg_p
  call void @foo_inreg_p(i32 inreg 9, ptr preallocated(%Foo) %a) ["preallocated"(token %t)]
  ret void
}

declare x86_thiscallcc void @foo_thiscall_p(ptr, ptr preallocated(%Foo))

define void @thiscall() {
; CHECK-LABEL: _thiscall:
  %t = call token @llvm.call.preallocated.setup(i32 1)
  %a = call ptr @llvm.call.preallocated.arg(token %t, i32 0) preallocated(%Foo)
; CHECK: subl $8, %esp
; CHECK: xorl %ecx, %ecx
; CHECK: calll _foo_thiscall_p
  call x86_thiscallcc void @foo_thiscall_p(ptr null, ptr preallocated(%Foo) %a) ["preallocated"(token %t)]
  ret void
}

declare x86_stdcallcc void @foo_stdcall_p(ptr preallocated(%Foo))
declare x86_stdcallcc void @i(i32)

define void @stdcall() {
; CHECK-LABEL: _stdcall:
  %t = call token @llvm.call.preallocated.setup(i32 1)
  %a = call ptr @llvm.call.preallocated.arg(token %t, i32 0) preallocated(%Foo)
; CHECK: subl $8, %esp
; CHECK: calll _foo_stdcall_p@8
  call x86_stdcallcc void @foo_stdcall_p(ptr preallocated(%Foo) %a) ["preallocated"(token %t)]
; CHECK-NOT: %esp
; CHECK: pushl
; CHECK: calll _i@4
  call x86_stdcallcc void @i(i32 0)
  ret void
}
