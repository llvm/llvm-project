; RUN: llc < %s -mtriple=i686-pc-win32 | FileCheck %s

declare token @llvm.call.preallocated.setup(i32)
declare i8* @llvm.call.preallocated.arg(token, i32)

%Foo = type { i32, i32 }

declare void @init(%Foo*)



declare void @foo_p(%Foo* preallocated(%Foo))

define void @one_preallocated() {
; CHECK-LABEL: _one_preallocated:
  %t = call token @llvm.call.preallocated.setup(i32 1)
  %a = call i8* @llvm.call.preallocated.arg(token %t, i32 0) preallocated(%Foo)
  %b = bitcast i8* %a to %Foo*
; CHECK: subl $8, %esp
; CHECK: calll _foo_p
  call void @foo_p(%Foo* preallocated(%Foo) %b) ["preallocated"(token %t)]
  ret void
}

define void @one_preallocated_two_blocks() {
; CHECK-LABEL: _one_preallocated_two_blocks:
  %t = call token @llvm.call.preallocated.setup(i32 1)
  br label %second
second:
  %a = call i8* @llvm.call.preallocated.arg(token %t, i32 0) preallocated(%Foo)
  %b = bitcast i8* %a to %Foo*
; CHECK: subl $8, %esp
; CHECK: calll _foo_p
  call void @foo_p(%Foo* preallocated(%Foo) %b) ["preallocated"(token %t)]
  ret void
}

define void @preallocated_with_store() {
; CHECK-LABEL: _preallocated_with_store:
; CHECK: subl $8, %esp
  %t = call token @llvm.call.preallocated.setup(i32 1)
; CHECK: leal (%esp), [[REGISTER:%[a-z]+]]
  %a = call i8* @llvm.call.preallocated.arg(token %t, i32 0) preallocated(%Foo)
  %b = bitcast i8* %a to %Foo*
  %p0 = getelementptr %Foo, %Foo* %b, i32 0, i32 0
  %p1 = getelementptr %Foo, %Foo* %b, i32 0, i32 1
  store i32 13, i32* %p0
  store i32 42, i32* %p1
; CHECK-DAG: movl $13, ([[REGISTER]])
; CHECK-DAG: movl $42, 4([[REGISTER]])
; CHECK-NOT: subl {{\$[0-9]+}}, %esp
; CHECK-NOT: pushl
; CHECK: calll _foo_p
  call void @foo_p(%Foo* preallocated(%Foo) %b) ["preallocated"(token %t)]
  ret void
}

define void @preallocated_with_init() {
; CHECK-LABEL: _preallocated_with_init:
; CHECK: subl $8, %esp
  %t = call token @llvm.call.preallocated.setup(i32 1)
; CHECK: leal (%esp), [[REGISTER:%[a-z]+]]
  %a = call i8* @llvm.call.preallocated.arg(token %t, i32 0) preallocated(%Foo)
  %b = bitcast i8* %a to %Foo*
; CHECK: pushl [[REGISTER]]
; CHECK: calll _init
  call void @init(%Foo* %b)
; CHECK-NOT: subl {{\$[0-9]+}}, %esp
; CHECK-NOT: pushl
; CHECK: calll _foo_p
  call void @foo_p(%Foo* preallocated(%Foo) %b) ["preallocated"(token %t)]
  ret void
}

declare void @foo_p_p(%Foo* preallocated(%Foo), %Foo* preallocated(%Foo))

define void @two_preallocated() {
; CHECK-LABEL: _two_preallocated:
  %t = call token @llvm.call.preallocated.setup(i32 2)
  %a1 = call i8* @llvm.call.preallocated.arg(token %t, i32 0) preallocated(%Foo)
  %b1 = bitcast i8* %a1 to %Foo*
  %a2 = call i8* @llvm.call.preallocated.arg(token %t, i32 1) preallocated(%Foo)
  %b2 = bitcast i8* %a2 to %Foo*
; CHECK: subl $16, %esp
; CHECK: calll _foo_p_p
  call void @foo_p_p(%Foo* preallocated(%Foo) %b1, %Foo* preallocated(%Foo) %b2) ["preallocated"(token %t)]
  ret void
}

declare void @foo_p_int(%Foo* preallocated(%Foo), i32)

define void @one_preallocated_one_normal() {
; CHECK-LABEL: _one_preallocated_one_normal:
; CHECK: subl $12, %esp
  %t = call token @llvm.call.preallocated.setup(i32 1)
; CHECK: leal (%esp), [[REGISTER:%[a-z]+]]
  %a = call i8* @llvm.call.preallocated.arg(token %t, i32 0) preallocated(%Foo)
  %b = bitcast i8* %a to %Foo*
; CHECK: pushl [[REGISTER]]
; CHECK: calll _init
  call void @init(%Foo* %b)
; CHECK-NOT: subl {{\$[0-9]+}}, %esp
; CHECK-NOT: pushl
; CHECK: movl $2, 8(%esp)
; CHECK: calll _foo_p_int
  call void @foo_p_int(%Foo* preallocated(%Foo) %b, i32 2) ["preallocated"(token %t)]
  ret void
}

declare void @foo_ret_p(%Foo* sret(%Foo), %Foo* preallocated(%Foo))

define void @nested_with_init() {
; CHECK-LABEL: _nested_with_init:
  %tmp = alloca %Foo

  %t1 = call token @llvm.call.preallocated.setup(i32 1)
; CHECK: subl $12, %esp
  %a1 = call i8* @llvm.call.preallocated.arg(token %t1, i32 0) preallocated(%Foo)
  %b1 = bitcast i8* %a1 to %Foo*
; CHECK: leal 4(%esp), [[REGISTER1:%[a-z]+]]

  %t2 = call token @llvm.call.preallocated.setup(i32 1)
; CHECK: subl $12, %esp
  %a2 = call i8* @llvm.call.preallocated.arg(token %t2, i32 0) preallocated(%Foo)
; CHECK: leal 4(%esp), [[REGISTER2:%[a-z]+]]
  %b2 = bitcast i8* %a2 to %Foo*

  call void @init(%Foo* %b2)
; CHECK: pushl [[REGISTER2]]
; CHECK: calll _init

  call void @foo_ret_p(%Foo* %b1, %Foo* preallocated(%Foo) %b2) ["preallocated"(token %t2)]
; CHECK-NOT: subl {{\$[0-9]+}}, %esp
; CHECK-NOT: pushl
; CHECK: calll _foo_ret_p
  call void @foo_ret_p(%Foo* %tmp, %Foo* preallocated(%Foo) %b1) ["preallocated"(token %t1)]
; CHECK-NOT: subl {{\$[0-9]+}}, %esp
; CHECK-NOT: pushl
; CHECK: calll _foo_ret_p
  ret void
}

declare void @foo_inreg_p(i32 inreg, %Foo* preallocated(%Foo))

define void @inreg() {
; CHECK-LABEL: _inreg:
  %t = call token @llvm.call.preallocated.setup(i32 1)
  %a = call i8* @llvm.call.preallocated.arg(token %t, i32 0) preallocated(%Foo)
  %b = bitcast i8* %a to %Foo*
; CHECK: subl $8, %esp
; CHECK: movl $9, %eax
; CHECK: calll _foo_inreg_p
  call void @foo_inreg_p(i32 9, %Foo* preallocated(%Foo) %b) ["preallocated"(token %t)]
  ret void
}

declare x86_thiscallcc void @foo_thiscall_p(i8*, %Foo* preallocated(%Foo))

define void @thiscall() {
; CHECK-LABEL: _thiscall:
  %t = call token @llvm.call.preallocated.setup(i32 1)
  %a = call i8* @llvm.call.preallocated.arg(token %t, i32 0) preallocated(%Foo)
  %b = bitcast i8* %a to %Foo*
; CHECK: subl $8, %esp
; CHECK: xorl %ecx, %ecx
; CHECK: calll _foo_thiscall_p
  call x86_thiscallcc void @foo_thiscall_p(i8* null, %Foo* preallocated(%Foo) %b) ["preallocated"(token %t)]
  ret void
}

declare x86_stdcallcc void @foo_stdcall_p(%Foo* preallocated(%Foo))
declare x86_stdcallcc void @i(i32)

define void @stdcall() {
; CHECK-LABEL: _stdcall:
  %t = call token @llvm.call.preallocated.setup(i32 1)
  %a = call i8* @llvm.call.preallocated.arg(token %t, i32 0) preallocated(%Foo)
  %b = bitcast i8* %a to %Foo*
; CHECK: subl $8, %esp
; CHECK: calll _foo_stdcall_p@8
  call x86_stdcallcc void @foo_stdcall_p(%Foo* preallocated(%Foo) %b) ["preallocated"(token %t)]
; CHECK-NOT: %esp
; CHECK: pushl
; CHECK: calll _i@4
  call x86_stdcallcc void @i(i32 0)
  ret void
}
