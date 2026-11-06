; RUN: opt -S -mtriple=bpfel -passes=bpf-arena-data < %s | FileCheck %s

; What gets to a helper or a kfunc stays where the kernel knows about it, also
; through a static function. The type of a parameter says that it wants arena.
; A declaration without section .ksyms is a function in another file.
; What gets a pointer of the kernel stays too: the verifier does not let
; a program store one in arena.

; CHECK: @name = internal global [4 x i64] zeroinitializer{{$}}
; CHECK: @mask = internal global [4 x i64] zeroinitializer, section ".bss.addr_space.1"
; CHECK: @key = internal global i32 0{{$}}
; CHECK: @other = internal global i32 0, section ".bss.addr_space.1"
; CHECK: @saved = internal global ptr null{{$}}
@name = internal global [4 x i64] zeroinitializer
@mask = internal global [4 x i64] zeroinitializer
@key = internal global i32 0
@other = internal global i32 0
@saved = internal global ptr null

declare void @fill_mask(ptr, ptr addrspace(1)) section ".ksyms"
declare void @other_file(ptr)

define internal ptr @inner(ptr %k) noinline {
  %v = call ptr inttoptr (i64 1 to ptr)(ptr null, ptr %k)
  ret ptr %v
}

define internal ptr @outer(ptr %k) noinline {
  %v = call ptr @inner(ptr %k)
  ret ptr %v
}

define ptr @global() {
  call void @fill_mask(ptr @name, ptr addrspace(1) addrspacecast (ptr @mask to ptr addrspace(1)))
  call void @other_file(ptr @other)
  %v = call ptr @outer(ptr @key)
  store ptr %v, ptr @saved
  ret ptr %v
}
