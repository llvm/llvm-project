; RUN: opt -S -mtriple=bpfel -passes=bpf-arena-data < %s | FileCheck %s
; RUN: opt -S -mtriple=bpfel -passes=bpf-arena-data \
; RUN:   -bpf-arena-kfunc-args=fill_mask:1,other:0 < %s | FileCheck --check-prefix=ARENA %s

; What is passed to a kfunc stays where the kernel knows about it, unless the
; kfunc wants a pointer to arena in that argument.

; CHECK:     @mask = internal global [4 x i64] zeroinitializer{{$}}
; CHECK:     @name = internal global [4 x i64] zeroinitializer{{$}}
; ARENA:     @mask = internal global [4 x i64] zeroinitializer, section ".bss.addr_space.1"
; ARENA:     @name = internal global [4 x i64] zeroinitializer{{$}}
@mask = internal global [4 x i64] zeroinitializer
@name = internal global [4 x i64] zeroinitializer

declare void @fill_mask(ptr, ptr)

define void @global() {
  call void @fill_mask(ptr @name, ptr @mask)
  ret void
}
