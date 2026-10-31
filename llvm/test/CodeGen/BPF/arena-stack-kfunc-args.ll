; RUN: opt -S -mtriple=bpfel -passes=bpf-arena-stack < %s | FileCheck %s
; RUN: opt -S -mtriple=bpfel -passes=bpf-arena-stack \
; RUN:   -bpf-arena-kfunc-args=fill_mask:1,other:0 < %s | FileCheck --check-prefix=ARENA %s

; What is passed to a kfunc stays where the kernel knows about it, unless the
; kfunc wants a pointer to arena in that argument.

declare void @fill_mask(ptr, ptr)

define i64 @local() {
; CHECK-LABEL: @local
; CHECK:       %n = alloca i64, align 8{{$}}
; CHECK:       %m = alloca i64, align 8{{$}}
; ARENA-LABEL: @local
; ARENA:       %n = alloca i64, align 8{{$}}
; ARENA:       %m = alloca i64, align 8, !bpf.arena.stack
  %n = alloca i64
  %m = alloca i64
  call void @fill_mask(ptr %n, ptr %m)
  %v = load i64, ptr %m
  ret i64 %v
}
