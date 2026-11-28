; RUN: opt -S -mtriple=bpfel -passes=bpf-arena-data < %s | FileCheck %s

; Nothing but the kfunc uses the string: no copy.
; CHECK-NOT: @fmt.arena
; CHECK-DAG: @fmt = private unnamed_addr constant [3 x i8] c"%d\00", align 1{{$}}
@fmt = private unnamed_addr constant [3 x i8] c"%d\00", align 1

; A constant that is not static stays, whatever the program does with it.
; CHECK-DAG: @config = constant [2 x i32] [i32 1, i32 2]{{$}}
@config = constant [2 x i32] [i32 1, i32 2]

declare void @bpf_printk(ptr) section ".ksyms"

define i32 @prog(ptr %ctx, i64 %n) section "syscall" {
  call void @bpf_printk(ptr @fmt)
  %p = getelementptr i32, ptr @config, i64 %n
  %v = load i32, ptr %p
  ret i32 %v
}
