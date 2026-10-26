; RUN: opt -S -mtriple=bpfel -passes='bpf-arena-kptr,function(bpf-arena-stack)' < %s | FileCheck %s

; The address of a local that holds a pointer of the kernel is loaded from
; another local and stored where it cannot be followed. A pointer to the stack
; of the program must not get there, so the local goes to arena.

declare ptr @bpf_task_get()
declare void @bpf_task_use(ptr)

@g = global ptr null

; CHECK-LABEL: define i64 @prog(
; CHECK:       %c = alloca [16 x i8], align 8, !bpf.arena.stack
define i64 @prog(ptr %ctx) section "syscall" {
  %c = alloca [16 x i8], align 8
  %env = alloca [8 x i8], align 8
  %t = call ptr @bpf_task_get()
  store ptr %t, ptr %c
  store ptr %c, ptr %env
  call void @publish(ptr %env)
  %u = load ptr, ptr %c
  call void @bpf_task_use(ptr %u)
  ret i64 0
}

; CHECK-LABEL: define internal void @publish(
; CHECK:       %c = load ptr, ptr %env, align 8{{$}}
define internal void @publish(ptr %env) noinline {
  %c = load ptr, ptr %env
  store ptr %c, ptr @g
  ret void
}
