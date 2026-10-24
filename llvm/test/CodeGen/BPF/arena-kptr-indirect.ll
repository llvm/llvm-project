; RUN: opt -S -mtriple=bpfel -passes='bpf-arena-kptr,function(bpf-arena-stack)' < %s | FileCheck %s
;
; The address of a local that holds a pointer of the kernel goes to an indirect
; call. Nothing here loads the pointer back, the function that is called may.
; The local stays in the stack of the program. One that holds a number goes to
; arena.

declare ptr @bpf_task_get()

; CHECK-LABEL: define i64 @prog(
; CHECK:       %c = alloca [8 x i8], align 8, !bpf.stack.keep
; CHECK:       %d = alloca [8 x i8], align 8, !bpf.arena.stack
define i64 @prog(ptr %ctx, ptr %f) section "syscall" {
  %c = alloca [8 x i8], align 8
  %d = alloca [8 x i8], align 8
  %t = call ptr @bpf_task_get()
  store ptr %t, ptr %c
  store i64 1, ptr %d
  %a = call i64 %f(ptr %c)
  %b = call i64 %f(ptr %d)
  %r = add i64 %a, %b
  ret i64 %r
}
