; RUN: opt -S -mtriple=bpfel -passes='bpf-arena-kptr,function(bpf-arena-stack)' < %s | FileCheck %s
; RUN: opt -mtriple=bpfel -passes='bpf-arena-kptr,function(bpf-arena-stack)' < %s \
; RUN:   | llc -mtriple=bpfel -mcpu=v4 -mattr=+arena-stack | FileCheck --check-prefix=ASM %s
;
; A local that holds a pointer of the kernel stays in the stack of the program.
; The function that gets its address is copied: in the copy the parameter is not
; cast.

declare ptr @bpf_task_get()
declare void @bpf_task_use(ptr)

define internal i64 @callee(ptr %c) noinline {
  %t = load ptr, ptr %c
  call void @bpf_task_use(ptr %t)
  %p = getelementptr i8, ptr %c, i64 8
  %v = load i64, ptr %p
  ret i64 %v
}

; CHECK-LABEL: define i64 @prog(
; CHECK:       %c = alloca [16 x i8], align 8, !bpf.stack.keep
; CHECK:       %env = alloca [8 x i8], align 8, !bpf.stack.keep
; CHECK:       %d = alloca [16 x i8], align 8, !bpf.arena.stack
; CHECK:       call i64 @callee.stack(ptr %c)
; CHECK:       call i64 @callee(ptr %d)
; CHECK:       call i64 @through(ptr %env)
; ASM-LABEL:   prog:
; ASM:         *(u64 *)(r10 - 16) = r0
; ASM:         r1 = r10
; ASM:         call callee.stack
define i64 @prog(ptr %ctx) section "syscall" {
  %c = alloca [16 x i8], align 8
  %env = alloca [8 x i8], align 8
  %d = alloca [16 x i8], align 8
  %t = call ptr @bpf_task_get()
  store ptr %t, ptr %c
  store ptr null, ptr %d
  store ptr %c, ptr %env
  %a = call i64 @callee(ptr %c)
  %b = call i64 @callee(ptr %d)
  %e = call i64 @through(ptr %env)
  %r = add i64 %a, %b
  %s = add i64 %r, %e
  ret i64 %s
}

; The address of such a local in another local: that one stays too, and what is
; loaded from it is a pointer to the stack.
; CHECK-LABEL: define internal i64 @through(ptr %env)
; CHECK:       %c = load ptr, ptr %env, align 8, !bpf.stack.ptr
; CHECK:       call i64 @callee.stack(ptr %c)
; ASM-LABEL:   through:
; ASM-NOT:     addr_space_cast
; ASM:         call callee.stack
define internal i64 @through(ptr %env) noinline {
  %c = load ptr, ptr %env
  %r = call i64 @callee(ptr %c)
  ret i64 %r
}

; CHECK-LABEL: define internal i64 @callee.stack(ptr %c)
; ASM-LABEL:   callee.stack:
; ASM-NOT:     addr_space_cast(r1
; ASM:         call bpf_task_use
