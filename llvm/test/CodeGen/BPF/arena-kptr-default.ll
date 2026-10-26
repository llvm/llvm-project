; RUN: llc -mtriple=bpfel -mcpu=v4 -mattr=+arena < %s | FileCheck %s

; The pass is a part of the feature.

declare ptr @bpf_task_get() section ".ksyms"
declare void @bpf_task_use(ptr) section ".ksyms"

define internal void @use(ptr %c) noinline {
  %t = load ptr, ptr %c
  call void @bpf_task_use(ptr %t)
  ret void
}

; A pointer of the kernel that is loaded back: the local stays on r10.
; CHECK-LABEL: read_back:
; CHECK-NOT:   r12
; CHECK:       *(u64 *)(r10 - 16) = r0
; CHECK:       call use
define void @read_back(ptr %ctx) section "syscall" {
  %c = alloca { ptr, i64 }
  %t = call ptr @bpf_task_get()
  store ptr %t, ptr %c
  call void @use(ptr %c)
  ret void
}

define internal i64 @number(ptr %c) noinline {
  %f = getelementptr i8, ptr %c, i64 8
  %v = load i64, ptr %f
  ret i64 %v
}

; Nothing loads it, but the verifier does not let it go to arena.
; CHECK-LABEL: write_only:
; CHECK-NOT:   r12
; CHECK:       *(u64 *)(r10 - 16) = r0
define i64 @write_only(ptr %ctx) section "syscall" {
  %c = alloca { ptr, i64 }
  %t = call ptr @bpf_task_get()
  store ptr %t, ptr %c
  %f = getelementptr i8, ptr %c, i64 8
  store i64 1, ptr %f
  %r = call i64 @number(ptr %c)
  ret i64 %r
}

define internal void @use2(ptr %c) noinline {
  ret void
}

@fns = internal global [2 x ptr] [ptr @use, ptr @use2]

; The function that is called is not known. Those that may be called get
; pointers to the stack only and don't cast them.
; CHECK-LABEL: indirect:
; CHECK-NOT:   r12
; CHECK:       *(u64 *)(r10 - 8) = r0
; CHECK:       callx
define void @indirect(ptr %ctx, i64 %n) section "syscall" {
  %c = alloca ptr
  %p = getelementptr ptr, ptr @fns, i64 %n
  %f = load ptr, ptr %p
  %t = call ptr @bpf_task_get()
  store ptr %t, ptr %c
  call void %f(ptr %c)
  %u = load ptr, ptr %c
  call void @bpf_task_use(ptr %u)
  ret void
}
