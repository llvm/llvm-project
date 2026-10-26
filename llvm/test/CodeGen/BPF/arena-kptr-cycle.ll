; RUN: opt -S -mtriple=bpfel -passes='bpf-arena-data,bpf-arena-kptr,function(bpf-arena-stack)' < %s | FileCheck %s

; The pass keeps the offsets at which a pointer of the kernel is stored. For
; the pass the indirect call in @outer can call @outer again, with 8 more, and
; so on. It has to give up on the offsets and come to an end.

declare ptr @bpf_task_get() section ".ksyms"

@ops = internal constant [2 x ptr] [ptr @outer, ptr @inner]

define internal i64 @inner(ptr %p) noinline {
  %t = load ptr, ptr %p
  %r = load i64, ptr %t
  ret i64 %r
}

define internal i64 @outer(ptr %p) noinline {
  %q = getelementptr i8, ptr %p, i64 8
  %f = load volatile ptr, ptr getelementptr (ptr, ptr @ops, i64 1)
  %r = call i64 %f(ptr %q)
  ret i64 %r
}

; CHECK-LABEL: define i64 @prog(
; CHECK:       %o = alloca [16 x i8], align 8, !bpf.stack.keep
define i64 @prog(ptr %ctx) section "syscall" {
  %o = alloca [16 x i8], align 8
  %t = call ptr @bpf_task_get()
  %s = getelementptr i8, ptr %o, i64 8
  store ptr %t, ptr %s
  %f = load volatile ptr, ptr @ops
  %r = call i64 %f(ptr %o)
  ret i64 %r
}
