; RUN: opt -S -mtriple=bpfel -passes='bpf-arena-kptr,function(bpf-arena-stack)' < %s | FileCheck %s

; Which loads yield pointers to the stack of the program.

declare ptr @bpf_task_get() section ".ksyms"

@map = global i64 0, section ".maps"

; Only the field that holds a pointer to the stack.
; CHECK-LABEL: define void @fields(
; CHECK:       %p0 = load ptr, ptr %env, align 8, !bpf.stack.ptr
; CHECK:       %p1 = load ptr, ptr %f, align 8{{$}}
define void @fields(ptr %ctx, ptr %heap) section "syscall" {
  %c = alloca ptr
  %env = alloca [16 x i8], align 8
  %t = call ptr @bpf_task_get()
  store ptr %t, ptr %c
  store ptr %c, ptr %env
  %f = getelementptr i8, ptr %env, i64 8
  store ptr %heap, ptr %f
  %p0 = load ptr, ptr %env
  %p1 = load ptr, ptr %f
  ret void
}

; Other pointers of the kernel are not pointers to the stack.
; CHECK-LABEL: define void @not_stack_ptr(
; CHECK:       %lt = load ptr, ptr %c, align 8{{$}}
; CHECK:       %lm = load ptr, ptr %g, align 8{{$}}
define void @not_stack_ptr(ptr %ctx) section "syscall" {
  %c = alloca [16 x i8], align 8
  %t = call ptr @bpf_task_get()
  store ptr %t, ptr %c
  %g = getelementptr i8, ptr %c, i64 8
  store ptr @map, ptr %g
  %lt = load ptr, ptr %c
  %lm = load ptr, ptr %g
  ret void
}

; A load at an offset that is not known.
; CHECK-LABEL: define void @load_unknown(
; CHECK:       %l = load ptr, ptr %p, align 8, !bpf.stack.ptr
define void @load_unknown(ptr %ctx, i64 %i) section "syscall" {
  %c = alloca ptr
  %env = alloca [16 x i8], align 8
  %t = call ptr @bpf_task_get()
  store ptr %t, ptr %c
  store ptr %c, ptr %env
  %p = getelementptr i8, ptr %env, i64 %i
  %l = load ptr, ptr %p
  ret void
}

define internal void @put8(ptr %e, ptr %v) noinline {
  %f = getelementptr i8, ptr %e, i64 8
  store ptr %v, ptr %f
  ret void
}

; A callee stores at a constant offset from an address that is not known.
; CHECK-LABEL: define void @unknown_plus_delta(
; CHECK:       %l = load ptr, ptr %q, align 8, !bpf.stack.ptr
define void @unknown_plus_delta(ptr %ctx, i64 %i) section "syscall" {
  %c = alloca ptr
  %s = alloca [40 x i8], align 8
  %t = call ptr @bpf_task_get()
  store ptr %t, ptr %c
  %p = getelementptr i8, ptr %s, i64 %i
  call void @put8(ptr %p, ptr %c)
  %q = getelementptr i8, ptr %s, i64 16
  %l = load ptr, ptr %q
  ret void
}

; A copy of unknown length to a constant offset.
; CHECK-LABEL: define void @delta_unknown(
; CHECK:       %l = load ptr, ptr %q, align 8, !bpf.stack.ptr
define void @delta_unknown(ptr %ctx, i64 %n) section "syscall" {
  %c = alloca ptr
  %a = alloca ptr
  %b = alloca [24 x i8], align 8
  %t = call ptr @bpf_task_get()
  store ptr %t, ptr %c
  store ptr %c, ptr %a
  %d = getelementptr i8, ptr %b, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %d, ptr %a, i64 %n, i1 false)
  %q = getelementptr i8, ptr %b, i64 16
  %l = load ptr, ptr %q
  ret void
}

; CHECK-LABEL: define internal void @ld_field(
; CHECK:       %c = load ptr, ptr %e, align 8, !bpf.stack.ptr
define internal void @ld_field(ptr %e) noinline {
  %c = load ptr, ptr %e
  ret void
}

; The callee gets the address of a field.
define void @callee_offset(ptr %ctx) section "syscall" {
  %c = alloca ptr
  %env = alloca [16 x i8], align 8
  %t = call ptr @bpf_task_get()
  store ptr %t, ptr %c
  %f = getelementptr i8, ptr %env, i64 8
  store ptr %c, ptr %f
  call void @ld_field(ptr %f)
  ret void
}

; CHECK-LABEL: define internal void @ld_slot(
; CHECK:       %c = load ptr, ptr %e, align 8, !bpf.stack.ptr
define internal void @ld_slot(ptr %e) noinline {
  %c = load ptr, ptr %e
  ret void
}

; The caller stored it at an offset that is not known.
define void @callee_slot_unknown(ptr %ctx, i64 %i) section "syscall" {
  %c = alloca ptr
  %env = alloca [24 x i8], align 8
  %t = call ptr @bpf_task_get()
  store ptr %t, ptr %c
  %p = getelementptr i8, ptr %env, i64 %i
  store ptr %c, ptr %p
  %f = getelementptr i8, ptr %env, i64 8
  call void @ld_slot(ptr %f)
  ret void
}

; CHECK-LABEL: define internal void @ld_arg(
; CHECK:       %c = load ptr, ptr %e, align 8, !bpf.stack.ptr
define internal void @ld_arg(ptr %e) noinline {
  %c = load ptr, ptr %e
  ret void
}

; The callee gets an address at an offset that is not known.
define void @callee_arg_unknown(ptr %ctx, i64 %i) section "syscall" {
  %c = alloca ptr
  %env = alloca [24 x i8], align 8
  %t = call ptr @bpf_task_get()
  store ptr %t, ptr %c
  %f = getelementptr i8, ptr %env, i64 8
  store ptr %c, ptr %f
  %p = getelementptr i8, ptr %env, i64 %i
  call void @ld_arg(ptr %p)
  ret void
}

@tbl = internal constant [1 x ptr] [ptr @ld_indirect]

; CHECK-LABEL: define internal void @ld_indirect(
; CHECK:       %c = load ptr, ptr %e, align 8, !bpf.stack.ptr
define internal void @ld_indirect(ptr %e, i8 %x) noinline {
  %c = load ptr, ptr %e
  ret void
}

; What the caller stored is seen by a function that is called indirectly.
define void @indirect_slots(ptr %ctx) section "syscall" {
  %c = alloca ptr
  %env = alloca ptr
  %t = call ptr @bpf_task_get()
  store ptr %t, ptr %c
  store ptr %c, ptr %env
  %f = load ptr, ptr @tbl
  call void %f(ptr %env, i8 0)
  ret void
}
