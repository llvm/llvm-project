; RUN: opt -S -mtriple=bpfel -passes='bpf-arena-kptr,function(bpf-arena-stack)' < %s | FileCheck %s

; A local that gets what may be a pointer of the kernel stays in the stack of
; the program, whatever else is done with it.

declare ptr @bpf_task_get() section ".ksyms"

define internal i64 @use(ptr %e) noinline {
  %t = load ptr, ptr %e
  %r = load i64, ptr %t
  ret i64 %r
}

; The kernel calls a callback with pointers of its own.
; CHECK-LABEL: define internal i64 @cb(
; CHECK:       %w = alloca [16 x i8], align 8, !bpf.stack.keep
define internal i64 @cb(ptr %map, ptr %key, ptr %val, ptr %data) {
  %w = alloca [16 x i8], align 8
  store ptr %val, ptr %w
  %r = call i64 @use(ptr %w)
  ret i64 %r
}

define i64 @callback(ptr %ctx) section "syscall" {
  %r = call i64 inttoptr (i64 164 to ptr)(ptr null, ptr @cb, ptr null, i64 0)
  ret i64 %r
}

define internal void @fill(ptr %e) noinline {
  %t = call ptr @bpf_task_get()
  store ptr %t, ptr %e
  ret void
}

; A callee stores it. Nothing loads it from %a, it is copied.
; CHECK-LABEL: define i64 @copy(
; CHECK:       %a = alloca [16 x i8], align 8, !bpf.stack.keep
; CHECK:       %b = alloca [16 x i8], align 8, !bpf.stack.keep
define i64 @copy(ptr %ctx) section "syscall" {
  %a = alloca [16 x i8], align 8
  %b = alloca [16 x i8], align 8
  call void @fill(ptr %a)
  call void @llvm.memcpy.p0.p0.i64(ptr %b, ptr %a, i64 16, i1 false)
  %r = call i64 @use(ptr %b)
  ret i64 %r
}

define internal i64 @run(ptr %e, ptr %f) noinline {
  %r = call i64 %f(ptr %e)
  ret i64 %r
}

; The address gets to an indirect call through a function.
; CHECK-LABEL: define i64 @closure(
; CHECK:       %e = alloca [16 x i8], align 8, !bpf.stack.keep
define i64 @closure(ptr %ctx, ptr %f) section "syscall" {
  %e = alloca [16 x i8], align 8
  %t = call ptr @bpf_task_get()
  store ptr %t, ptr %e
  %r = call i64 @run(ptr %e, ptr %f)
  ret i64 %r
}

@table = internal constant [1 x ptr] [ptr @take]

; The pointer comes through an indirect call.
; CHECK-LABEL: define internal i32 @take(
; CHECK:       %e = alloca [16 x i8], align 8, !bpf.stack.keep
define internal i32 @take(ptr %t) noinline {
  %e = alloca [16 x i8], align 8
  store ptr %t, ptr %e
  %r = call i64 @use(ptr %e)
  %r32 = trunc i64 %r to i32
  ret i32 %r32
}

define i32 @indirect(ptr %ctx, i64 %n) section "syscall" {
  %p = getelementptr ptr, ptr @table, i64 %n
  %f = load ptr, ptr %p
  %t = call ptr @bpf_task_get()
  %r = call i32 %f(ptr %t)
  ret i32 %r
}

; CHECK-LABEL: define i64 @select(
; CHECK:       %a = alloca [16 x i8], align 8, !bpf.stack.keep
; CHECK:       %b = alloca [16 x i8], align 8, !bpf.stack.keep
define i64 @select(ptr %ctx, i1 %c) section "syscall" {
  %a = alloca [16 x i8], align 8
  %b = alloca [16 x i8], align 8
  %p = select i1 %c, ptr %a, ptr %b
  %t = call ptr @bpf_task_get()
  store ptr %t, ptr %p
  %r = call i64 @use(ptr %p)
  ret i64 %r
}

; CHECK-LABEL: define i64 @variable_offset(
; CHECK:       %s = alloca [40 x i8], align 8, !bpf.stack.keep
define i64 @variable_offset(ptr %ctx, i64 %i) section "syscall" {
  %s = alloca [40 x i8], align 8
  %t = call ptr @bpf_task_get()
  store ptr %t, ptr %s
  %p = getelementptr i8, ptr %s, i64 %i
  store i8 1, ptr %p
  %r = call i64 @use(ptr %s)
  ret i64 %r
}

; CHECK-LABEL: define i64 @large(
; CHECK:       %s = alloca [264 x i8], align 8, !bpf.stack.keep
define i64 @large(ptr %ctx) section "syscall" {
  %s = alloca [264 x i8], align 8
  %t = call ptr @bpf_task_get()
  store ptr %t, ptr %s
  %r = call i64 @use(ptr %s)
  ret i64 %r
}

; An integer from the context that becomes a pointer. One that does not is
; a number.
; CHECK-LABEL: define i64 @from_ctx(
; CHECK:       %s = alloca [16 x i8], align 8, !bpf.stack.keep
; CHECK:       %n = alloca [16 x i8], align 8, !bpf.arena.stack
define i64 @from_ctx(ptr %ctx) section "tc" {
  %s = alloca [16 x i8], align 8
  %n = alloca [16 x i8], align 8
  %len = load i32, ptr %ctx
  store i32 %len, ptr %n
  %r0 = call i64 @use(ptr %n)
  %f = getelementptr i8, ptr %ctx, i64 80
  %end32 = load i32, ptr %f
  %end64 = zext i32 %end32 to i64
  %end = inttoptr i64 %end64 to ptr
  store ptr %end, ptr %s
  %r = call i64 @use(ptr %s)
  ret i64 %r
}

@map = global i64 0, section ".maps"
@ksym = external global i64, section ".ksyms"
@data = global i64 0
@in_arena = global i64 0, section ".bss.addr_space.1"

; The addresses of a map, of a variable of the kernel and of data that is not in
; arena are pointers of the kernel.
; CHECK-LABEL: define i64 @addresses(
; CHECK:       %a = alloca [8 x i8], align 8, !bpf.stack.keep
; CHECK:       %b = alloca [8 x i8], align 8, !bpf.stack.keep
; CHECK:       %c = alloca [8 x i8], align 8, !bpf.stack.keep
; CHECK:       %d = alloca [8 x i8], align 8, !bpf.arena.stack
define i64 @addresses(ptr %ctx) section "syscall" {
  %a = alloca [8 x i8], align 8
  %b = alloca [8 x i8], align 8
  %c = alloca [8 x i8], align 8
  %d = alloca [8 x i8], align 8
  store ptr @map, ptr %a
  store i64 ptrtoint (ptr @ksym to i64), ptr %b
  store ptr @data, ptr %c
  store ptr @in_arena, ptr %d
  %r0 = call i64 @use(ptr %a)
  %r1 = call i64 @use(ptr %b)
  %r2 = call i64 @use(ptr %c)
  %r3 = call i64 @use(ptr %d)
  ret i64 %r3
}

define internal i64 @use_env(ptr %env) noinline {
  %c = load ptr, ptr %env
  %r = call i64 @use(ptr %c)
  ret i64 %r
}

; The address of a local that stays is a pointer to the stack of the program.
; CHECK-LABEL: define i64 @address_of_local(
; CHECK:       %c = alloca [16 x i8], align 8, !bpf.stack.keep
; CHECK:       %env = alloca [8 x i8], align 8, !bpf.stack.keep
define i64 @address_of_local(ptr %ctx) section "syscall" {
  %c = alloca [16 x i8], align 8
  %env = alloca [8 x i8], align 8
  %t = call ptr @bpf_task_get()
  store ptr %t, ptr %c
  store ptr %c, ptr %env
  %r = call i64 @use_env(ptr %env)
  ret i64 %r
}

; Numbers and addresses in arena are fine in arena.
; CHECK-LABEL: define i64 @number(
; CHECK:       %s = alloca [16 x i8], align 8, !bpf.arena.stack
; CHECK:       %p = alloca [8 x i8], align 8, !bpf.arena.stack
define i64 @number(ptr %ctx, i64 %n) section "syscall" {
  %s = alloca [16 x i8], align 8
  %p = alloca [8 x i8], align 8
  store i64 %n, ptr %s
  store ptr %s, ptr %p
  %r = call i64 @use_env(ptr %p)
  ret i64 %r
}
