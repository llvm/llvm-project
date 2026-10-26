; RUN: opt -S -mtriple=bpfel -passes='bpf-arena-kptr,function(bpf-arena-stack)' < %s | FileCheck %s

; An address does not have to be an argument to get to a helper or a kfunc.
; The verifier sees a number in these cases, so it cannot tell that the memory
; is of the wrong kind.

declare i64 @bpf_read(ptr, i64) section ".ksyms"

; As a number in memory that the kernel gets: bpf_snprintf("%s"), bpf_sys_bpf().
; CHECK-LABEL: define i64 @in_memory(
; CHECK:       %name = alloca [8 x i8], align 1, !bpf.stack.keep
; CHECK:       %args = alloca [1 x i64], align 8, !bpf.stack.keep
define i64 @in_memory(ptr %ctx) section "syscall" {
  %name = alloca [8 x i8]
  %args = alloca [1 x i64]
  %n = ptrtoint ptr %name to i64
  store i64 %n, ptr %args
  %r = call i64 inttoptr (i64 165 to ptr)(ptr null, i32 0, ptr null, ptr %args, i32 8)
  ret i64 %r
}

; As a number, after select and phi, with an offset that is not constant.
; CHECK-LABEL: define i64 @number(
; CHECK:       %a = alloca [2 x i64], align 8, !bpf.stack.keep
; CHECK:       %b = alloca [2 x i64], align 8, !bpf.stack.keep
; CHECK:       %c = alloca [2 x i64], align 8, !bpf.stack.keep
define i64 @number(ptr %ctx, i1 %x, i64 %i) section "syscall" {
entry:
  %a = alloca [2 x i64]
  %b = alloca [2 x i64]
  %c = alloca [2 x i64]
  %s = select i1 %x, ptr %a, ptr %b
  br i1 %x, label %other, label %join
other:
  %e = getelementptr i64, ptr %c, i64 %i
  br label %join
join:
  %p = phi ptr [ %s, %entry ], [ %e, %other ]
  %n = ptrtoint ptr %p to i64
  %r = call i64 @bpf_read(ptr null, i64 %n)
  ret i64 %r
}

define internal i64 @read(ptr %src) noinline {
  %p = load ptr, ptr %src
  %r = call i64 @bpf_read(ptr %p, i64 0)
  ret i64 %r
}

; Through memory: a callee loads the address and passes it on. %src does not
; get to the kernel itself.
; CHECK-LABEL: define i64 @through_memory(
; CHECK:       %buf = alloca i64, align 8, !bpf.stack.keep
; CHECK:       %src = alloca ptr, align 8, !bpf.stack.keep
define i64 @through_memory(ptr %ctx) section "syscall" {
  %buf = alloca i64
  %src = alloca ptr
  store ptr %buf, ptr %src
  %r = call i64 @read(ptr %src)
  ret i64 %r
}

; Other files call a function that is not static, and for them its parameters
; are in arena. Only a plain use changes that: what @report gets goes to arena.
define i64 @report(ptr %what) noinline {
  %args = alloca [1 x i64]
  %n = ptrtoint ptr %what to i64
  store i64 %n, ptr %args
  %r = call i64 inttoptr (i64 165 to ptr)(ptr null, i32 0, ptr null, ptr %args, i32 8)
  ret i64 %r
}

; CHECK-LABEL: define i64 @not_static(
; CHECK:       %what = alloca [8 x i8], align 1, !bpf.arena.stack
define i64 @not_static(ptr %ctx) section "syscall" {
  %what = alloca [8 x i8]
  %r = call i64 @report(ptr %what)
  ret i64 %r
}

define internal i64 @print(ptr %what) noinline {
  %args = alloca [1 x i64]
  %n = ptrtoint ptr %what to i64
  store i64 %n, ptr %args
  %r = call i64 inttoptr (i64 165 to ptr)(ptr null, i32 0, ptr null, ptr %args, i32 8)
  ret i64 %r
}

; The same through a static function.
define i64 @report2(ptr %what) noinline {
  %r = call i64 @print(ptr %what)
  ret i64 %r
}

; CHECK-LABEL: define i64 @not_static2(
; CHECK:       %what = alloca [8 x i8], align 1, !bpf.arena.stack
define i64 @not_static2(ptr %ctx) section "syscall" {
  %what = alloca [8 x i8]
  %r = call i64 @report2(ptr %what)
  ret i64 %r
}

declare void @other_file(ptr)
@list = internal global ptr null

define internal void @link(ptr %node, ptr %prev) noinline {
  store ptr %prev, ptr %node
  ret void
}

; A number can be data for the kernel: "%lx". It does not count if the address
; also goes where only an address in arena makes sense: to the heap, also in
; a callee, to a global variable, to another file.
; CHECK-LABEL: define i64 @weak(
; CHECK:       %a = alloca i64, align 8, !bpf.arena.stack
; CHECK:       %b = alloca i64, align 8, !bpf.arena.stack
; CHECK:       %c = alloca i64, align 8, !bpf.arena.stack
; CHECK:       %d = alloca i64, align 8, !bpf.arena.stack
; CHECK:       %e = alloca i64, align 8, !bpf.stack.keep
define i64 @weak(ptr %ctx) section "syscall" {
  %a = alloca i64
  %b = alloca i64
  %c = alloca i64
  %d = alloca i64
  %e = alloca i64
  %heap = load ptr, ptr %ctx
  %r0 = call i64 @print(ptr %a)
  store ptr %a, ptr %heap
  %r1 = call i64 @print(ptr %b)
  call void @link(ptr %heap, ptr %b)
  %r2 = call i64 @print(ptr %c)
  store ptr %c, ptr @list
  %r3 = call i64 @print(ptr %d)
  call void @other_file(ptr %d)
  %r4 = call i64 @print(ptr %e)
  ret i64 %r4
}

; A pointer that a helper or a kfunc gets counts whatever else happens to it.
; CHECK-LABEL: define i64 @strong(
; CHECK:       %a = alloca i64, align 8, !bpf.stack.keep
define i64 @strong(ptr %ctx) section "syscall" {
  %a = alloca i64
  %heap = load ptr, ptr %ctx
  store ptr %a, ptr %heap
  %r = call i64 @bpf_read(ptr %a, i64 0)
  ret i64 %r
}

; An address that is only stored in a local is fine in arena.
; CHECK-LABEL: define i64 @only_stored(
; CHECK:       %v = alloca i64, align 8, !bpf.arena.stack
; CHECK:       %h = alloca ptr, align 8, !bpf.arena.stack
define i64 @only_stored(ptr %ctx) section "syscall" {
  %v = alloca i64
  %h = alloca ptr
  store ptr %v, ptr %h
  %p = load ptr, ptr %h
  %r = load i64, ptr %p
  ret i64 %r
}
