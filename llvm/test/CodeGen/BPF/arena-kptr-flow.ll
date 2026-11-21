; RUN: opt -S -mtriple=bpfel -passes='bpf-arena-kptr,function(bpf-arena-stack)' < %s | FileCheck %s

; How a pointer of the kernel gets to a local, one rule per function.
; Functions that are called indirectly have types of their own, so that one test
; does not reach the functions of another.

declare ptr @bpf_task_get() section ".ksyms"
declare ptr addrspace(1) @bpf_alloc() section ".ksyms"

; An integer from the context that becomes a pointer after sext.
; CHECK-LABEL: define void @sext(
; CHECK:       %s = alloca ptr, align 8, !bpf.stack.keep
define void @sext(ptr %ctx) section "tc" {
  %s = alloca ptr
  %v32 = load i32, ptr %ctx
  %v64 = sext i32 %v32 to i64
  %p = inttoptr i64 %v64 to ptr
  store ptr %p, ptr %s
  ret void
}

; After freeze.
; CHECK-LABEL: define void @freeze(
; CHECK:       %s = alloca ptr, align 8, !bpf.stack.keep
define void @freeze(ptr %ctx) section "tc" {
  %s = alloca ptr
  %v = load i64, ptr %ctx
  %f = freeze i64 %v
  %p = inttoptr i64 %f to ptr
  store ptr %p, ptr %s
  ret void
}

; The address of a field of an object of the kernel.
; CHECK-LABEL: define void @gep(
; CHECK:       %s = alloca ptr, align 8, !bpf.stack.keep
define void @gep(ptr %ctx) section "syscall" {
  %s = alloca ptr
  %t = call ptr @bpf_task_get()
  %f = getelementptr i8, ptr %t, i64 8
  store ptr %f, ptr %s
  ret void
}

; CHECK-LABEL: define void @select_val(
; CHECK:       %s = alloca ptr, align 8, !bpf.stack.keep
define void @select_val(ptr %ctx, i1 %c) section "syscall" {
  %s = alloca ptr
  %t = call ptr @bpf_task_get()
  %v = select i1 %c, ptr %t, ptr null
  store ptr %v, ptr %s
  ret void
}

; CHECK-LABEL: define void @phi_val(
; CHECK:       %s = alloca ptr, align 8, !bpf.stack.keep
define void @phi_val(ptr %ctx, i1 %c) section "syscall" {
entry:
  %s = alloca ptr
  br i1 %c, label %get, label %out
get:
  %t = call ptr @bpf_task_get()
  br label %out
out:
  %v = phi ptr [ %t, %get ], [ null, %entry ]
  store ptr %v, ptr %s
  ret void
}

; The condition of a select is not a value that it yields.
; CHECK-LABEL: define void @select_cond(
; CHECK:       %s = alloca i64, align 8, !bpf.arena.stack
define void @select_cond(ptr %ctx) section "syscall" {
  %s = alloca i64
  %t = call ptr @bpf_task_get()
  %i = ptrtoint ptr %t to i64
  %c = trunc i64 %i to i1
  %v = select i1 %c, i64 1, i64 2
  store i64 %v, ptr %s
  ret void
}

; The address of a local that a helper gets is a pointer to the stack too.
; CHECK-LABEL: define void @passed_address(
; CHECK:       %key = alloca i32, align 4, !bpf.stack.keep
; CHECK:       %env = alloca ptr, align 8, !bpf.stack.keep
define void @passed_address(ptr %ctx) section "syscall" {
  %key = alloca i32
  %env = alloca ptr
  %v = call ptr inttoptr (i64 1 to ptr)(ptr null, ptr %key)
  store ptr %key, ptr %env
  ret void
}

; The type of the kfunc says that the result is in arena.
; CHECK-LABEL: define void @arena_result(
; CHECK:       %s = alloca ptr, align 8, !bpf.arena.stack
define void @arena_result(ptr %ctx) section "syscall" {
  %s = alloca ptr
  %p1 = call ptr addrspace(1) @bpf_alloc()
  %p = addrspacecast ptr addrspace(1) %p1 to ptr
  store ptr %p, ptr %s
  ret void
}

; A parameter of a callback that is not a pointer is a number.
; CHECK-LABEL: define internal i64 @cb(
; CHECK:       %w = alloca i64, align 8, !bpf.arena.stack
define internal i64 @cb(i64 %idx, ptr %data) {
  %w = alloca i64
  store i64 %idx, ptr %w
  ret i64 0
}

define i64 @callback_int(ptr %ctx) section "syscall" {
  %r = call i64 inttoptr (i64 181 to ptr)(i32 1, ptr @cb, ptr null, i64 0)
  ret i64 %r
}

@tbl = internal constant [6 x ptr] [ptr @taken, ptr @taken_asm, ptr @taken_helper, ptr @fill, ptr @get, ptr null]

define internal void @taken(ptr %t) noinline {
  ret void
}

; An indirect call does not get to a function whose address is not taken.
; CHECK-LABEL: define internal void @not_taken(
; CHECK:       %e = alloca ptr, align 8, !bpf.arena.stack
define internal void @not_taken(ptr %t) noinline {
  %e = alloca ptr
  store ptr %t, ptr %e
  ret void
}

define void @call_taken(ptr %ctx) section "syscall" {
  %f = load ptr, ptr @tbl
  %t = call ptr @bpf_task_get()
  call void %f(ptr %t)
  call void @not_taken(ptr null)
  ret void
}

; Inline asm is not an indirect call.
; CHECK-LABEL: define internal void @taken_asm(
; CHECK:       %e = alloca ptr, align 8, !bpf.arena.stack
define internal void @taken_asm(ptr %t, i8 %x) noinline {
  %e = alloca ptr
  store ptr %t, ptr %e
  ret void
}

define void @asm(ptr %ctx) section "syscall" {
  %t = call ptr @bpf_task_get()
  call void asm sideeffect "", "r,r"(ptr %t, i8 0)
  ret void
}

; The call of a helper is not an indirect call.
; CHECK-LABEL: define internal i64 @taken_helper(
; CHECK:       %e = alloca ptr, align 8, !bpf.arena.stack
define internal i64 @taken_helper(ptr %t) noinline {
  %e = alloca ptr
  store ptr %t, ptr %e
  ret i64 0
}

define i64 @helper(ptr %ctx) section "syscall" {
  %r = call i64 inttoptr (i64 35 to ptr)(ptr %ctx)
  ret i64 %r
}

define internal void @fill(ptr %e, i16 %x) noinline {
  %t = call ptr @bpf_task_get()
  store ptr %t, ptr %e
  ret void
}

; A function that is called indirectly stores it.
; CHECK-LABEL: define void @indirect_written(
; CHECK:       %a = alloca ptr, align 8, !bpf.stack.keep
define void @indirect_written(ptr %ctx) section "syscall" {
  %a = alloca ptr
  %f = load ptr, ptr @tbl
  call void %f(ptr %a, i16 0)
  ret void
}

define internal ptr @get() noinline {
  %t = call ptr @bpf_task_get()
  ret ptr %t
}

; A function that is called indirectly returns it.
; CHECK-LABEL: define void @indirect_result(
; CHECK:       %a = alloca ptr, align 8, !bpf.stack.keep
define void @indirect_result(ptr %ctx) section "syscall" {
  %a = alloca ptr
  %f = load ptr, ptr @tbl
  %t = call ptr %f()
  store ptr %t, ptr %a
  ret void
}

define internal void @fill8(ptr %e) noinline {
  %t = call ptr @bpf_task_get()
  %f = getelementptr i8, ptr %e, i64 8
  store ptr %t, ptr %f
  ret void
}

; The offset at which a callee stores it.
; CHECK-LABEL: define void @written_offset(
; CHECK:       %b = alloca ptr, align 8, !bpf.stack.keep
define void @written_offset(ptr %ctx) section "syscall" {
  %a = alloca [16 x i8], align 8
  %b = alloca ptr
  call void @fill8(ptr %a)
  %a8 = getelementptr i8, ptr %a, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %b, ptr %a8, i64 8, i1 false)
  ret void
}

; The offset of the pointer in a copy: %c gets the half of %b that has it.
; CHECK-LABEL: define void @copy_offset(
; CHECK:       %c = alloca ptr, align 8, !bpf.stack.keep
define void @copy_offset(ptr %ctx) section "syscall" {
  %a = alloca [16 x i8], align 8
  %b = alloca [16 x i8], align 8
  %c = alloca ptr
  %t = call ptr @bpf_task_get()
  %a8 = getelementptr i8, ptr %a, i64 8
  store ptr %t, ptr %a8
  call void @llvm.memcpy.p0.p0.i64(ptr %b, ptr %a, i64 16, i1 false)
  %b8 = getelementptr i8, ptr %b, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %c, ptr %b8, i64 8, i1 false)
  ret void
}

; The copy starts in the middle of the source.
; CHECK-LABEL: define void @copy_src_offset(
; CHECK:       %c = alloca ptr, align 8, !bpf.stack.keep
define void @copy_src_offset(ptr %ctx) section "syscall" {
  %a = alloca [16 x i8], align 8
  %b = alloca [16 x i8], align 8
  %c = alloca ptr
  %t = call ptr @bpf_task_get()
  %a8 = getelementptr i8, ptr %a, i64 8
  store ptr %t, ptr %a8
  call void @llvm.memcpy.p0.p0.i64(ptr %b, ptr %a8, i64 8, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %c, ptr %b, i64 8, i1 false)
  ret void
}

; A copy of the part of a local that has no pointer.
; CHECK-LABEL: define void @copy_part(
; CHECK:       %lo = alloca i64, align 8, !bpf.arena.stack
; CHECK:       %hi = alloca i64, align 8, !bpf.arena.stack
define void @copy_part(ptr %ctx) section "syscall" {
  %a = alloca [24 x i8], align 8
  %lo = alloca i64
  %hi = alloca i64
  %t = call ptr @bpf_task_get()
  %a8 = getelementptr i8, ptr %a, i64 8
  store ptr %t, ptr %a8
  call void @llvm.memcpy.p0.p0.i64(ptr %lo, ptr %a, i64 8, i1 false)
  %a16 = getelementptr i8, ptr %a, i64 16
  call void @llvm.memcpy.p0.p0.i64(ptr %hi, ptr %a16, i64 8, i1 false)
  ret void
}

; The place of the pointer in the source is not known.
; CHECK-LABEL: define void @copy_slot_unknown(
; CHECK:       %b = alloca [40 x i8], align 8, !bpf.stack.keep
define void @copy_slot_unknown(ptr %ctx, i64 %i) section "syscall" {
  %a = alloca [40 x i8], align 8
  %b = alloca [40 x i8], align 8
  %t = call ptr @bpf_task_get()
  %p = getelementptr i8, ptr %a, i64 %i
  store ptr %t, ptr %p
  call void @llvm.memcpy.p0.p0.i64(ptr %b, ptr %a, i64 40, i1 false)
  ret void
}

; The start of the copy is not known.
; CHECK-LABEL: define void @copy_src_unknown(
; CHECK:       %b = alloca ptr, align 8, !bpf.stack.keep
define void @copy_src_unknown(ptr %ctx, i64 %i) section "syscall" {
  %a = alloca [40 x i8], align 8
  %b = alloca ptr
  %t = call ptr @bpf_task_get()
  store ptr %t, ptr %a
  %p = getelementptr i8, ptr %a, i64 %i
  call void @llvm.memcpy.p0.p0.i64(ptr %b, ptr %p, i64 8, i1 false)
  ret void
}

; The length of the copy is not known.
; CHECK-LABEL: define void @copy_len_unknown(
; CHECK:       %b = alloca [40 x i8], align 8, !bpf.stack.keep
define void @copy_len_unknown(ptr %ctx, i64 %n) section "syscall" {
  %a = alloca [40 x i8], align 8
  %b = alloca [40 x i8], align 8
  %t = call ptr @bpf_task_get()
  store ptr %t, ptr %a
  call void @llvm.memcpy.p0.p0.i64(ptr %b, ptr %a, i64 %n, i1 false)
  ret void
}

; The mark of an earlier run of bpf-arena-stack goes away.
; CHECK-LABEL: define void @remark(
; CHECK:       %s = alloca ptr, align 8, !bpf.stack.keep ![[#]]{{$}}
define void @remark(ptr %ctx) section "syscall" {
  %s = alloca ptr, !bpf.arena.stack !0
  %t = call ptr @bpf_task_get()
  store ptr %t, ptr %s
  ret void
}

define internal void @va(ptr %a, ...) noinline {
  ret void
}

; More arguments than parameters.
; CHECK-LABEL: define void @vararg(
define void @vararg(ptr %ctx) section "syscall" {
  call void (ptr, ...) @va(ptr %ctx, ptr %ctx)
  ret void
}

!0 = !{}

define internal i64 @as_number() noinline {
  %t = call ptr @bpf_task_get()
  %n = ptrtoint ptr %t to i64
  ret i64 %n
}

; A result that is an integer.
; CHECK-LABEL: define void @int_result(
; CHECK:       %a = alloca i64, align 8, !bpf.stack.keep
define void @int_result(ptr %ctx) section "syscall" {
  %a = alloca i64
  %n = call i64 @as_number()
  store i64 %n, ptr %a
  ret void
}

; The copy of a struct of 8 bytes moves the pointer as an integer.
; CHECK-LABEL: define void @copy_as_int(
; CHECK:       %a = alloca ptr, align 8, !bpf.stack.keep
; CHECK:       %b = alloca ptr, align 8, !bpf.stack.keep
define void @copy_as_int(ptr %ctx) section "syscall" {
  %a = alloca ptr
  %b = alloca ptr
  %t = call ptr @bpf_task_get()
  store ptr %t, ptr %a
  %v = load i64, ptr %a
  store i64 %v, ptr %b
  ret void
}

; An integer in another field of that local is a number.
; CHECK-LABEL: define void @copy_other_field(
; CHECK:       %a = alloca [16 x i8], align 8, !bpf.stack.keep
; CHECK:       %b = alloca i64, align 8, !bpf.arena.stack
define void @copy_other_field(ptr %ctx) section "syscall" {
  %a = alloca [16 x i8], align 8
  %b = alloca i64
  %t = call ptr @bpf_task_get()
  store ptr %t, ptr %a
  %f = getelementptr i8, ptr %a, i64 8
  %v = load i64, ptr %f
  store i64 %v, ptr %b
  ret void
}

define internal { ptr, i64 } @get_pair() noinline {
  %t = call ptr @bpf_task_get()
  %a = insertvalue { ptr, i64 } poison, ptr %t, 0
  %b = insertvalue { ptr, i64 } %a, i64 8, 1
  ret { ptr, i64 } %b
}

; A result that is a pair. The integer in it is a number.
; CHECK-LABEL: define void @pair_result(
; CHECK:       %s = alloca ptr, align 8, !bpf.stack.keep
; CHECK:       %n = alloca i64, align 8, !bpf.arena.stack
define void @pair_result(ptr %ctx) section "syscall" {
  %s = alloca ptr
  %n = alloca i64
  %r = call { ptr, i64 } @get_pair()
  %p = extractvalue { ptr, i64 } %r, 0
  %l = extractvalue { ptr, i64 } %r, 1
  store ptr %p, ptr %s
  store i64 %l, ptr %n
  ret void
}
