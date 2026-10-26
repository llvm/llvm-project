; RUN: opt -S -mtriple=bpfel -passes='bpf-arena-kptr,function(bpf-arena-stack)' < %s | FileCheck %s
; RUN: llc -mtriple=bpfel -mcpu=v4 -mattr=+arena < %s | FileCheck --check-prefix=ASM %s

; What gets to a helper or a kfunc stays where the kernel knows about it, also
; through a static function. The type of a parameter says that it wants arena.
; A declaration without section .ksyms is a function in another file.

declare void @fill_mask(ptr, ptr addrspace(1)) section ".ksyms"
declare void @other_file(ptr)

define i64 @local() {
; CHECK-LABEL: @local
; CHECK:       %n = alloca i64, align 8, !bpf.stack.keep
; CHECK:       %m = alloca i64, align 8, !bpf.arena.stack
; CHECK:       %o = alloca i64, align 8, !bpf.arena.stack
  %n = alloca i64
  %m = alloca i64
  %o = alloca i64
  %m1 = addrspacecast ptr %m to ptr addrspace(1)
  call void @fill_mask(ptr %n, ptr addrspace(1) %m1)
  call void @other_file(ptr %o)
  %v = load i64, ptr %m
  ret i64 %v
}

define internal ptr @inner(ptr %k) noinline {
  %v = call ptr inttoptr (i64 1 to ptr)(ptr null, ptr %k)
  ret ptr %v
}

define internal ptr @outer(ptr %k) noinline {
  %v = call ptr @inner(ptr %k)
  ret ptr %v
}

; ASM-LABEL: through:
; ASM-NOT:   r12
; ASM:       r1 = r10
; ASM:       call outer
define ptr @through(ptr %ctx) section "syscall" {
; CHECK-LABEL: @through
; CHECK:       %key = alloca i32, align 4, !bpf.stack.keep
  %key = alloca i32
  store i32 1, ptr %key
  %v = call ptr @outer(ptr %key)
  ret ptr %v
}

; Inline asm gets no cast: what it gets stays. Asm without text is a barrier for
; the compiler and accesses nothing.
define void @asm(ptr %ctx) section "syscall" {
; CHECK-LABEL: @asm
; CHECK:       %a = alloca i64, align 8, !bpf.stack.keep
; CHECK:       %b = alloca i64, align 8, !bpf.arena.stack
  %a = alloca i64
  %b = alloca i64
  call void asm sideeffect "*(u64 *)($0 + 0) = 1", "r,~{memory}"(ptr %a)
  call void asm sideeffect "", "r,~{memory}"(ptr %b)
  ret void
}
