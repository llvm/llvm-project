; RUN: opt < %s -passes='asan-function-pipeline' -S -o %t.ll
; RUN: FileCheck %s < %t.ll
; RUN: llc < %t.ll | FileCheck %s --check-prefix=ASM

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-S32"
target triple = "i386-pc-windows-msvc"

define void @MyCPUID(i32 %fxn, i32* %out) sanitize_address {
  %fxn.ptr = alloca i32
  %a.ptr = alloca i32
  %b.ptr = alloca i32
  %c.ptr = alloca i32
  %d.ptr = alloca i32
  store i32 %fxn, i32* %fxn.ptr
  call void asm sideeffect inteldialect "xchg ebx, esi\0A\09mov eax, dword ptr $4\0A\09cpuid\0A\09mov dword ptr $0, eax\0A\09mov dword ptr $1, ebx\0A\09mov dword ptr $2, ecx\0A\09mov dword ptr $3, edx\0A\09xchg ebx, esi", "=*m,=*m,=*m,=*m,*m,~{eax},~{ebx},~{ecx},~{edx},~{esi},~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) %a.ptr, i32* elementtype(i32) %b.ptr, i32* elementtype(i32) %c.ptr, i32* elementtype(i32) %d.ptr, i32* elementtype(i32) %fxn.ptr)

  %a = load i32, i32* %a.ptr
  %a.out = getelementptr inbounds i32, i32* %out, i32 0
  store i32 %a, i32* %a.out

  %b = load i32, i32* %b.ptr
  %b.out = getelementptr inbounds i32, i32* %out, i32 1
  store i32 %b, i32* %b.out

  %c = load i32, i32* %c.ptr
  %c.out = getelementptr inbounds i32, i32* %out, i32 2
  store i32 %c, i32* %c.out

  %d = load i32, i32* %d.ptr
  %d.out = getelementptr inbounds i32, i32* %out, i32 3
  store i32 %d, i32* %d.out

  ret void
}

; We used to introduce stack mallocs for UAR detection, but that makes LLVM run
; out of registers on 32-bit platforms. Therefore, we don't do stack malloc on
; such functions.

; CHECK-LABEL: define void @MyCPUID(i32 %fxn, i32* %out)
; CHECK: %MyAlloca = alloca [96 x i8], align 32
; CHECK-NOT: call {{.*}} @__asan_stack_malloc

; The code generator should recognize that all operands are just stack memory.
; This is important with MS inline asm where operand lists are implicit and all
; local variables can be referenced freely.

; ASM-LABEL: MyCPUID:
; ASM:      cpuid
; ASM-NEXT: movl    %eax, {{[0-9]+}}(%esp)
; ASM-NEXT: movl    %ebx, {{[0-9]+}}(%esp)
; ASM-NEXT: movl    %ecx, {{[0-9]+}}(%esp)
; ASM-NEXT: movl    %edx, {{[0-9]+}}(%esp)
