; RUN: llc -mattr=+egpr %s -mtriple=x86_64 --relocation-model=pic -enable-tlsdesc -filetype=obj -o %t.o
; RUN: llvm-objdump --no-print-imm-hex -dr %t.o | FileCheck %s --check-prefix=TLSDESC
; RUN: echo '.tbss; .globl b,c,d,e,f,g,h,i,j; b: .zero 4;c: .zero 4;d: .zero 4;e: .zero 4;f: .zero 4;g: .zero 4;h: .zero 4;i: .zero 4;j: .zero 4' | llvm-mc -filetype=obj -triple=x86_64 - -o %t1.o

; RUN: llc -mattr=+egpr %s -mtriple=x86_64 -filetype=obj -o %t.o -x86-enable-apx-for-relocation=true
; RUN: llvm-objdump --no-print-imm-hex -dr %t.o | FileCheck %s --check-prefix=GOTTPOFF_APXRELAX
; RUN: echo '.tbss; .globl b,c,d,e,f,g,h,i,j; b: .zero 4;c: .zero 4;d: .zero 4;e: .zero 4;f: .zero 4;g: .zero 4;h: .zero 4;i: .zero 4;j: .zero 4' | llvm-mc -filetype=obj -triple=x86_64 - -o %t1.o

; RUN: llc -mattr=+egpr %s -mtriple=x86_64 -filetype=obj -o %t.o
; RUN: llvm-objdump --no-print-imm-hex -dr %t.o | FileCheck %s --check-prefix=GOTTPOFF_NOAPXRELAX
; RUN: echo '.tbss; .globl b,c,d,e,f,g,h,i,j; b: .zero 4;c: .zero 4;d: .zero 4;e: .zero 4;f: .zero 4;g: .zero 4;h: .zero 4;i: .zero 4;j: .zero 4' | llvm-mc -filetype=obj -triple=x86_64 - -o %t1.o


; TLSDESC: d5 18 89 c0       movq %rax, %r16
; TLSDESC-NEXT: 48 8d 05 00 00 00 00       leaq (%rip), %rax
; TLSDESC-NEXT: R_X86_64_GOTPC32_TLSDESC j-0x4

; GOTTPOFF_APXRELAX: d5 48 8b 05 00 00 00 00       movq (%rip), %r16
; GOTTPOFF_APXRELAX-NEXT: R_X86_64_CODE_4_GOTTPOFF j-0x4

; GOTTPOFF_NOAPXRELAX: 48 8b 1d 00 00 00 00       movq (%rip), %rbx
; GOTTPOFF_NOAPXRELAX-NEXT: R_X86_64_GOTTPOFF j-0x4

@a = thread_local global i32 0, align 4
@b = external thread_local global i32, align 4
@c = external thread_local global i32, align 4
@d = external thread_local global i32, align 4
@e = external thread_local global i32, align 4
@f = external thread_local global i32, align 4
@g = external thread_local global i32, align 4
@h = external thread_local global i32, align 4
@i = external thread_local global i32, align 4
@j = external thread_local global i32, align 4

define i32 @f2() nounwind {
  %1 = tail call ptr @llvm.threadlocal.address.p0(ptr @a)
  %2 = tail call ptr @llvm.threadlocal.address.p0(ptr @b)
  %3 = tail call ptr @llvm.threadlocal.address.p0(ptr @c)
  %4 = tail call ptr @llvm.threadlocal.address.p0(ptr @d)
  %5 = tail call ptr @llvm.threadlocal.address.p0(ptr @e)
  %6 = tail call ptr @llvm.threadlocal.address.p0(ptr @f)
  %7 = tail call ptr @llvm.threadlocal.address.p0(ptr @g)
  %8 = tail call ptr @llvm.threadlocal.address.p0(ptr @h)
  %9 = tail call ptr @llvm.threadlocal.address.p0(ptr @i)
  %10 = tail call ptr @llvm.threadlocal.address.p0(ptr @j)

  %11 = load i32, ptr %1
  %12 = load i32, ptr %2
  %13 = load i32, ptr %3
  %14 = load i32, ptr %4
  %15 = load i32, ptr %5
  %16 = load i32, ptr %6
  %17 = load i32, ptr %7
  %18 = load i32, ptr %8
  %19 = load i32, ptr %9
  %20 = load i32, ptr %10

  %21 = add i32 %11, %12
  %22 = add i32 %13, %14
  %23 = add i32 %15, %16
  %24 = add i32 %17, %18
  %25 = add i32 %19, %20

  %26 = add i32 %21, %22
  %27 = add i32 %23, %24
  %28 = add i32 %26, %27
  %29 = add i32 %25, %28

  ret i32 %29
}
