; RUN: llc < %s -mattr=-bmi -mtriple=x86_64-linux | FileCheck %s -check-prefix=X86-64
; RUN: llc < %s -mattr=-bmi -mtriple=x86_64-linux-gnux32 | FileCheck %s -check-prefix=X86-64
; RUN: llc < %s -mattr=-bmi -mtriple=x86_64-win32 | FileCheck %s -check-prefix=WIN64
; RUN: llc < %s -mattr=-bmi -march=x86    | FileCheck %s -check-prefix=X86-32

; Use h registers. On x86-64, codegen doesn't support general allocation
; of h registers yet, due to x86 encoding complications.

define void @bar64(i64 inreg %x, i8* inreg %p) nounwind {
; X86-64-LABEL: bar64:
; X86-64: shrq $8, %rdi
; X86-64: incb %dil

; See FIXME: on regclass GR8.
; It could be optimally transformed like; incb %ch; movb %ch, (%rdx)
; WIN64-LABEL:  bar64:
; WIN64:  shrq $8, %rcx
; WIN64:  incb %cl

; X86-32-LABEL: bar64:
; X86-32: incb %ah
  %t0 = lshr i64 %x, 8
  %t1 = trunc i64 %t0 to i8
  %t2 = add i8 %t1, 1
  store i8 %t2, i8* %p
  ret void
}

define void @bar32(i32 inreg %x, i8* inreg %p) nounwind {
; X86-64-LABEL: bar32:
; X86-64: shrl $8, %edi
; X86-64: incb %dil

; WIN64-LABEL:  bar32:
; WIN64:  shrl $8, %ecx
; WIN64:  incb %cl

; X86-32-LABEL: bar32:
; X86-32: incb %ah
  %t0 = lshr i32 %x, 8
  %t1 = trunc i32 %t0 to i8
  %t2 = add i8 %t1, 1
  store i8 %t2, i8* %p
  ret void
}

define void @bar16(i16 inreg %x, i8* inreg %p) nounwind {
; X86-64-LABEL: bar16:
; X86-64: shrl $8, %edi
; X86-64: incb %dil

; WIN64-LABEL:  bar16:
; WIN64:  shrl $8, %ecx
; WIN64:  incb %cl

; X86-32-LABEL: bar16:
; X86-32: incb %ah
  %t0 = lshr i16 %x, 8
  %t1 = trunc i16 %t0 to i8
  %t2 = add i8 %t1, 1
  store i8 %t2, i8* %p
  ret void
}

define i64 @qux64(i64 inreg %x) nounwind {
; X86-64-LABEL: qux64:
; X86-64: movq %rdi, %rax
; X86-64: movzbl %ah, %eax

; WIN64-LABEL:  qux64:
; WIN64:  movzbl %ch, %eax

; X86-32-LABEL: qux64:
; X86-32: movzbl %ah, %eax
  %t0 = lshr i64 %x, 8
  %t1 = and i64 %t0, 255
  ret i64 %t1
}

define i32 @qux32(i32 inreg %x) nounwind {
; X86-64-LABEL: qux32:
; X86-64: movl %edi, %eax
; X86-64: movzbl %ah, %eax

; WIN64-LABEL:  qux32:
; WIN64:  movzbl %ch, %eax

; X86-32-LABEL: qux32:
; X86-32: movzbl %ah, %eax
  %t0 = lshr i32 %x, 8
  %t1 = and i32 %t0, 255
  ret i32 %t1
}

define i16 @qux16(i16 inreg %x) nounwind {
; X86-64-LABEL: qux16:
; X86-64: movl %edi, %eax
; X86-64: movzbl %ah, %eax

; WIN64-LABEL:  qux16:
; WIN64:  movzbl %ch, %eax

; X86-32-LABEL: qux16:
; X86-32: movzbl %ah, %eax
  %t0 = lshr i16 %x, 8
  ret i16 %t0
}
