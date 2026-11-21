; RUN: llc -mtriple=bpfel -mcpu=v4 -mattr=+arena < %s | FileCheck %s

; The module is one file of a program. What another file defines can be in
; arena. The kernel makes a move of the cast if it's not.

@var = external global i32
@ksym = external global i32, section ".ksyms"
declare ptr @other_file()

; CHECK-LABEL: variable:
; CHECK:       [[K:r[0-9]]] = ksym ll
; CHECK-NEXT:  = *(u32 *)([[K]] + 0)
; CHECK-NEXT:  [[V:r[0-9]]] = var ll
; CHECK-NEXT:  [[V]] = addr_space_cast([[V]], 0, 1)
; CHECK-NEXT:  = *(u32 *)([[V]] + 0)
define i32 @variable(ptr %ctx) section "syscall" {
  %a = load i32, ptr @var
  %b = load i32, ptr @ksym
  %r = add i32 %a, %b
  ret i32 %r
}

; CHECK-LABEL: result:
; CHECK:       call other_file
; CHECK-NEXT:  r1 = addr_space_cast(r0, 0, 1)
define i32 @result(ptr %ctx) section "syscall" {
  %p = call ptr @other_file()
  %v = load i32, ptr %p
  ret i32 %v
}
