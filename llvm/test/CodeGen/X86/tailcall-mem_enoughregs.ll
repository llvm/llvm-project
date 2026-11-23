; RUN: llc < %s -mtriple=i686-linux-gnu   | FileCheck %s --check-prefix=CHECK --check-prefix=LIN32
; RUN: llc < %s -mtriple=x86_64-linux-gnu | FileCheck %s --check-prefix=CHECK --check-prefix=LIN64
; RUN: llc < %s -mtriple=x86_64-pc-win32  | FileCheck %s --check-prefix=CHECK --check-prefix=WIN64

; Check that we only fold the address computation (load) into a tail call
; when we're sure there is enough volatile registers available.

@globl = global ptr null

; CHECK-LABEL: test0:
define i32 @test0(ptr %a, ptr %b) {
entry:
  %func = load ptr, ptr %a
  %call = tail call i32 %func()
  ret i32 %call

; Call address load gets folded into the tail call.
; LIN32: jmpl *(%
; LIN64: jmpq *(%
; WIN64: jmpq *(%
}

; CHECK-LABEL: test1:
define i32 @test1(ptr %a, ptr %b) {
entry:
  %func = load ptr, ptr %a
  %call = tail call i32 %func(i32 inreg 1)
  ret i32 %call

; Call address load gets folded into the tail call.
; LIN32: jmpl *(%
; LIN64: jmpq *(%
; WIN64: jmpq *(%
}

; CHECK-LABEL: test2:
define i32 @test2(ptr %a, ptr %b) {
entry:
  %func = load ptr, ptr %a
  %call = tail call i32 %func(i32 inreg 1, i32 inreg 2)
  ret i32 %call

; On 32-bit we're not sure there is enough register to fold the load.
; LIN32: jmpl *%
; LIN64: jmpq *(%
; WIN64: jmpq *(%
}

; CHECK-LABEL: test2_globl:
define i32 @test2_globl(ptr %a, ptr %b) {
entry:
  %func = load ptr, ptr @globl
  %call = tail call i32 %func(i32 inreg 1, i32 inreg 2)
  ret i32 %call

; .. but if the load is from a global, we can fold it.
; LIN32: jmpl *globl
; LIN64: jmpq *(%
; WIN64: jmpq *globl(%rip)
}

; CHECK-LABEL: test2_stack:
define i32 @test2_stack(ptr %func, ptr %b) {
entry:
  %call = tail call i32 %func(i32 inreg 1, i32 inreg 2)
  ret i32 %call

; and if the load is from the stack (on 32-bit, %func is passed on the stack):
; LIN32: jmpl *4(%esp)
}

define i32 @test6(ptr %a, ptr %b) {
entry:
  %func = load ptr, ptr %a
  %call = tail call i32 %func(i32 inreg 1, i32 inreg 2, i32 inreg 3, i32 inreg 4, i32 inreg 5, i32 inreg 6)
  ret i32 %call

; LIN64: jmpq *(%

; I wasn't able to pass more than 4 arguments in registers on Win64.
; WIN64: callq *(
}
