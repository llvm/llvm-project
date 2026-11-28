; RUN: llc -enable-ipra < %s -o - | FileCheck %s

target triple = "x86_64-unknown-unknown"

; Prologue/Epilogue should not skip R13 and R14 when IPRA enabled since it is used.
; CHECK-LABEL: function1:
; CHECK:    pushq   %r14
; CHECK:    pushq   %r13

define internal void @function1() norecurse noinline {
  call void asm sideeffect "movl %r13d, %eax\0A addl %r14d, %eax", "~{eax},~{r13},~{r14}"()
  ret void
}

; r13 and r14 live across call to function1 so it must be saved by callee before use.
; CHECK-LABEL: function2:
; CHECK:    pushq   %r14
; CHECK:    pushq   %r13
define void @function2(i32 %x) noinline {
  call void asm sideeffect "movl %edi, %r13d\0A movl %edi, %r14d", "~{r13},~{r14}"()
  call void @function1()
  call void asm sideeffect "movl %r13d, %eax\0A addl %r14d, %eax", "~{eax},~{r13},~{r14}"()
  ret void
}

define i32 @main() {
  call void @function2(i32 7)
  ret i32 0
}



