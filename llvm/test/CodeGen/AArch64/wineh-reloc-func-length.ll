; RUN: llc -mtriple=aarch64-pc-windows-msvc -filetype=obj %s -o %t.o
; RUN: llvm-readobj --unwind %t.o | FileCheck %s

; When an alignment directive inside a function (here from inline asm) makes the
; function length impossible to compute as an absolute value at unwind-emission
; time, the AArch64 SEH .xdata function length and epilog offset must be emitted
; as relocations resolved once the layout is final, instead of crashing with
; "Failed to evaluate function length in SEH unwind info".

declare dso_local void @g()

define dso_local i32 @f(i32 %x) {
entry:
  %c = icmp sgt i32 %x, 0
  br i1 %c, label %call, label %cont
call:
  call void @g()
  br label %cont
cont:
  call void asm sideeffect ".p2align 4", ""()
  %r = add i32 %x, 1
  ret i32 %r
}

; CHECK:      Function: f
; CHECK:      ExceptionData {
; CHECK-NEXT:   FunctionLength:
; CHECK:        EpilogueScopes [
; CHECK-NEXT:     EpilogueScope {
; CHECK-NEXT:       StartOffset:
