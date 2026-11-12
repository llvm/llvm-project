; RUN: llc -o - %s -relocation-model=pic | FileCheck %s
; Check that we do not get GOT relocations with the x86_64-pc-windows-macho
; triple.
target triple = "x86_64-pc-windows-macho"

@g = common global i32 0, align 4

declare i32 @extbar()

; CHECK-LABEL: bar:
; CHECK: callq _extbar
; CHECK: leaq _extbar(%rip),
; CHECK-NOT: @GOT
define ptr @bar() {
  call i32 @extbar()
  ret ptr @extbar
}

; CHECK-LABEL: foo:
; CHECK: callq _bar
; CHECK: movl _g(%rip),
; CHECK-NOT: @GOT
define i32 @foo() {
  call ptr @bar()
  %gval = load i32, ptr @g, align 4
  ret i32 %gval
}
