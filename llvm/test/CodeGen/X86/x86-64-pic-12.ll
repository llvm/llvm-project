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
define i8* @bar() {
  call i32 @extbar()
  ret i8* bitcast (i32 ()* @extbar to i8*)
}

; CHECK-LABEL: foo:
; CHECK: callq _bar
; CHECK: movl _g(%rip),
; CHECK-NOT: @GOT
define i32 @foo() {
  call i8* @bar()
  %gval = load i32, i32* @g, align 4
  ret i32 %gval
}
