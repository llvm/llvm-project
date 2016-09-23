; RUN: llc < %s -O1 -mtriple=x86_64-pc-win32 | FileCheck %s -check-prefix=ASM
; RUN: llc < %s -O1 -mtriple=x86_64-pc-win32 -filetype=obj -o %t
; RUN: llvm-readobj -unwind %t | FileCheck %s -check-prefix=READOBJ

declare void @g(i32)

define i32 @not_leaf(i32) uwtable {
entry:
  call void @g(i32 42)
  ret i32 42

; ASM-LABEL: not_leaf:
; ASM: .seh

; READOBJ: RuntimeFunction {
; READOBJ-NEXT: StartAddress: not_leaf
; READOBJ-NEXT: EndAddress: not_leaf
}

define void @leaf_func(i32) uwtable {
entry:
  tail call void @g(i32 42)
  ret void

; A Win64 "leaf" function gets no .seh directives in the asm.
; ASM-LABEL: leaf_func:
; ASM-NOT: .seh

; and no unwind info in the object file.
; READOBJ-NOT: leaf_func
}
