; RUN: llvm-as %S/Inputs/2003-05-31-LinkerRename.ll -o %t.1.bc
; RUN: llvm-as  %s -o %t.2.bc
; RUN: llvm-link %t.1.bc %t.2.bc -S | FileCheck %s

; CHECK: @bar = global i32 ()* @foo2

; CHECK:      define internal i32 @foo2() {
; CHECK-NEXT:   ret i32 7
; CHECK-NEXT: }

; CHECK: declare i32 @foo()

; CHECK:      define i32 @test() {
; CHECK-NEXT:   %X = call i32 @foo()
; CHECK-NEXT:   ret i32 %X
; CHECK-NEXT: }

declare i32 @foo()

define i32 @test() {
  %X = call i32 @foo()
  ret i32 %X
}
