; RUN: llvm-link %S/Inputs/2003-05-31-LinkerRename.ll %s -S | FileCheck %s

; CHECK: @bar = global i32 ()* @foo.2

; CHECK:      define internal i32 @foo.2() {
; CHECK-NEXT:   ret i32 7
; CHECK-NEXT: }

; CHECK:      define i32 @test() {
; CHECK-NEXT:   %X = call i32 @foo()
; CHECK-NEXT:   ret i32 %X
; CHECK-NEXT: }

; CHECK: declare i32 @foo()

declare i32 @foo()

define i32 @test() {
  %X = call i32 @foo()
  ret i32 %X
}
