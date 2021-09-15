; RUN: llvm-extract -func foo -S < %s | FileCheck %s
; RUN: llvm-extract -delete -func foo -S < %s | FileCheck --check-prefix=DELETE %s

; Test that linkonce definitions are mapped to weak so that they are not
; dropped.

; CHECK:      @bar = external global i32
; CHECK:      define weak i32* @foo() {
; CHECK-NEXT:  ret i32* @bar
; CHECK-NEXT: }

; DELETE: @bar = weak global i32 42
; DELETE: declare i32* @foo()

@bar = linkonce global i32 42

define linkonce i32* @foo() {
  ret i32* @bar
}

define void @g() {
  call i32* @foo()
  ret void
}
