; RUN: llvm-extract -func foo -S < %s | FileCheck %s
; RUN: llvm-extract -delete -func foo -S < %s | FileCheck --check-prefix=DELETE %s

; Test that linkonce definitions are mapped to weak so that they are not
; dropped.

; CHECK:      @bar = external global i32
; CHECK:      define weak ptr @foo() {
; CHECK-NEXT:  ret ptr @bar
; CHECK-NEXT: }

; DELETE: @bar = weak global i32 42
; DELETE: declare ptr @foo()

@bar = linkonce global i32 42

define linkonce ptr @foo() {
  ret ptr @bar
}

define void @g() {
  call ptr @foo()
  ret void
}
