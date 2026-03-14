; RUN: llvm-extract -func foo -S < %s | FileCheck %s
; RUN: llvm-extract -delete -func foo -S < %s | FileCheck --check-prefix=DELETE %s

; Test that we don't convert weak_odr to external definitions.

; CHECK:      @bar = external global i32
; CHECK:      define weak_odr ptr @foo() {
; CHECK-NEXT:  ret ptr @bar
; CHECK-NEXT: }

; DELETE: @bar = weak_odr global i32 42
; DELETE: declare ptr @foo()

@bar = weak_odr global i32 42

define weak_odr ptr  @foo() {
  ret ptr @bar
}

define void @g() {
  %c = call ptr @foo()
  ret void
}
