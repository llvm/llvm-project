; RUN: llvm-extract -func foo -S < %s | FileCheck %s
; RUN: llvm-extract -delete -func foo -S < %s | FileCheck --check-prefix=DELETE %s

; Test that we don't convert weak_odr to external definitions.

; CHECK:      @bar = external global i32
; CHECK:      define weak_odr i32* @foo() {
; CHECK-NEXT:  ret i32* @bar
; CHECK-NEXT: }

; DELETE: @bar = weak_odr global i32 42
; DELETE: declare i32* @foo()

@bar = weak_odr global i32 42

define weak_odr i32*  @foo() {
  ret i32* @bar
}

define void @g() {
  %c = call i32* @foo()
  ret void
}
