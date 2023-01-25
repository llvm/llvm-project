; RUN: llvm-link -opaque-pointers=0 -S %s %p/Inputs/type-unique-name.ll | FileCheck %s

; Test that we keep the type name
; CHECK: %abc = type { i8 }

%abc = type opaque

declare %abc* @f()

define %abc* @g() {
  %x = call %abc* @f()
  ret %abc* %x
}
