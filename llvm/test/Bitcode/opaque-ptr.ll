; RUN: llvm-as --opaque-pointers < %s | not llvm-dis --opaque-pointers=0 2>&1 | FileCheck %s

; CHECK: error: Opaque pointers are only supported in -opaque-pointers mode

@g = external global i16

define void @f(i32* %p) {
  %a = alloca i17
  ret void
}
