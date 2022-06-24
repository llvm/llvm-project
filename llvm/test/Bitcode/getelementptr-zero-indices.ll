; RUN: llvm-as --opaque-pointers=0 < %s | llvm-dis --opaque-pointers=0 | FileCheck %s

; CHECK: %g = getelementptr i8, i8* %p

define i8* @ptr(i8* %p) {
  %g = getelementptr i8, i8* %p
  ret i8* %p
}
