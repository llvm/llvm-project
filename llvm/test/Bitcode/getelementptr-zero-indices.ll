; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; CHECK: %g = getelementptr i8, ptr %p

define i8* @ptr(i8* %p) {
  %g = getelementptr i8, i8* %p
  ret i8* %p
}
