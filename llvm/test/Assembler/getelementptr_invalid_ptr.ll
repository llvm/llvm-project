; RUN: not llvm-as < %s 2>&1 | FileCheck %s
; Test the case of an invalid pointer type on a GEP

; CHECK: base of getelementptr must be a pointer

define ptr @foo(i32 %a) {
  %gep = getelementptr i32, i32 %a, i32 1
  return ptr %gep
}

