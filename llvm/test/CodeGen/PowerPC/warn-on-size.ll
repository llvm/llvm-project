; RUN: llc -mtriple=powerpc64-unknown-unknown < %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=powerpc64le-unknown-unknown < %s 2>&1 | FileCheck %s

; CHECK-NOT: warning
; CHECK-Label: large_stack
define i8* @large_stack() {
  %s = alloca [281474976710656 x i8], align 1
  %e = getelementptr i8, i8* %s, i64 0
  ret i8* %e
}

; CHECK: warning: {{.*}} stack frame size {{.*}} exceeds limit (4294967295) in function 'warn_on_large_stack'
; CHECK-Label: warn_on_large_stack
define i8* @warn_on_large_stack() "warn-stack-size"="4294967295" {
  %s = alloca [281474976710656 x i8], align 1
  %e = getelementptr i8, i8* %s, i64 0
  ret i8* %e
}
