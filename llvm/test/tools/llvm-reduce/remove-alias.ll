; RUN: llvm-reduce --abort-on-invalid-reduction --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=CHECK-FINAL --input-file=%t %s

; Test handling of 'alias'.

; CHECK-INTERESTINGNESS: define void @fn3

; CHECK-FINAL-NOT: = {{.*}} global
; CHECK-FINAL-NOT: = alias

; CHECK-FINAL-NOT: @llvm.used
; CHECK-FINAL-NOT: @llvm.compiler.used

; CHECK-FINAL-NOT: define void @fn1
; CHECK-FINAL-NOT: define void @fn2
; CHECK-FINAL: define void @fn3
; CHECK-FINAL-NOT: define void @fn4

@g1 = global [ 4 x i32 ] zeroinitializer
@g2 = global [ 4 x i32 ] zeroinitializer

@"$a1" = alias void (), ptr @fn1
@"$a2" = alias void (), ptr @fn2
@"$a3" = alias void (), ptr @fn3
@"$a4" = alias void (), ptr @fn4

@"$a5" = alias i64, getelementptr ([ 4 x i32 ], ptr @g1, i32 0, i32 1)
@"$a6" = alias i64, getelementptr ([ 4 x i32 ], ptr @g2, i32 0, i32 1)

@llvm.used = appending global [1 x ptr] [
   ptr @"$a5"
], section "llvm.metadata"

@llvm.compiler.used = appending global [1 x ptr] [
   ptr @"$a6"
], section "llvm.metadata"

define void @fn1() {
  ret void
}

define void @fn2() {
  ret void
}

define void @fn3() {
  ret void
}

define void @fn4() {
  ret void
}
