; RUN: llvm-reduce --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
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

@"$a1" = alias void (), void ()* @fn1
@"$a2" = alias void (), void ()* @fn2
@"$a3" = alias void (), void ()* @fn3
@"$a4" = alias void (), void ()* @fn4

@"$a5" = alias i64, bitcast (i32* getelementptr ([ 4 x i32 ], [ 4 x i32 ]* @g1, i32 0, i32 1) to i64*)
@"$a6" = alias i64, bitcast (i32* getelementptr ([ 4 x i32 ], [ 4 x i32 ]* @g2, i32 0, i32 1) to i64*)

@llvm.used = appending global [1 x i8*] [
   i8* bitcast (i64* @"$a5" to i8*)
], section "llvm.metadata"

@llvm.compiler.used = appending global [1 x i8*] [
   i8* bitcast (i64* @"$a6" to i8*)
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
