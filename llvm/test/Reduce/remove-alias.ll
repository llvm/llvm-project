; RUN: llvm-reduce --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=CHECK-FINAL --input-file=%t %s

; Test handling of 'alias'.

; CHECK-INTERESTINGNESS: define void @fn3

; CHECK-FINAL: $a1
; CHECK-FINAL: $a2
; CHECK-FINAL: $a3
; CHECK-FINAL: $a4

; CHECK-FINAL: define void @fn1
; CHECK-FINAL: define void @fn2
; CHECK-FINAL: define void @fn3
; CHECK-FINAL: define void @fn4

@"$a1" = alias void (), void ()* @fn1
@"$a2" = alias void (), void ()* @fn2
@"$a3" = alias void (), void ()* @fn3
@"$a4" = alias void (), void ()* @fn4

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
