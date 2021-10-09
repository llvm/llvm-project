; RUN: llvm-reduce --delta-passes=arguments --test FileCheck --test-arg --check-prefixes=CHECK-ALL,CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: cat %t | FileCheck --check-prefixes=CHECK-ALL,CHECK-FINAL %s

; CHECK-INTERESTINGNESS: define void @func(
; CHECK-FINAL: define void @func()
define void @func(i1 %arg) {
; CHECK-ALL: bb:
; CHECK-ALL: br label %bb4
bb:
  br label %bb4

; CHECK-ALL: bb4
bb4:
; CHECK-INTERESTINGNESS: callbr void asm
; CHECK-INTERESTINGNESS-SAME: blockaddress
; CHECK-FINAL: callbr void asm sideeffect "", "X"(i8* blockaddress(@func, %bb11))
; CHECK-ALL: to label %bb5 [label %bb11]
  callbr void asm sideeffect "", "X"(i8* blockaddress(@func, %bb11))
          to label %bb5 [label %bb11]

; CHECK-ALL: bb5:
; CHECK-ALL: br label %bb11
bb5:
  br label %bb11

; CHECK-ALL: bb11:
; CHECK-ALL: ret void
bb11:
  ret void
}
