; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; CHECK-LABEL: test_readcyclecounter
; CHECK: r1:0 = c15:14
define i64 @test_readcyclecounter() nounwind {
  %t0 = call i64 @llvm.readcyclecounter()
  ret i64 %t0
}

;; Make sure readcyclecounter calls are not optmized out and
;; not moved across function calls.

; CHECK-LABEL: test_readcyclecounter2
; CHECK: [[R1:r[0-9:]+]] = c15:14
; CHECK: call fun
; CHECK: [[R2:r[0-9:]+]] = c15:14
; CHECK: [[R3:r[0-9:]+]] = sub([[R2]],[[R1]])
define i64 @test_readcyclecounter2() {
  %1 = tail call i64 @llvm.readcyclecounter()
  tail call void @fun()
  %2 = tail call i64 @llvm.readcyclecounter()
  %sub = sub i64 %2, %1
  ret i64 %sub
}

declare i64 @llvm.readcyclecounter()
declare void @fun()
