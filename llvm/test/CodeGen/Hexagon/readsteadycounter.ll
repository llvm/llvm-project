
; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; CHECK-LABEL: test_readsteadycounter
; CHECK: r1:0 = c31:30
define i64 @test_readsteadycounter() nounwind {
  %t0 = call i64 @llvm.readsteadycounter()
  ret i64 %t0
}

;; Make sure readsteadycounter calls are not optmized out and
;; not moved across function calls.

; CHECK-LABEL: test_readsteadycounter2
; CHECK: [[R1:r[0-9:]+]] = c31:30
; CHECK: call fun
; CHECK: [[R2:r[0-9:]+]] = c31:30
; CHECK: [[R3:r[0-9:]+]] = sub([[R2]],[[R1]])
define i64 @test_readsteadycounter2() {
  %1 = tail call i64 @llvm.readsteadycounter()
  tail call void @fun()
  %2 = tail call i64 @llvm.readsteadycounter()
  %sub = sub i64 %2, %1
  ret i64 %sub
}

declare i64 @llvm.readsteadycounter()
declare void @fun()
