
; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; CHECK-LABEL: test_readsteadycounter
; CHECK: r1:0 = c31:30
define i64 @test_readsteadycounter() nounwind {
  %t0 = call i64 @llvm.readsteadycounter()
  ret i64 %t0
}

declare i64 @llvm.readsteadycounter()
