; RUN: llc -march=hexagon < %s | FileCheck %s

; CHECK-LABEL: test_readcyclecounter
; CHECK: r1:0 = c15:14
define i64 @test_readcyclecounter() nounwind {
  %t0 = call i64 @llvm.readcyclecounter()
  ret i64 %t0
}

declare i64 @llvm.readcyclecounter()
