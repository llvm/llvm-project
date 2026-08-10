; RUN: llc -mtriple=aarch64 < %s | FileCheck %s


define i64 @test_readcyclecounter() nounwind {
  ; CHECK-LABEL:   test_readcyclecounter:
  ; CHECK:         // %bb.0:
  ; CHECK-NEXT:    mrs x0, CNTVCT_EL0
  ; CHECK-NEXT:    ret
  %tmp0 = call i64 @llvm.readcyclecounter()
  ret i64 %tmp0
}

declare i64 @llvm.readcyclecounter()
