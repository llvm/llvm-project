; This test should cause the TargetMaterializeAlloca to be invoked
; RUN: llc -O0 -fast-isel -fast-isel-abort=1 -verify-machineinstrs -mtriple=arm64-apple-darwin -disable-fp-elim < %s | FileCheck %s

%struct.S1Ty = type { i64 }
%struct.S2Ty = type { %struct.S1Ty, %struct.S1Ty }

define void @takeS1(%struct.S1Ty* %V) nounwind {
entry:
  %V.addr = alloca %struct.S1Ty*, align 8
  store %struct.S1Ty* %V, %struct.S1Ty** %V.addr, align 8
  ret void
}

define void @main() nounwind {
entry:
; CHECK: main
; CHECK: add x29, sp, #16
; CHECK: mov [[REG:x[0-9]+]], sp
; CHECK-NEXT: add x0, [[REG]], #8
  %E = alloca %struct.S2Ty, align 4
  %B = getelementptr inbounds %struct.S2Ty, %struct.S2Ty* %E, i32 0, i32 1
  call void @takeS1(%struct.S1Ty* %B)
  ret void
}
