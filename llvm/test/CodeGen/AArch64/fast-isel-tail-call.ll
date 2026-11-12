; RUN: llc -fast-isel -pass-remarks-missed=isel -pass-remarks-missed=isel \
; RUN:     -mtriple arm64-- < %s 2> %t | FileCheck %s
; RUN: cat %t | FileCheck %s --check-prefix MISSED

%struct = type { [4 x i32] }

declare %struct @external()

; Check that, when fastisel falls back to SDAG, we don't emit instructions
; that follow a tail-call and would have been dropped by pure SDAGISel.

; Here, the %struct extractvalue should fail FastISel.

; MISSED: FastISel missed:   %tmp1 = extractvalue %struct %tmp0, 0

; CHECK-LABEL: test:
; CHECK: b external
; CHECK-NEXT: .Lfunc_end0:
define i32 @test() nounwind {
  %tmp0 = tail call %struct @external()
  %tmp1 = extractvalue %struct %tmp0, 0
  %tmp2 = extractvalue [4 x i32] %tmp1, 0
  ret i32 %tmp2
}
