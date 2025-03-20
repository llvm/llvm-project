; RUN: llc < %s -mtriple=sparc -mattr=fix-tn0011,hasleoncasa | FileCheck %s

; CHECK: .p2align 4
; CHECK-NEXT: casa
define i32 @test_casarr(i32* %p, i32 %v) {
entry:
  %0 = atomicrmw nand i32* %p, i32 %v seq_cst
  ret i32 %0
}

; CHECK: .p2align 4
; CHECK-NEXT: swap
define i32 @test_swaprr(i32* %p, i32 %v) {
entry:
  %0 = atomicrmw xchg i32* %p, i32 %v seq_cst
  ret i32 %0
}

; CHECK: .p2align 4
; CHECK-NEXT: swap
define i32 @test_swapri(i32* %p, i32 %v) {
entry:
  %1 = getelementptr inbounds i32, ptr %p, i32 1
  %2 = atomicrmw xchg ptr %1, i32 %v seq_cst, align 4
  ret i32 %2
}
