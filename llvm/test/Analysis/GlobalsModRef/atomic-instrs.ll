; RUN: opt < %s -aa-pipeline=globals-aa -passes='require<globals-aa>,gvn' -S | FileCheck %s

@X = internal global i32 4

define i32 @test_cmpxchg(i32* %P) {
; CHECK-LABEL: @test_cmpxchg
; CHECK-NEXT: store i32 12, i32* @X
; CHECK-NEXT: call void @modrefX_cmpxchg()
; CHECK-NEXT: %V = load i32, i32* @X
; CHECK-NEXT: ret i32 %V
  store i32 12, i32* @X
  call void @modrefX_cmpxchg()
  %V = load i32, i32* @X
  ret i32 %V
}

define void @modrefX_cmpxchg() {
  %1 = cmpxchg i32* @X, i32 0, i32 -1 monotonic monotonic
  ret void
}

define i32 @test_atomicrmw(i32* %P) {
; CHECK-LABEL: @test_atomicrmw
; CHECK-NEXT: store i32 12, i32* @X
; CHECK-NEXT: call void @modrefXatomicrmw()
; CHECK-NEXT: %V = load i32, i32* @X
; CHECK-NEXT: ret i32 %V
  store i32 12, i32* @X
  call void @modrefXatomicrmw()
  %V = load i32, i32* @X
  ret i32 %V
}

define void @modrefXatomicrmw() {
  %1 = atomicrmw add i32* @X, i32 1 acquire
  ret void
}
