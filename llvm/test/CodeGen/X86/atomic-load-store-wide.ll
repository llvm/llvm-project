; RUN: llc < %s -mcpu=corei7 -march=x86 -verify-machineinstrs | FileCheck %s

; 64-bit load/store on x86-32
; FIXME: The generated code can be substantially improved.

define void @test1(i64* %ptr, i64 %val1) {
; CHECK-LABEL: test1
; CHECK: lock cmpxchg8b
; CHECK-NEXT: jne
  store atomic i64 %val1, i64* %ptr seq_cst, align 8
  ret void
}

define i64 @test2(i64* %ptr) {
; CHECK-LABEL: test2
; CHECK: lock cmpxchg8b
  %val = load atomic i64, i64* %ptr seq_cst, align 8
  ret i64 %val
}
