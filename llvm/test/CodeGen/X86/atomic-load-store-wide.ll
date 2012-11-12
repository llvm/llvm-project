; RUN: llc < %s -march=x86 | FileCheck %s

; 64-bit load/store on x86-32
; FIXME: The generated code can be substantially improved.

define void @test1(i64* %ptr, i64 %val1) {
; CHECK: test1
; CHECK: cmpxchg8b
; CHECK-NEXT: jne
  store atomic i64 %val1, i64* %ptr seq_cst, align 8
  ret void
}

define i64 @test2(i64* %ptr) {
; CHECK: test2
; CHECK: cmpxchg8b
  %val = load atomic i64* %ptr seq_cst, align 8
  ret i64 %val
}
