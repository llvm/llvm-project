; RUN: llc < %s -mtriple=armv7-apple-ios | FileCheck %s

define i64 @test1(i64* %ptr, i64 %val) {
; CHECK: test1
; CHECK: dmb ish
; CHECK: ldrexd r2, r3
; CHECK: adds r0, r2
; CHECK: adc r1, r3
; CHECK: strexd {{[a-z0-9]+}}, r0, r1
; CHECK: cmp
; CHECK: bne
; CHECK: dmb ish
  %r = atomicrmw add i64* %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test2(i64* %ptr, i64 %val) {
; CHECK: test2
; CHECK: dmb ish
; CHECK: ldrexd r2, r3
; CHECK: subs r0, r2
; CHECK: sbc r1, r3
; CHECK: strexd {{[a-z0-9]+}}, r0, r1
; CHECK: cmp
; CHECK: bne
; CHECK: dmb ish
  %r = atomicrmw sub i64* %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test3(i64* %ptr, i64 %val) {
; CHECK: test3
; CHECK: dmb ish
; CHECK: ldrexd r2, r3
; CHECK: and r0, r2
; CHECK: and r1, r3
; CHECK: strexd {{[a-z0-9]+}}, r0, r1
; CHECK: cmp
; CHECK: bne
; CHECK: dmb ish
  %r = atomicrmw and i64* %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test4(i64* %ptr, i64 %val) {
; CHECK: test4
; CHECK: dmb ish
; CHECK: ldrexd r2, r3
; CHECK: orr r0, r2
; CHECK: orr r1, r3
; CHECK: strexd {{[a-z0-9]+}}, r0, r1
; CHECK: cmp
; CHECK: bne
; CHECK: dmb ish
  %r = atomicrmw or i64* %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test5(i64* %ptr, i64 %val) {
; CHECK: test5
; CHECK: dmb ish
; CHECK: ldrexd r2, r3
; CHECK: eor r0, r2
; CHECK: eor r1, r3
; CHECK: strexd {{[a-z0-9]+}}, r0, r1
; CHECK: cmp
; CHECK: bne
; CHECK: dmb ish
  %r = atomicrmw xor i64* %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test6(i64* %ptr, i64 %val) {
; CHECK: test6
; CHECK: dmb ish
; CHECK: ldrexd r2, r3
; CHECK: strexd {{[a-z0-9]+}}, r0, r1
; CHECK: cmp
; CHECK: bne
; CHECK: dmb ish
  %r = atomicrmw xchg i64* %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test7(i64* %ptr, i64 %val1, i64 %val2) {
; CHECK: test7
; CHECK: dmb ish
; CHECK: ldrexd r2, r3
; CHECK: cmp r2
; CHECK: cmpeq r3
; CHECK: bne
; CHECK: strexd {{[a-z0-9]+}}, r0, r1
; CHECK: cmp
; CHECK: bne
; CHECK: dmb ish
  %r = cmpxchg i64* %ptr, i64 %val1, i64 %val2 seq_cst
  ret i64 %r
}

; Compiles down to cmpxchg
; FIXME: Should compile to a single ldrexd
define i64 @test8(i64* %ptr) {
; CHECK: test8
; CHECK: ldrexd r2, r3
; CHECK: cmp r2
; CHECK: cmpeq r3
; CHECK: bne
; CHECK: strexd {{[a-z0-9]+}}, r0, r1
; CHECK: cmp
; CHECK: bne
; CHECK: dmb ish
  %r = load atomic i64* %ptr seq_cst, align 8
  ret i64 %r
}

; Compiles down to atomicrmw xchg; there really isn't any more efficient
; way to write it.
define void @test9(i64* %ptr, i64 %val) {
; CHECK: test9
; CHECK: dmb ish
; CHECK: ldrexd r2, r3
; CHECK: strexd {{[a-z0-9]+}}, r0, r1
; CHECK: cmp
; CHECK: bne
; CHECK: dmb ish
  store atomic i64 %val, i64* %ptr seq_cst, align 8
  ret void
}
