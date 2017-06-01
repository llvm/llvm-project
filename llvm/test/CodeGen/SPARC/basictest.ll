; RUN: llc < %s -march=sparc | FileCheck %s

define i32 @test0(i32 %X) {
	%tmp.1 = add i32 %X, 1
	ret i32 %tmp.1
; CHECK-LABEL: test0:
; CHECK: add %o0, 1, %o0
}


;; xnor tests.
define i32 @test1(i32 %X, i32 %Y) {
        %A = xor i32 %X, %Y
        %B = xor i32 %A, -1
        ret i32 %B
; CHECK-LABEL: test1:
; CHECK: xnor %o0, %o1, %o0
}

define i32 @test2(i32 %X, i32 %Y) {
        %A = xor i32 %X, -1
        %B = xor i32 %A, %Y
        ret i32 %B
; CHECK-LABEL: test2:
; CHECK: xnor %o0, %o1, %o0
}

; CHECK-LABEL: store_zero:
; CHECK: st   %g0, [%o0]
; CHECK: st   %g0, [%o1+4]
define i32 @store_zero(i32* %a, i32* %b) {
entry:
  store i32 0, i32* %a, align 4
  %0 = getelementptr inbounds i32, i32* %b, i32 1
  store i32 0, i32* %0, align 4
  ret i32 0
}

; CHECK-LABEL: signed_divide:
; CHECK: sra %o0, 31, %o2
; CHECK: wr %g0, %o2, %y
; CHECK: sdiv %o0, %o1, %o0
define i32 @signed_divide(i32 %a, i32 %b) {
  %r = sdiv i32 %a, %b
  ret i32 %r
}

; CHECK-LABEL: unsigned_divide:
; CHECK: wr %g0, %g0, %y
; CHECK: udiv %o0, %o1, %o0
define i32 @unsigned_divide(i32 %a, i32 %b) {
  %r = udiv i32 %a, %b
  ret i32 %r
}

; CHECK-LABEL: multiply_32x32:
; CHECK: smul %o0, %o1, %o0
define i32 @multiply_32x32(i32 %a, i32 %b) {
  %r = mul i32 %a, %b
  ret i32 %r
}

; CHECK-LABEL: signed_multiply_32x32_64:
; CHECK: smul %o0, %o1, %o1
; CHECK: rd %y, %o0
define i64 @signed_multiply_32x32_64(i32 %a, i32 %b) {
  %xa = sext i32 %a to i64
  %xb = sext i32 %b to i64
  %r = mul i64 %xa, %xb
  ret i64 %r
}

; CHECK-LABEL: unsigned_multiply_32x32_64:
; CHECK: umul %o0, %o1, %o1
; CHECK: rd %y, %o0
; CHECK: retl
; CHECK: nop
define i64 @unsigned_multiply_32x32_64(i32 %a, i32 %b) {
  %xa = zext i32 %a to i64
  %xb = zext i32 %b to i64
  %r = mul i64 %xa, %xb
  ret i64 %r
}

; CHECK-LABEL: load_store_64bit:
; CHECK: ldd [%o0], %o2
; CHECK: addcc %o3, 3, %o5
; CHECK: addxcc %o2, 0, %o4
; CHECK: retl
; CHECK: std %o4, [%o1]
define void @load_store_64bit(i64* %x, i64* %y) {
entry:
  %0 = load i64, i64* %x
  %add = add nsw i64 %0, 3
  store i64 %add, i64* %y
  ret void
}
