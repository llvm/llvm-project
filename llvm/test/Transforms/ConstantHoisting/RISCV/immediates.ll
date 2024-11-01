; RUN: opt -mtriple=riscv32-unknown-elf -S -passes=consthoist < %s | FileCheck %s
; RUN: opt -mtriple=riscv64-unknown-elf -S -passes=consthoist < %s | FileCheck %s

; Check that we don't hoist immediates with small values.
define i64 @test1(i64 %a) nounwind {
; CHECK-LABEL: test1
; CHECK-NOT: %const = bitcast i64 2 to i64
  %1 = mul i64 %a, 2
  %2 = add i64 %1, 2
  ret i64 %2
}

; Check that we don't hoist immediates with small values.
define i64 @test2(i64 %a) nounwind {
; CHECK-LABEL: test2
; CHECK-NOT: %const = bitcast i64 2047 to i64
  %1 = mul i64 %a, 2047
  %2 = add i64 %1, 2047
  ret i64 %2
}

; Check that we hoist immediates with large values.
define i64 @test3(i64 %a) nounwind {
; CHECK-LABEL: test3
; CHECK: %const = bitcast i64 32767 to i64
  %1 = mul i64 %a, 32767
  %2 = add i64 %1, 32767
  ret i64 %2
}

; Check that we hoist immediates with very large values.
define i128 @test4(i128 %a) nounwind {
; CHECK-LABEL: test4
; CHECK: %const = bitcast i128 12297829382473034410122878 to i128
  %1 = add i128 %a, 12297829382473034410122878
  %2 = add i128 %1, 12297829382473034410122878
  ret i128 %2
}

; Check that we hoist zext.h without Zbb.
define i32 @test5(i32 %a) nounwind {
; CHECK-LABEL: test5
; CHECK: %const = bitcast i32 65535 to i32
  %1 = and i32 %a, 65535
  %2 = and i32 %1, 65535
  ret i32 %2
}

; Check that we don't hoist zext.h with 65535 with Zbb.
define i32 @test6(i32 %a) nounwind "target-features"="+zbb" {
; CHECK-LABEL: test6
; CHECK: and i32 %a, 65535
  %1 = and i32 %a, 65535
  %2 = and i32 %1, 65535
  ret i32 %2
}

; Check that we hoist zext.w without Zba.
define i64 @test7(i64 %a) nounwind {
; CHECK-LABEL: test7
; CHECK: %const = bitcast i64 4294967295 to i64
  %1 = and i64 %a, 4294967295
  %2 = and i64 %1, 4294967295
  ret i64 %2
}

; Check that we don't hoist zext.w with Zba.
define i64 @test8(i64 %a) nounwind "target-features"="+zba" {
; CHECK-LABEL: test8
; CHECK: and i64 %a, 4294967295
  %1 = and i64 %a, 4294967295
  %2 = and i64 %1, 4294967295
  ret i64 %2
}

; Check that we don't hoist mul with negated power of 2.
define i64 @test9(i64 %a) nounwind {
; CHECK-LABEL: test9
; CHECK: mul i64 %a, -4294967296
  %1 = mul i64 %a, -4294967296
  %2 = mul i64 %1, -4294967296
  ret i64 %2
}

define i32 @test10(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: @test10(
; CHECK: shl i32 %a, 8
; CHECK: and i32 %1, 65280
; CHECK: shl i32 %b, 8
; CHECK: and i32 %3, 65280
  %1 = shl i32 %a, 8
  %2 = and i32 %1, 65280
  %3 = shl i32 %b, 8
  %4 = and i32 %3, 65280
  %5 = mul i32 %2, %4
  ret i32 %5
}

; bseti
define i64 @test11(i64 %a) nounwind "target-features"="+zbs" {
; CHECK-LABEL: test11
; CHECK: or i64 %a, 8589934592
  %1 = or i64 %a, 8589934592 ; 1 << 33
  %2 = or i64 %1, 8589934592 ; 1 << 33
  ret i64 %2
}

; binvi
define i64 @test12(i64 %a) nounwind "target-features"="+zbs" {
; CHECK-LABEL: test12
; CHECK: xor i64 %a, -9223372036854775808
  %1 = xor i64 %a, -9223372036854775808 ; 1 << 63
  %2 = xor i64 %1, -9223372036854775808 ; 1 << 63
  ret i64 %2
}

; bclri
define i64 @test13(i64 %a) nounwind "target-features"="+zbs" {
; CHECK-LABEL: test13
; CHECK: and i64 %a, -281474976710657
  %1 = and i64 %a, -281474976710657 ; ~(1 << 48)
  %2 = and i64 %1, -281474976710657 ; ~(1 << 48)
  ret i64 %2
}

; Check that we don't hoist mul by a power of 2.
define i64 @test14(i64 %a) nounwind {
; CHECK-LABEL: test14
; CHECK: mul i64 %a, 2048
  %1 = mul i64 %a, 2048
  %2 = mul i64 %1, 2048
  ret i64 %2
}

