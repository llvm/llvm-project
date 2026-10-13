; RUN: llc -mtriple=hexagon -mcpu=hexagonv68 -O3 < %s | FileCheck %s

; Verify that bitcasts between f64 and the integer vector types that share the
; DoubleRegs register class (v2i32, v4i16, v8i8) are treated as no-ops by the
; instruction selector and do not cause a "Cannot select" crash.
;
; All of i64, f64, v2i32, v4i16, v8i8 live in DoubleRegs (64-bit register
; pairs). A bitcast between any two of them is a pure reinterpretation of the
; same 64 bits. Therefore no instruction is emitted.
;
; Regression test for: llvm.org/PR195495
;   llc -mtriple=hexagon -mcpu=hexagonv68 -O3 crashed with
;   "Cannot select: f64 = bitcast v2i32" when compiling Eigen's packetmath.

; CHECK-LABEL: test_v2i32_to_f64:
; CHECK: dfcmp
; CHECK: jumpr r31
define i1 @test_v2i32_to_f64(<2 x i32> %a) {
  %bc = bitcast <2 x i32> %a to double
  %cmp = fcmp une double %bc, 0.0
  ret i1 %cmp
}

; f64->v2i32 is a no-op: the argument is already in a DoubleReg pair.
; CHECK-LABEL: test_f64_to_v2i32:
; CHECK-NOT: combine
; CHECK: jumpr r31
define <2 x i32> @test_f64_to_v2i32(double %a) {
  %bc = bitcast double %a to <2 x i32>
  ret <2 x i32> %bc
}

; CHECK-LABEL: test_v4i16_to_f64:
; CHECK: dfcmp
; CHECK: jumpr r31
define i1 @test_v4i16_to_f64(<4 x i16> %a) {
  %bc = bitcast <4 x i16> %a to double
  %cmp = fcmp une double %bc, 0.0
  ret i1 %cmp
}

; CHECK-LABEL: test_v8i8_to_f64:
; CHECK: dfcmp
; CHECK: jumpr r31
define i1 @test_v8i8_to_f64(<8 x i8> %a) {
  %bc = bitcast <8 x i8> %a to double
  %cmp = fcmp une double %bc, 0.0
  ret i1 %cmp
}

; Regression test: the original crash.
; <4 x i32> is passed in two v2i32 DoubleReg pairs; after type-legalizing
; <2 x f64> setcc into two scalar f64 setcc ops, each f64 operand is produced
; by a "f64 = bitcast v2i32" node that previously had no matching pattern.
; CHECK-LABEL: test_packetmath_reduced:
; CHECK: dfcmp
; CHECK: jumpr r31
define <2 x i1> @test_packetmath_reduced(<4 x i32> %arg) {
entry:
  %bc = bitcast <4 x i32> %arg to <2 x double>
  %cmp = fcmp une <2 x double> %bc, zeroinitializer
  ret <2 x i1> %cmp
}
