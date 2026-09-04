; RUN: llc -mtriple=hexagon -o - %s | FileCheck %s

; Test coverage for HexagonDepTimingClasses and HexagonInstrInfo::isComplex.
; Exercise multiply instructions (TC3x/TC4x) and ALU instructions
; (TC1/TC2) so the timing-class classification functions get invoked
; during scheduling.

; TC3x: multiply instructions
; CHECK-LABEL: test_multiply_timing:
; CHECK: mpyi
define i32 @test_multiply_timing(i32 %a, i32 %b, i32 %c) {
entry:
  %m1 = mul i32 %a, %b
  %m2 = mul i32 %m1, %c
  %m3 = mul i32 %m2, %a
  ret i32 %m3
}

; TC4x: multiply-accumulate instructions
; CHECK-LABEL: test_mac_timing:
; CHECK: mpyi
define i32 @test_mac_timing(i32 %a, i32 %b, i32 %c, i32 %d) {
entry:
  %m1 = mul i32 %a, %b
  %add1 = add i32 %m1, %c
  %m2 = mul i32 %add1, %d
  %add2 = add i32 %m2, %a
  ret i32 %add2
}

; TC1: simple ALU operations mixed with TC3x multiplies to force
; scheduling decisions that query timing classes.
; CHECK-LABEL: test_alu_mix:
; CHECK: add
define i32 @test_alu_mix(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) {
entry:
  %add1 = add i32 %a, %b
  %mul1 = mul i32 %c, %d
  %add2 = add i32 %add1, %mul1
  %xor1 = xor i32 %add2, %e
  %and1 = and i32 %xor1, %a
  %mul2 = mul i32 %and1, %b
  %or1 = or i32 %mul2, %c
  ret i32 %or1
}

; TC2early: compare + jump patterns that exercise early-timing predicates.
; CHECK-LABEL: test_cmp_jump:
; CHECK: cmp
define i32 @test_cmp_jump(i32 %a, i32 %b, i32 %c) {
entry:
  %cmp = icmp sgt i32 %a, %b
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %mul = mul i32 %a, %c
  br label %if.end

if.else:
  %add = add i32 %b, %c
  br label %if.end

if.end:
  %result = phi i32 [ %mul, %if.then ], [ %add, %if.else ]
  ret i32 %result
}

; Exercise 64-bit operations that may use different timing classes.
; CHECK-LABEL: test_64bit_ops:
; CHECK: add
define i64 @test_64bit_ops(i64 %a, i64 %b, i32 %c) {
entry:
  %add = add i64 %a, %b
  %ext = zext i32 %c to i64
  %mul = mul i64 %add, %ext
  ret i64 %mul
}

