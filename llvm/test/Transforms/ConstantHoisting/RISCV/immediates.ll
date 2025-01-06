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
; CHECK: %const = bitcast i64 32766 to i64
  %1 = mul i64 %a, 32766
  %2 = add i64 %1, 32766
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

; Check that we don't hoist mul by one less than a power of 2.
define i64 @test15(i64 %a) nounwind {
; CHECK-LABEL: test15
; CHECK: mul i64 %a, 65535
  %1 = mul i64 %a, 65535
  %2 = mul i64 %1, 65535
  ret i64 %2
}

; Check that we don't hoist mul by one more than a power of 2.
define i64 @test16(i64 %a) nounwind {
; CHECK-LABEL: test16
; CHECK: mul i64 %a, 65537
  %1 = mul i64 %a, 65537
  %2 = mul i64 %1, 65537
  ret i64 %2
}

; Check that we hoist the absolute address of the stores to the entry block.
define void @test17(ptr %s, i32 %size) nounwind {
; CHECK-LABEL: test17
; CHECK: %const = bitcast i32 -1073741792 to i32
; CHECK: %0 = inttoptr i32 %const to ptr
; CHECK: store i32 20, ptr %0
; CHECK: %1 = inttoptr i32 %const to ptr
; CHECK: store i32 10, ptr %1
entry:
  %cond = icmp eq i32 %size, 0
  br i1 %cond, label %if.true, label %exit
if.true:
  store i32 20, ptr inttoptr (i32 -1073741792 to ptr)
  br label %exit
exit:
  store i32 10, ptr inttoptr (i32 -1073741792 to ptr)
  ret void
}

; Check that we hoist the absolute address of the loads to the entry block.
define i32 @test18(ptr %s, i32 %size) nounwind {
; CHECK-LABEL: test18
; CHECK: %const = bitcast i32 -1073741792 to i32
; CHECK: %0 = inttoptr i32 %const to ptr
; CHECK: %1 = load i32, ptr %0
; CHECK: %2 = inttoptr i32 %const to ptr
; CHECK: %3 = load i32, ptr %2
entry:
  %cond = icmp eq i32 %size, 0
  br i1 %cond, label %if.true, label %if.false
if.true:
  %0 = load i32, ptr inttoptr (i32 -1073741792 to ptr)
  br label %return
if.false:
  %1 = load i32, ptr inttoptr (i32 -1073741792 to ptr)
  br label %return
return:
  %val = phi i32 [%0, %if.true], [%1, %if.false]
  ret i32 %val
}


; For addresses between [0, 2048), we can use ld/sd xN, address(zero), so don't
; hoist.
define void @test19(ptr %s, i32 %size) nounwind {
; CHECK-LABEL: test19
; CHECK: store i32 20, ptr inttoptr (i32 2044 to ptr)
; CHECK: store i32 10, ptr inttoptr (i32 2044 to ptr)
entry:
  %cond = icmp eq i32 %size, 0
  br i1 %cond, label %if.true, label %exit
if.true:
  store i32 20, ptr inttoptr (i32 2044 to ptr)
  br label %exit
exit:
  store i32 10, ptr inttoptr (i32 2044 to ptr)
  ret void
}

; Check that we use a common base for immediates needed by a store if the
; constants require more than 1 instruction.
define void @test20(ptr %p1, ptr %p2) {
; CHECK-LABEL: test20
; CHECK: %const = bitcast i32 15111111 to i32
; CHECK: store i32 %const, ptr %p1, align 4
; CHECK: %const_mat = add i32 %const, 1
; CHECK: store i32 %const_mat, ptr %p2, align 4
  store i32 15111111, ptr %p1, align 4
  store i32 15111112, ptr %p2, align 4
  ret void
}

define void @test21(ptr %p1, ptr %p2) {
; CHECK-LABEL: define void @test21(
; CHECK-SAME: ptr [[P1:%.*]], ptr [[P2:%.*]]) {
; CHECK-NEXT:    store i32 15111111, ptr [[P1]], align 1
; CHECK-NEXT:    store i32 15111112, ptr [[P2]], align 1
; CHECK-NEXT:    ret void
;
  store i32 15111111, ptr %p1, align 1
  store i32 15111112, ptr %p2, align 1
  ret void
}

; 0 immediates shouldn't be hoisted.
define void @test22(ptr %p1, ptr %p2) {
; CHECK-LABEL: define void @test22(
; CHECK-SAME: ptr [[P1:%.*]], ptr [[P2:%.*]]) {
; CHECK-NEXT:    store i64 0, ptr [[P1]], align 8
; CHECK-NEXT:    store i64 -1, ptr [[P2]], align 8
; CHECK-NEXT:    ret void
;
  store i64 0, ptr %p1, align 8
  store i64 -1, ptr %p2, align 8
  ret void
}

; 0 immediates shouldn't be hoisted.
define void @test23(ptr %p1, ptr %p2) {
; CHECK-LABEL: define void @test23(
; CHECK-SAME: ptr [[P1:%.*]], ptr [[P2:%.*]]) {
; CHECK-NEXT:    store i127 0, ptr [[P1]], align 8
; CHECK-NEXT:    store i127 -1, ptr [[P2]], align 8
; CHECK-NEXT:    ret void
;
  store i127 0, ptr %p1, align 8
  store i127 -1, ptr %p2, align 8
  ret void
}

; Hoisting doesn't happen for types that aren't legal.
define void @test24(ptr %p1, ptr %p2) {
; CHECK-LABEL: define void @test24(
; CHECK-SAME: ptr [[P1:%.*]], ptr [[P2:%.*]]) {
; CHECK-NEXT:    store i128 15111111, ptr [[P1]], align 4
; CHECK-NEXT:    store i128 15111112, ptr [[P2]], align 4
; CHECK-NEXT:    ret void
;
  store i128 15111111, ptr %p1, align 4
  store i128 15111112, ptr %p2, align 4
  ret void
}
