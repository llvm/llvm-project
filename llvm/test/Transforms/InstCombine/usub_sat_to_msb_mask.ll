
; RUN: opt -passes=instcombine -S < %s 2>&1 | FileCheck %s

declare i8 @llvm.usub.sat.i8(i8, i8)
declare i16 @llvm.usub.sat.i16(i16, i16)
declare i32 @llvm.usub.sat.i32(i32, i32)
declare i64 @llvm.usub.sat.i64(i64, i64)

define i8 @test_i8(i8 %a, i8 %b) {
; CHECK-LABEL: @test_i8(
; CHECK-NEXT: call i8 @llvm.usub.sat.i8(i8 %a, i8 96)
; CHECK-NEXT: call i8 @llvm.usub.sat.i8(i8 %b, i8 112)
; CHECK-NEXT: or i8
; CHECK-NEXT: and i8
; CHECK-NEXT: ret i8

  %a_sub = call i8 @llvm.usub.sat.i8(i8 %a, i8 223)
  %b_sub = call i8 @llvm.usub.sat.i8(i8 %b, i8 239)
  %or = or i8 %a_sub, %b_sub
  %cmp = icmp eq i8 %or, 0
  %res = select i1 %cmp, i8 0, i8 128
  ret i8 %res
}

define i16 @test_i16(i16 %a, i16 %b) {
; CHECK-LABEL: @test_i16(
; CHECK-NEXT: call i16 @llvm.usub.sat.i16(i16 %a, i16 32642)
; CHECK-NEXT: call i16 @llvm.usub.sat.i16(i16 %b, i16 32656)
; CHECK-NEXT: or i16
; CHECK-NEXT: and i16
; CHECK-NEXT: ret i16

  %a_sub = call i16 @llvm.usub.sat.i16(i16 %a, i16 65409)
  %b_sub = call i16 @llvm.usub.sat.i16(i16 %b, i16 65423)
  %or = or i16 %a_sub, %b_sub
  %cmp = icmp eq i16 %or, 0
  %res = select i1 %cmp, i16 0, i16 32768
  ret i16 %res
}

define i32 @test_i32(i32 %a, i32 %b) {
; CHECK-LABEL: @test_i32(
; CHECK-NEXT: call i32 @llvm.usub.sat.i32(i32 %a, i32 224)
; CHECK-NEXT: call i32 @llvm.usub.sat.i32(i32 %b, i32 240)
; CHECK-NEXT: or i32
; CHECK-NEXT: and i32
; CHECK-NEXT: ret i32

  %a_sub = call i32 @llvm.usub.sat.i32(i32 %a, i32 2147483871)
  %b_sub = call i32 @llvm.usub.sat.i32(i32 %b, i32 2147483887)
  %or = or i32 %a_sub, %b_sub
  %cmp = icmp eq i32 %or, 0
  %res = select i1 %cmp, i32 0, i32 2147483648
  ret i32 %res
}

define i64 @test_i64(i64 %a, i64 %b) {
; CHECK-LABEL: @test_i64(
; CHECK-NEXT: call i64 @llvm.usub.sat.i64(i64 %a, i64 224)
; CHECK-NEXT: call i64 @llvm.usub.sat.i64(i64 %b, i64 240)
; CHECK-NEXT: or i64
; CHECK-NEXT: and i64
; CHECK-NEXT: ret i64

  %a_sub = call i64 @llvm.usub.sat.i64(i64 %a, i64 9223372036854776031)
  %b_sub = call i64 @llvm.usub.sat.i64(i64 %b, i64 9223372036854776047)
  %or = or i64 %a_sub, %b_sub
  %cmp = icmp eq i64 %or, 0
  %res = select i1 %cmp, i64 0, i64 9223372036854775808
  ret i64 %res
}

define i32 @no_fold_due_to_small_K(i32 %a, i32 %b) {
; CHECK-LABEL: @no_fold_due_to_small_K(
; CHECK: call i32 @llvm.usub.sat.i32(i32 %a, i32 100)
; CHECK: call i32 @llvm.usub.sat.i32(i32 %b, i32 239)
; CHECK: or i32
; CHECK: icmp eq i32
; CHECK: select
; CHECK: ret i32

  %a_sub = call i32 @llvm.usub.sat.i32(i32 %a, i32 100)
  %b_sub = call i32 @llvm.usub.sat.i32(i32 %b, i32 239)
  %or = or i32 %a_sub, %b_sub
  %cmp = icmp eq i32 %or, 0
  %res = select i1 %cmp, i32 0, i32 2147483648
  ret i32 %res
}

define i32 @commuted_test_neg(i32 %a, i32 %b) {
; CHECK-LABEL: @commuted_test_neg(
; CHECK-NEXT: call i32 @llvm.usub.sat.i32(i32 %b, i32 239)
; CHECK-NEXT: call i32 @llvm.usub.sat.i32(i32 %a, i32 223)
; CHECK-NEXT: or i32
; CHECK-NEXT: icmp eq i32
; CHECK-NEXT: select
; CHECK-NEXT: ret i32

  %b_sub = call i32 @llvm.usub.sat.i32(i32 %b, i32 239)
  %a_sub = call i32 @llvm.usub.sat.i32(i32 %a, i32 223)
  %or = or i32 %b_sub, %a_sub
  %cmp = icmp eq i32 %or, 0
  %res = select i1 %cmp, i32 0, i32 2147483648
  ret i32 %res
}
define <4 x i32> @vector_test(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: @vector_test(
; CHECK-NEXT: call <4 x i32> @llvm.usub.sat.v4i32(<4 x i32> %a, <4 x i32> splat (i32 224))
; CHECK-NEXT: call <4 x i32> @llvm.usub.sat.v4i32(<4 x i32> %b, <4 x i32> splat (i32 240))
; CHECK-NEXT: or <4 x i32>
; CHECK-NEXT: and <4 x i32>
; CHECK-NEXT: ret <4 x i32>


  %a_sub = call <4 x i32> @llvm.usub.sat.v4i32(
              <4 x i32> %a,
              <4 x i32> <i32 2147483871, i32 2147483871, i32 2147483871, i32 2147483871>)
  %b_sub = call <4 x i32> @llvm.usub.sat.v4i32(
              <4 x i32> %b,
              <4 x i32> <i32 2147483887, i32 2147483887, i32 2147483887, i32 2147483887>)
  %or = or <4 x i32> %a_sub, %b_sub
  %cmp = icmp eq <4 x i32> %or, zeroinitializer
  %res = select <4 x i1> %cmp, <4 x i32> zeroinitializer,
                         <4 x i32> <i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648>
  ret <4 x i32> %res
}
