; RUN: opt -O2 -S < %s  | FileCheck %s
; RUN: llc -o /dev/null 2>&1 < %s
; RUN: llc -O0 -o /dev/null 2>&1 < %s

;; The llc runs above are just to ensure it doesn't blow up upon
;; seeing an is_constant intrinsic.

declare i1 @llvm.is.constant.i32(i32 %a)
declare i1 @llvm.is.constant.i64(i64 %a)
declare i1 @llvm.is.constant.i256(i256 %a)
declare i1 @llvm.is.constant.v2i64(<2 x i64> %a)
declare i1 @llvm.is.constant.f32(float %a)
declare i1 @llvm.is.constant.sl_i32i32s({i32, i32} %a)
declare i1 @llvm.is.constant.a2i64([2 x i64] %a)
declare i1 @llvm.is.constant.p0i64(i64* %a)

;; Basic test that optimization folds away the is.constant when given
;; a constant.
define i1 @test_constant() #0 {
; CHECK-LABEL: @test_constant(
; CHECK-NOT: llvm.is.constant
; CHECK: ret i1 true
%y = call i1 @llvm.is.constant.i32(i32 44)
  ret i1 %y
}

;; And test that the intrinsic sticks around when given a
;; non-constant.
define i1 @test_nonconstant(i32 %x) #0 {
; CHECK-LABEL: @test_nonconstant(
; CHECK: @llvm.is.constant
  %y = call i1 @llvm.is.constant.i32(i32 %x)
  ret i1 %y
}

;; Ensure that nested is.constants fold.
define i32 @test_nested() #0 {
; CHECK-LABEL: @test_nested(
; CHECK-NOT: llvm.is.constant
; CHECK: ret i32 13
  %val1 = call i1 @llvm.is.constant.i32(i32 27)
  %val2 = zext i1 %val1 to i32
  %val3 = add i32 %val2, 12
  %1 = call i1 @llvm.is.constant.i32(i32 %val3)
  %2 = zext i1 %1 to i32
  %3 = add i32 %2, 12
  ret i32 %3
}

@G = global [2 x i64] zeroinitializer
define i1 @test_global() #0 {
; CHECK-LABEL: @test_global(
; CHECK: llvm.is.constant
  %ret = call i1 @llvm.is.constant.p0i64(i64* getelementptr ([2 x i64], [2 x i64]* @G, i32 0, i32 0))
  ret i1 %ret
}

define i1 @test_diff() #0 {
; CHECK-LABEL: @test_diff(
  %ret = call i1 @llvm.is.constant.i64(i64 sub (
      i64 ptrtoint (i64* getelementptr inbounds ([2 x i64], [2 x i64]* @G, i64 0, i64 1) to i64),
      i64 ptrtoint ([2 x i64]* @G to i64)))
  ret i1 %ret
}

define i1 @test_various_types(i256 %int, float %float, <2 x i64> %vec, {i32, i32} %struct, [2 x i64] %arr, i64* %ptr) #0 {
; CHECK-LABEL: @test_various_types(
; CHECK: llvm.is.constant
; CHECK: llvm.is.constant
; CHECK: llvm.is.constant
; CHECK: llvm.is.constant
; CHECK: llvm.is.constant
; CHECK: llvm.is.constant
; CHECK-NOT: llvm.is.constant
  %v1 = call i1 @llvm.is.constant.i256(i256 %int)
  %v2 = call i1 @llvm.is.constant.f32(float %float)
  %v3 = call i1 @llvm.is.constant.v2i64(<2 x i64> %vec)
  %v4 = call i1 @llvm.is.constant.sl_i32i32s({i32, i32} %struct)
  %v5 = call i1 @llvm.is.constant.a2i64([2 x i64] %arr)
  %v6 = call i1 @llvm.is.constant.p0i64(i64* %ptr)

  %c1 = call i1 @llvm.is.constant.i256(i256 -1)
  %c2 = call i1 @llvm.is.constant.f32(float 17.0)
  %c3 = call i1 @llvm.is.constant.v2i64(<2 x i64> <i64 -1, i64 44>)
  %c4 = call i1 @llvm.is.constant.sl_i32i32s({i32, i32} {i32 -1, i32 32})
  %c5 = call i1 @llvm.is.constant.a2i64([2 x i64] [i64 -1, i64 32])
  %c6 = call i1 @llvm.is.constant.p0i64(i64* inttoptr (i32 42 to i64*))

  %x1 = add i1 %v1, %c1
  %x2 = add i1 %v2, %c2
  %x3 = add i1 %v3, %c3
  %x4 = add i1 %v4, %c4
  %x5 = add i1 %v5, %c5
  %x6 = add i1 %v6, %c6

  %res2 = add i1 %x1, %x2
  %res3 = add i1 %res2, %x3
  %res4 = add i1 %res3, %x4
  %res5 = add i1 %res4, %x5
  %res6 = add i1 %res5, %x6

  ret i1 %res6
}

define i1 @test_various_types2() #0 {
; CHECK-LABEL: @test_various_types2(
; CHECK: ret i1 false
  %r = call i1 @test_various_types(i256 -1, float 22.0, <2 x i64> <i64 -1, i64 44>,
                     {i32, i32} {i32 -1, i32 55}, [2 x i64] [i64 -1, i64 55],
		     i64* inttoptr (i64 42 to i64*))
  ret i1 %r
}

attributes #0 = { nounwind uwtable }
