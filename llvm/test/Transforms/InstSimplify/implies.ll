; RUN: opt -S %s -instsimplify | FileCheck %s

; A ==> A -> true
define i1 @test(i32 %length.i, i32 %i) {
; CHECK-LABEL: @test
; CHECK: ret i1 true
  %var29 = icmp slt i32 %i, %length.i
  %res = icmp uge i1 %var29, %var29
  ret i1 %res
}

; i +_{nsw} C_{>0} <s L ==> i <s L -> true
define i1 @test2(i32 %length.i, i32 %i) {
; CHECK-LABEL: @test2
; CHECK: ret i1 true
  %iplus1 = add nsw i32 %i, 1
  %var29 = icmp slt i32 %i, %length.i
  %var30 = icmp slt i32 %iplus1, %length.i
  %res = icmp ule i1 %var30, %var29
  ret i1 %res
}

; i + C_{>0} <s L ==> i <s L -> unknown without the nsw
define i1 @test2_neg(i32 %length.i, i32 %i) {
; CHECK-LABEL: @test2_neg
; CHECK:   ret i1 %res
  %iplus1 = add i32 %i, 1
  %var29 = icmp slt i32 %i, %length.i
  %var30 = icmp slt i32 %iplus1, %length.i
  %res = icmp ule i1 %var30, %var29
  ret i1 %res
}

; sle is not implication
define i1 @test2_neg2(i32 %length.i, i32 %i) {
; CHECK-LABEL: @test2_neg2
; CHECK:   ret i1 %res
  %iplus1 = add i32 %i, 1
  %var29 = icmp slt i32 %i, %length.i
  %var30 = icmp slt i32 %iplus1, %length.i
  %res = icmp sle i1 %var30, %var29
  ret i1 %res
}

; The binary operator has to be an add
define i1 @test2_neg3(i32 %length.i, i32 %i) {
; CHECK-LABEL: @test2_neg3
; CHECK:   ret i1 %res
  %iplus1 = sub nsw i32 %i, 1
  %var29 = icmp slt i32 %i, %length.i
  %var30 = icmp slt i32 %iplus1, %length.i
  %res = icmp ule i1 %var30, %var29
  ret i1 %res
}

; i +_{nsw} C_{>0} <s L ==> i <s L -> true
; With an inverted conditional (ule B A rather than canonical ugt A B
define i1 @test3(i32 %length.i, i32 %i) {
; CHECK-LABEL: @test3
; CHECK: ret i1 true
  %iplus1 = add nsw i32 %i, 1
  %var29 = icmp slt i32 %i, %length.i
  %var30 = icmp slt i32 %iplus1, %length.i
  %res = icmp uge i1 %var29, %var30
  ret i1 %res
}

; i +_{nuw} C_{>0} <u L ==> i <u L
define i1 @test4(i32 %length.i, i32 %i) {
; CHECK-LABEL: @test4
; CHECK: ret i1 true
  %iplus1 = add nuw i32 %i, 1
  %var29 = icmp ult i32 %i, %length.i
  %var30 = icmp ult i32 %iplus1, %length.i
  %res = icmp ule i1 %var30, %var29
  ret i1 %res
}

; A ==> A for vectors
define <4 x i1> @test5(<4 x i1> %vec) {
; CHECK-LABEL: @test5
; CHECK: ret <4 x i1> <i1 true, i1 true, i1 true, i1 true>
  %res = icmp ule <4 x i1> %vec, %vec
  ret <4 x i1> %res
}

; Don't crash on vector inputs - pr25040
define <4 x i1> @test6(<4 x i1> %a, <4 x i1> %b) {
; CHECK-LABEL: @test6
; CHECK: ret <4 x i1> %res
  %res = icmp ule <4 x i1> %a, %b
  ret <4 x i1> %res
}

; X >=(s) Y == X ==> Y (i1 1 becomes -1 for reasoning)
define i1 @test_sge(i32 %length.i, i32 %i) {
; CHECK-LABEL: @test_sge
; CHECK: ret i1 true
  %iplus1 = add nsw nuw i32 %i, 1
  %var29 = icmp ult i32 %i, %length.i
  %var30 = icmp ult i32 %iplus1, %length.i
  %res = icmp sge i1 %var30, %var29
  ret i1 %res
}
