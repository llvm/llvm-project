; RUN: opt < %s -passes="print<cost-model>" 2>&1 -disable-output -mtriple=systemz-unknown -mcpu=z13 | FileCheck %s

define void @and() {
  %res0 = and i8 undef, undef
  %res1 = and i16 undef, undef
  %res2 = and i32 undef, undef
  %res3 = and i64 undef, undef
  %resQ = and i128 undef, undef
  %res4 = and <2 x i8> undef, undef
  %res5 = and <2 x i16> undef, undef
  %res6 = and <2 x i32> undef, undef
  %res7 = and <2 x i64> undef, undef
  %res8 = and <4 x i8> undef, undef
  %res9 = and <4 x i16> undef, undef
  %res10 = and <4 x i32> undef, undef
  %res11 = and <4 x i64> undef, undef
  %res12 = and <8 x i8> undef, undef
  %res13 = and <8 x i16> undef, undef
  %res14 = and <8 x i32> undef, undef
  %res15 = and <8 x i64> undef, undef
  %res16 = and <16 x i8> undef, undef
  %res17 = and <16 x i16> undef, undef
  %res18 = and <16 x i32> undef, undef
  %res19 = and <16 x i64> undef, undef

; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res0 = and i8 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res1 = and i16 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res2 = and i32 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res3 = and i64 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %resQ = and i128 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res4 = and <2 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res5 = and <2 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res6 = and <2 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res7 = and <2 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res8 = and <4 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res9 = and <4 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res10 = and <4 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res11 = and <4 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res12 = and <8 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res13 = and <8 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res14 = and <8 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %res15 = and <8 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res16 = and <16 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res17 = and <16 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %res18 = and <16 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 8 for instruction:   %res19 = and <16 x i64> undef, undef

  ret void;
}

define void @ashr() {
  %res0 = ashr i8 undef, undef
  %res1 = ashr i16 undef, undef
  %res2 = ashr i32 undef, undef
  %res3 = ashr i64 undef, undef
  %resQ = ashr i128 undef, undef
  %res4 = ashr <2 x i8> undef, undef
  %res5 = ashr <2 x i16> undef, undef
  %res6 = ashr <2 x i32> undef, undef
  %res7 = ashr <2 x i64> undef, undef
  %res8 = ashr <4 x i8> undef, undef
  %res9 = ashr <4 x i16> undef, undef
  %res10 = ashr <4 x i32> undef, undef
  %res11 = ashr <4 x i64> undef, undef
  %res12 = ashr <8 x i8> undef, undef
  %res13 = ashr <8 x i16> undef, undef
  %res14 = ashr <8 x i32> undef, undef
  %res15 = ashr <8 x i64> undef, undef
  %res16 = ashr <16 x i8> undef, undef
  %res17 = ashr <16 x i16> undef, undef
  %res18 = ashr <16 x i32> undef, undef
  %res19 = ashr <16 x i64> undef, undef

; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res0 = ashr i8 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res1 = ashr i16 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res2 = ashr i32 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res3 = ashr i64 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %resQ = ashr i128 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res4 = ashr <2 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res5 = ashr <2 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res6 = ashr <2 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res7 = ashr <2 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res8 = ashr <4 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res9 = ashr <4 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res10 = ashr <4 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res11 = ashr <4 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res12 = ashr <8 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res13 = ashr <8 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res14 = ashr <8 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %res15 = ashr <8 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res16 = ashr <16 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res17 = ashr <16 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %res18 = ashr <16 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 8 for instruction:   %res19 = ashr <16 x i64> undef, undef

  ret void;
}

define void @lshr() {
  %res0 = lshr i8 undef, undef
  %res1 = lshr i16 undef, undef
  %res2 = lshr i32 undef, undef
  %res3 = lshr i64 undef, undef
  %resQ = lshr i128 undef, undef
  %res4 = lshr <2 x i8> undef, undef
  %res5 = lshr <2 x i16> undef, undef
  %res6 = lshr <2 x i32> undef, undef
  %res7 = lshr <2 x i64> undef, undef
  %res8 = lshr <4 x i8> undef, undef
  %res9 = lshr <4 x i16> undef, undef
  %res10 = lshr <4 x i32> undef, undef
  %res11 = lshr <4 x i64> undef, undef
  %res12 = lshr <8 x i8> undef, undef
  %res13 = lshr <8 x i16> undef, undef
  %res14 = lshr <8 x i32> undef, undef
  %res15 = lshr <8 x i64> undef, undef
  %res16 = lshr <16 x i8> undef, undef
  %res17 = lshr <16 x i16> undef, undef
  %res18 = lshr <16 x i32> undef, undef
  %res19 = lshr <16 x i64> undef, undef

; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res0 = lshr i8 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res1 = lshr i16 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res2 = lshr i32 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res3 = lshr i64 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %resQ = lshr i128 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res4 = lshr <2 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res5 = lshr <2 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res6 = lshr <2 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res7 = lshr <2 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res8 = lshr <4 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res9 = lshr <4 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res10 = lshr <4 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res11 = lshr <4 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res12 = lshr <8 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res13 = lshr <8 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res14 = lshr <8 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %res15 = lshr <8 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res16 = lshr <16 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res17 = lshr <16 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %res18 = lshr <16 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 8 for instruction:   %res19 = lshr <16 x i64> undef, undef

  ret void;
}

define void @or() {
  %res0 = or i8 undef, undef
  %res1 = or i16 undef, undef
  %res2 = or i32 undef, undef
  %res3 = or i64 undef, undef
  %resQ = or i128 undef, undef
  %res4 = or <2 x i8> undef, undef
  %res5 = or <2 x i16> undef, undef
  %res6 = or <2 x i32> undef, undef
  %res7 = or <2 x i64> undef, undef
  %res8 = or <4 x i8> undef, undef
  %res9 = or <4 x i16> undef, undef
  %res10 = or <4 x i32> undef, undef
  %res11 = or <4 x i64> undef, undef
  %res12 = or <8 x i8> undef, undef
  %res13 = or <8 x i16> undef, undef
  %res14 = or <8 x i32> undef, undef
  %res15 = or <8 x i64> undef, undef
  %res16 = or <16 x i8> undef, undef
  %res17 = or <16 x i16> undef, undef
  %res18 = or <16 x i32> undef, undef
  %res19 = or <16 x i64> undef, undef
  
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res0 = or i8 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res1 = or i16 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res2 = or i32 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res3 = or i64 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %resQ = or i128 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res4 = or <2 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res5 = or <2 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res6 = or <2 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res7 = or <2 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res8 = or <4 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res9 = or <4 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res10 = or <4 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res11 = or <4 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res12 = or <8 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res13 = or <8 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res14 = or <8 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %res15 = or <8 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res16 = or <16 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res17 = or <16 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %res18 = or <16 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 8 for instruction:   %res19 = or <16 x i64> undef, undef

  ret void;
}

define void @shl() {
  %res0 = shl i8 undef, undef
  %res1 = shl i16 undef, undef
  %res2 = shl i32 undef, undef
  %res3 = shl i64 undef, undef
  %resQ = shl i128 undef, undef
  %res4 = shl <2 x i8> undef, undef
  %res5 = shl <2 x i16> undef, undef
  %res6 = shl <2 x i32> undef, undef
  %res7 = shl <2 x i64> undef, undef
  %res8 = shl <4 x i8> undef, undef
  %res9 = shl <4 x i16> undef, undef
  %res10 = shl <4 x i32> undef, undef
  %res11 = shl <4 x i64> undef, undef
  %res12 = shl <8 x i8> undef, undef
  %res13 = shl <8 x i16> undef, undef
  %res14 = shl <8 x i32> undef, undef
  %res15 = shl <8 x i64> undef, undef
  %res16 = shl <16 x i8> undef, undef
  %res17 = shl <16 x i16> undef, undef
  %res18 = shl <16 x i32> undef, undef
  %res19 = shl <16 x i64> undef, undef
  
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res0 = shl i8 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res1 = shl i16 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res2 = shl i32 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res3 = shl i64 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %resQ = shl i128 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res4 = shl <2 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res5 = shl <2 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res6 = shl <2 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res7 = shl <2 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res8 = shl <4 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res9 = shl <4 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res10 = shl <4 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res11 = shl <4 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res12 = shl <8 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res13 = shl <8 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res14 = shl <8 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %res15 = shl <8 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res16 = shl <16 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res17 = shl <16 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %res18 = shl <16 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 8 for instruction:   %res19 = shl <16 x i64> undef, undef

  ret void;
}

define void @xor() {
  %res0 = xor i8 undef, undef
  %res1 = xor i16 undef, undef
  %res2 = xor i32 undef, undef
  %res3 = xor i64 undef, undef
  %resQ = xor i128 undef, undef
  %res4 = xor <2 x i8> undef, undef
  %res5 = xor <2 x i16> undef, undef
  %res6 = xor <2 x i32> undef, undef
  %res7 = xor <2 x i64> undef, undef
  %res8 = xor <4 x i8> undef, undef
  %res9 = xor <4 x i16> undef, undef
  %res10 = xor <4 x i32> undef, undef
  %res11 = xor <4 x i64> undef, undef
  %res12 = xor <8 x i8> undef, undef
  %res13 = xor <8 x i16> undef, undef
  %res14 = xor <8 x i32> undef, undef
  %res15 = xor <8 x i64> undef, undef
  %res16 = xor <16 x i8> undef, undef
  %res17 = xor <16 x i16> undef, undef
  %res18 = xor <16 x i32> undef, undef
  %res19 = xor <16 x i64> undef, undef
  
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res0 = xor i8 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res1 = xor i16 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res2 = xor i32 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res3 = xor i64 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %resQ = xor i128 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res4 = xor <2 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res5 = xor <2 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res6 = xor <2 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res7 = xor <2 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res8 = xor <4 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res9 = xor <4 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res10 = xor <4 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res11 = xor <4 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res12 = xor <8 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res13 = xor <8 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res14 = xor <8 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %res15 = xor <8 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res16 = xor <16 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res17 = xor <16 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %res18 = xor <16 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 8 for instruction:   %res19 = xor <16 x i64> undef, undef

  ret void;
}
