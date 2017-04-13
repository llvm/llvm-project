; RUN: opt < %s -cost-model -analyze -mtriple=systemz-unknown -mcpu=z13 | FileCheck %s
;
; Note: The scalarized vector instructions costs are not including any
; extracts, due to the undef operands.

define void @add() {
  %res0 = add i8 undef, undef
  %res1 = add i16 undef, undef
  %res2 = add i32 undef, undef
  %res3 = add i64 undef, undef
  %res4 = add <2 x i8> undef, undef
  %res5 = add <2 x i16> undef, undef
  %res6 = add <2 x i32> undef, undef
  %res7 = add <2 x i64> undef, undef
  %res8 = add <4 x i8> undef, undef
  %res9 = add <4 x i16> undef, undef
  %res10 = add <4 x i32> undef, undef
  %res11 = add <4 x i64> undef, undef
  %res12 = add <8 x i8> undef, undef
  %res13 = add <8 x i16> undef, undef
  %res14 = add <8 x i32> undef, undef
  %res15 = add <8 x i64> undef, undef
  %res16 = add <16 x i8> undef, undef
  %res17 = add <16 x i16> undef, undef
  %res18 = add <16 x i32> undef, undef
  %res19 = add <16 x i64> undef, undef

; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res0 = add i8 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res1 = add i16 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res2 = add i32 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res3 = add i64 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res4 = add <2 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res5 = add <2 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res6 = add <2 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res7 = add <2 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res8 = add <4 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res9 = add <4 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res10 = add <4 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res11 = add <4 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res12 = add <8 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res13 = add <8 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res14 = add <8 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %res15 = add <8 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res16 = add <16 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res17 = add <16 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %res18 = add <16 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 8 for instruction:   %res19 = add <16 x i64> undef, undef

  ret void;
}

define void @sub() {
  %res0 = sub i8 undef, undef
  %res1 = sub i16 undef, undef
  %res2 = sub i32 undef, undef
  %res3 = sub i64 undef, undef
  %res4 = sub <2 x i8> undef, undef
  %res5 = sub <2 x i16> undef, undef
  %res6 = sub <2 x i32> undef, undef
  %res7 = sub <2 x i64> undef, undef
  %res8 = sub <4 x i8> undef, undef
  %res9 = sub <4 x i16> undef, undef
  %res10 = sub <4 x i32> undef, undef
  %res11 = sub <4 x i64> undef, undef
  %res12 = sub <8 x i8> undef, undef
  %res13 = sub <8 x i16> undef, undef
  %res14 = sub <8 x i32> undef, undef
  %res15 = sub <8 x i64> undef, undef
  %res16 = sub <16 x i8> undef, undef
  %res17 = sub <16 x i16> undef, undef
  %res18 = sub <16 x i32> undef, undef
  %res19 = sub <16 x i64> undef, undef

; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res0 = sub i8 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res1 = sub i16 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res2 = sub i32 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res3 = sub i64 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res4 = sub <2 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res5 = sub <2 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res6 = sub <2 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res7 = sub <2 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res8 = sub <4 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res9 = sub <4 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res10 = sub <4 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res11 = sub <4 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res12 = sub <8 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res13 = sub <8 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res14 = sub <8 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %res15 = sub <8 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res16 = sub <16 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res17 = sub <16 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %res18 = sub <16 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 8 for instruction:   %res19 = sub <16 x i64> undef, undef

  ret void;
}

define void @mul() {
  %res0 = mul i8 undef, undef
  %res1 = mul i16 undef, undef
  %res2 = mul i32 undef, undef
  %res3 = mul i64 undef, undef
  %res4 = mul <2 x i8> undef, undef
  %res5 = mul <2 x i16> undef, undef
  %res6 = mul <2 x i32> undef, undef
  %res7 = mul <2 x i64> undef, undef
  %res8 = mul <4 x i8> undef, undef
  %res9 = mul <4 x i16> undef, undef
  %res10 = mul <4 x i32> undef, undef
  %res11 = mul <4 x i64> undef, undef
  %res12 = mul <8 x i8> undef, undef
  %res13 = mul <8 x i16> undef, undef
  %res14 = mul <8 x i32> undef, undef
  %res15 = mul <8 x i64> undef, undef
  %res16 = mul <16 x i8> undef, undef
  %res17 = mul <16 x i16> undef, undef
  %res18 = mul <16 x i32> undef, undef
  %res19 = mul <16 x i64> undef, undef

; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res0 = mul i8 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res1 = mul i16 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res2 = mul i32 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res3 = mul i64 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res4 = mul <2 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res5 = mul <2 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res6 = mul <2 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 3 for instruction:   %res7 = mul <2 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res8 = mul <4 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res9 = mul <4 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res10 = mul <4 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 6 for instruction:   %res11 = mul <4 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res12 = mul <8 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res13 = mul <8 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res14 = mul <8 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 12 for instruction:   %res15 = mul <8 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res16 = mul <16 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res17 = mul <16 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %res18 = mul <16 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 24 for instruction:   %res19 = mul <16 x i64> undef, undef

  ret void;
}

define void @sdiv() {
  %res0 = sdiv i8 undef, undef
  %res1 = sdiv i16 undef, undef
  %res2 = sdiv i32 undef, undef
  %res3 = sdiv i64 undef, undef
  %res4 = sdiv <2 x i8> undef, undef
  %res5 = sdiv <2 x i16> undef, undef
  %res6 = sdiv <2 x i32> undef, undef
  %res7 = sdiv <2 x i64> undef, undef
  %res8 = sdiv <4 x i8> undef, undef
  %res9 = sdiv <4 x i16> undef, undef
  %res10 = sdiv <4 x i32> undef, undef
  %res11 = sdiv <4 x i64> undef, undef
  %res12 = sdiv <8 x i8> undef, undef
  %res13 = sdiv <8 x i16> undef, undef
  %res14 = sdiv <8 x i32> undef, undef
  %res15 = sdiv <8 x i64> undef, undef
  %res16 = sdiv <16 x i8> undef, undef
  %res17 = sdiv <16 x i16> undef, undef
  %res18 = sdiv <16 x i32> undef, undef
  %res19 = sdiv <16 x i64> undef, undef

; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %res0 = sdiv i8 undef, undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %res1 = sdiv i16 undef, undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res2 = sdiv i32 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res3 = sdiv i64 undef, undef
; CHECK: Cost Model: Found an estimated cost of 10 for instruction:   %res4 = sdiv <2 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 10 for instruction:   %res5 = sdiv <2 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 6 for instruction:   %res6 = sdiv <2 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 3 for instruction:   %res7 = sdiv <2 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %res8 = sdiv <4 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %res9 = sdiv <4 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 12 for instruction:   %res10 = sdiv <4 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 6 for instruction:   %res11 = sdiv <4 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 40 for instruction:   %res12 = sdiv <8 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 40 for instruction:   %res13 = sdiv <8 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 24 for instruction:   %res14 = sdiv <8 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 12 for instruction:   %res15 = sdiv <8 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 80 for instruction:   %res16 = sdiv <16 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 80 for instruction:   %res17 = sdiv <16 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 48 for instruction:   %res18 = sdiv <16 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 24 for instruction:   %res19 = sdiv <16 x i64> undef, undef

  ret void;
}

define void @srem() {
  %res0 = srem i8 undef, undef
  %res1 = srem i16 undef, undef
  %res2 = srem i32 undef, undef
  %res3 = srem i64 undef, undef
  %res4 = srem <2 x i8> undef, undef
  %res5 = srem <2 x i16> undef, undef
  %res6 = srem <2 x i32> undef, undef
  %res7 = srem <2 x i64> undef, undef
  %res8 = srem <4 x i8> undef, undef
  %res9 = srem <4 x i16> undef, undef
  %res10 = srem <4 x i32> undef, undef
  %res11 = srem <4 x i64> undef, undef
  %res12 = srem <8 x i8> undef, undef
  %res13 = srem <8 x i16> undef, undef
  %res14 = srem <8 x i32> undef, undef
  %res15 = srem <8 x i64> undef, undef
  %res16 = srem <16 x i8> undef, undef
  %res17 = srem <16 x i16> undef, undef
  %res18 = srem <16 x i32> undef, undef
  %res19 = srem <16 x i64> undef, undef

; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %res0 = srem i8 undef, undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %res1 = srem i16 undef, undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res2 = srem i32 undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res3 = srem i64 undef, undef
; CHECK: Cost Model: Found an estimated cost of 10 for instruction:   %res4 = srem <2 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 10 for instruction:   %res5 = srem <2 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 6 for instruction:   %res6 = srem <2 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 3 for instruction:   %res7 = srem <2 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %res8 = srem <4 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %res9 = srem <4 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 12 for instruction:   %res10 = srem <4 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 6 for instruction:   %res11 = srem <4 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 40 for instruction:   %res12 = srem <8 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 40 for instruction:   %res13 = srem <8 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 24 for instruction:   %res14 = srem <8 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 12 for instruction:   %res15 = srem <8 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 80 for instruction:   %res16 = srem <16 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 80 for instruction:   %res17 = srem <16 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 48 for instruction:   %res18 = srem <16 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 24 for instruction:   %res19 = srem <16 x i64> undef, undef

  ret void;
}

define void @udiv() {
  %res0 = udiv i8 undef, undef
  %res1 = udiv i16 undef, undef
  %res2 = udiv i32 undef, undef
  %res3 = udiv i64 undef, undef
  %res4 = udiv <2 x i8> undef, undef
  %res5 = udiv <2 x i16> undef, undef
  %res6 = udiv <2 x i32> undef, undef
  %res7 = udiv <2 x i64> undef, undef
  %res8 = udiv <4 x i8> undef, undef
  %res9 = udiv <4 x i16> undef, undef
  %res10 = udiv <4 x i32> undef, undef
  %res11 = udiv <4 x i64> undef, undef
  %res12 = udiv <8 x i8> undef, undef
  %res13 = udiv <8 x i16> undef, undef
  %res14 = udiv <8 x i32> undef, undef
  %res15 = udiv <8 x i64> undef, undef
  %res16 = udiv <16 x i8> undef, undef
  %res17 = udiv <16 x i16> undef, undef
  %res18 = udiv <16 x i32> undef, undef
  %res19 = udiv <16 x i64> undef, undef

; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %res0 = udiv i8 undef, undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %res1 = udiv i16 undef, undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res2 = udiv i32 undef, undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res3 = udiv i64 undef, undef
; CHECK: Cost Model: Found an estimated cost of 10 for instruction:   %res4 = udiv <2 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 10 for instruction:   %res5 = udiv <2 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 6 for instruction:   %res6 = udiv <2 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 5 for instruction:   %res7 = udiv <2 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %res8 = udiv <4 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %res9 = udiv <4 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 12 for instruction:   %res10 = udiv <4 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 10 for instruction:   %res11 = udiv <4 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 40 for instruction:   %res12 = udiv <8 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 40 for instruction:   %res13 = udiv <8 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 24 for instruction:   %res14 = udiv <8 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %res15 = udiv <8 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 80 for instruction:   %res16 = udiv <16 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 80 for instruction:   %res17 = udiv <16 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 48 for instruction:   %res18 = udiv <16 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 40 for instruction:   %res19 = udiv <16 x i64> undef, undef

  ret void;
}

define void @urem() {
  %res0 = urem i8 undef, undef
  %res1 = urem i16 undef, undef
  %res2 = urem i32 undef, undef
  %res3 = urem i64 undef, undef
  %res4 = urem <2 x i8> undef, undef
  %res5 = urem <2 x i16> undef, undef
  %res6 = urem <2 x i32> undef, undef
  %res7 = urem <2 x i64> undef, undef
  %res8 = urem <4 x i8> undef, undef
  %res9 = urem <4 x i16> undef, undef
  %res10 = urem <4 x i32> undef, undef
  %res11 = urem <4 x i64> undef, undef
  %res12 = urem <8 x i8> undef, undef
  %res13 = urem <8 x i16> undef, undef
  %res14 = urem <8 x i32> undef, undef
  %res15 = urem <8 x i64> undef, undef
  %res16 = urem <16 x i8> undef, undef
  %res17 = urem <16 x i16> undef, undef
  %res18 = urem <16 x i32> undef, undef
  %res19 = urem <16 x i64> undef, undef

; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %res0 = urem i8 undef, undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %res1 = urem i16 undef, undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res2 = urem i32 undef, undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res3 = urem i64 undef, undef
; CHECK: Cost Model: Found an estimated cost of 10 for instruction:   %res4 = urem <2 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 10 for instruction:   %res5 = urem <2 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 6 for instruction:   %res6 = urem <2 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 5 for instruction:   %res7 = urem <2 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %res8 = urem <4 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %res9 = urem <4 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 12 for instruction:   %res10 = urem <4 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 10 for instruction:   %res11 = urem <4 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 40 for instruction:   %res12 = urem <8 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 40 for instruction:   %res13 = urem <8 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 24 for instruction:   %res14 = urem <8 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %res15 = urem <8 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 80 for instruction:   %res16 = urem <16 x i8> undef, undef
; CHECK: Cost Model: Found an estimated cost of 80 for instruction:   %res17 = urem <16 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 48 for instruction:   %res18 = urem <16 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 40 for instruction:   %res19 = urem <16 x i64> undef, undef

  ret void;
}
