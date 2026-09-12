; RUN: opt < %s -passes=slp-vectorizer -S -mtriple=nvptx64-nvidia-cuda -mcpu=sm_100 | FileCheck %s

target triple = "nvptx64-nvidia-cuda"

; Keep this scalar so later NVPTX lowering can recognize the i8 dot-product
; idiom instead of unpacking v2i8/v2i16 operations.
define i32 @i8-dot-product(ptr addrspace(3) %a, ptr addrspace(3) %b, i32 %acc) {
; CHECK-LABEL: @i8-dot-product(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[VA:%.*]] = load <8 x i8>, ptr addrspace(3) [[A:%.*]], align 8
; CHECK-NEXT:    [[VB:%.*]] = load <8 x i8>, ptr addrspace(3) [[B:%.*]], align 8
; CHECK-NEXT:    [[A0:%.*]] = extractelement <8 x i8> [[VA]], i32 0
; CHECK-NEXT:    [[A1:%.*]] = extractelement <8 x i8> [[VA]], i32 1
; CHECK-NEXT:    [[A2:%.*]] = extractelement <8 x i8> [[VA]], i32 2
; CHECK-NEXT:    [[A3:%.*]] = extractelement <8 x i8> [[VA]], i32 3
; CHECK-NEXT:    [[A4:%.*]] = extractelement <8 x i8> [[VA]], i32 4
; CHECK-NEXT:    [[A5:%.*]] = extractelement <8 x i8> [[VA]], i32 5
; CHECK-NEXT:    [[A6:%.*]] = extractelement <8 x i8> [[VA]], i32 6
; CHECK-NEXT:    [[A7:%.*]] = extractelement <8 x i8> [[VA]], i32 7
; CHECK-NEXT:    [[B0:%.*]] = extractelement <8 x i8> [[VB]], i32 0
; CHECK-NEXT:    [[B1:%.*]] = extractelement <8 x i8> [[VB]], i32 1
; CHECK-NEXT:    [[B2:%.*]] = extractelement <8 x i8> [[VB]], i32 2
; CHECK-NEXT:    [[B3:%.*]] = extractelement <8 x i8> [[VB]], i32 3
; CHECK-NEXT:    [[B4:%.*]] = extractelement <8 x i8> [[VB]], i32 4
; CHECK-NEXT:    [[B5:%.*]] = extractelement <8 x i8> [[VB]], i32 5
; CHECK-NEXT:    [[B6:%.*]] = extractelement <8 x i8> [[VB]], i32 6
; CHECK-NEXT:    [[B7:%.*]] = extractelement <8 x i8> [[VB]], i32 7
; CHECK-NEXT:    [[SA0:%.*]] = sext i8 [[A0]] to i32
; CHECK-NEXT:    [[SA1:%.*]] = sext i8 [[A1]] to i32
; CHECK-NEXT:    [[SA2:%.*]] = sext i8 [[A2]] to i32
; CHECK-NEXT:    [[SA3:%.*]] = sext i8 [[A3]] to i32
; CHECK-NEXT:    [[SA4:%.*]] = sext i8 [[A4]] to i32
; CHECK-NEXT:    [[SA5:%.*]] = sext i8 [[A5]] to i32
; CHECK-NEXT:    [[SA6:%.*]] = sext i8 [[A6]] to i32
; CHECK-NEXT:    [[SA7:%.*]] = sext i8 [[A7]] to i32
; CHECK-NEXT:    [[SB0:%.*]] = sext i8 [[B0]] to i32
; CHECK-NEXT:    [[SB1:%.*]] = sext i8 [[B1]] to i32
; CHECK-NEXT:    [[SB2:%.*]] = sext i8 [[B2]] to i32
; CHECK-NEXT:    [[SB3:%.*]] = sext i8 [[B3]] to i32
; CHECK-NEXT:    [[SB4:%.*]] = sext i8 [[B4]] to i32
; CHECK-NEXT:    [[SB5:%.*]] = sext i8 [[B5]] to i32
; CHECK-NEXT:    [[SB6:%.*]] = sext i8 [[B6]] to i32
; CHECK-NEXT:    [[SB7:%.*]] = sext i8 [[B7]] to i32
; CHECK-NEXT:    [[M0:%.*]] = mul nsw i32 [[SA0]], [[SB0]]
; CHECK-NEXT:    [[SUM0:%.*]] = add nsw i32 [[ACC:%.*]], [[M0]]
; CHECK-NEXT:    [[M1:%.*]] = mul nsw i32 [[SA1]], [[SB1]]
; CHECK-NEXT:    [[SUM1:%.*]] = add nsw i32 [[SUM0]], [[M1]]
; CHECK-NEXT:    [[M2:%.*]] = mul nsw i32 [[SA2]], [[SB2]]
; CHECK-NEXT:    [[SUM2:%.*]] = add nsw i32 [[SUM1]], [[M2]]
; CHECK-NEXT:    [[M3:%.*]] = mul nsw i32 [[SA3]], [[SB3]]
; CHECK-NEXT:    [[SUM3:%.*]] = add nsw i32 [[SUM2]], [[M3]]
; CHECK-NEXT:    [[M4:%.*]] = mul nsw i32 [[SA4]], [[SB4]]
; CHECK-NEXT:    [[SUM4:%.*]] = add nsw i32 [[SUM3]], [[M4]]
; CHECK-NEXT:    [[M5:%.*]] = mul nsw i32 [[SA5]], [[SB5]]
; CHECK-NEXT:    [[SUM5:%.*]] = add nsw i32 [[SUM4]], [[M5]]
; CHECK-NEXT:    [[M6:%.*]] = mul nsw i32 [[SA6]], [[SB6]]
; CHECK-NEXT:    [[SUM6:%.*]] = add nsw i32 [[SUM5]], [[M6]]
; CHECK-NEXT:    [[M7:%.*]] = mul nsw i32 [[SA7]], [[SB7]]
; CHECK-NEXT:    [[SUM7:%.*]] = add nsw i32 [[SUM6]], [[M7]]
; CHECK-NEXT:    ret i32 [[SUM7]]
entry:
  %va = load <8 x i8>, ptr addrspace(3) %a, align 8
  %vb = load <8 x i8>, ptr addrspace(3) %b, align 8
  %a0 = extractelement <8 x i8> %va, i32 0
  %a1 = extractelement <8 x i8> %va, i32 1
  %a2 = extractelement <8 x i8> %va, i32 2
  %a3 = extractelement <8 x i8> %va, i32 3
  %a4 = extractelement <8 x i8> %va, i32 4
  %a5 = extractelement <8 x i8> %va, i32 5
  %a6 = extractelement <8 x i8> %va, i32 6
  %a7 = extractelement <8 x i8> %va, i32 7
  %b0 = extractelement <8 x i8> %vb, i32 0
  %b1 = extractelement <8 x i8> %vb, i32 1
  %b2 = extractelement <8 x i8> %vb, i32 2
  %b3 = extractelement <8 x i8> %vb, i32 3
  %b4 = extractelement <8 x i8> %vb, i32 4
  %b5 = extractelement <8 x i8> %vb, i32 5
  %b6 = extractelement <8 x i8> %vb, i32 6
  %b7 = extractelement <8 x i8> %vb, i32 7
  %sa0 = sext i8 %a0 to i32
  %sa1 = sext i8 %a1 to i32
  %sa2 = sext i8 %a2 to i32
  %sa3 = sext i8 %a3 to i32
  %sa4 = sext i8 %a4 to i32
  %sa5 = sext i8 %a5 to i32
  %sa6 = sext i8 %a6 to i32
  %sa7 = sext i8 %a7 to i32
  %sb0 = sext i8 %b0 to i32
  %sb1 = sext i8 %b1 to i32
  %sb2 = sext i8 %b2 to i32
  %sb3 = sext i8 %b3 to i32
  %sb4 = sext i8 %b4 to i32
  %sb5 = sext i8 %b5 to i32
  %sb6 = sext i8 %b6 to i32
  %sb7 = sext i8 %b7 to i32
  %m0 = mul nsw i32 %sa0, %sb0
  %sum0 = add nsw i32 %acc, %m0
  %m1 = mul nsw i32 %sa1, %sb1
  %sum1 = add nsw i32 %sum0, %m1
  %m2 = mul nsw i32 %sa2, %sb2
  %sum2 = add nsw i32 %sum1, %m2
  %m3 = mul nsw i32 %sa3, %sb3
  %sum3 = add nsw i32 %sum2, %m3
  %m4 = mul nsw i32 %sa4, %sb4
  %sum4 = add nsw i32 %sum3, %m4
  %m5 = mul nsw i32 %sa5, %sb5
  %sum5 = add nsw i32 %sum4, %m5
  %m6 = mul nsw i32 %sa6, %sb6
  %sum6 = add nsw i32 %sum5, %m6
  %m7 = mul nsw i32 %sa7, %sb7
  %sum7 = add nsw i32 %sum6, %m7
  ret i32 %sum7
}

; Keep the minimal scalar-load form scalar too. This catches the two-lane
; unpack shape independently of vector-load extraction.
define i32 @dp4a_like_i8(ptr %a, ptr %b) {
; CHECK-LABEL: @dp4a_like_i8(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[A0:%.*]] = load i8, ptr [[A:%.*]], align 4
; CHECK-NEXT:    [[A1P:%.*]] = getelementptr i8, ptr [[A]], i64 1
; CHECK-NEXT:    [[A1:%.*]] = load i8, ptr [[A1P]], align 1
; CHECK-NEXT:    [[B0:%.*]] = load i8, ptr [[B:%.*]], align 4
; CHECK-NEXT:    [[B1P:%.*]] = getelementptr i8, ptr [[B]], i64 1
; CHECK-NEXT:    [[B1:%.*]] = load i8, ptr [[B1P]], align 1
; CHECK-NEXT:    [[SA0:%.*]] = sext i8 [[A0]] to i32
; CHECK-NEXT:    [[SA1:%.*]] = sext i8 [[A1]] to i32
; CHECK-NEXT:    [[SB0:%.*]] = sext i8 [[B0]] to i32
; CHECK-NEXT:    [[SB1:%.*]] = sext i8 [[B1]] to i32
; CHECK-NEXT:    [[M0:%.*]] = mul i32 [[SA0]], [[SB0]]
; CHECK-NEXT:    [[M1:%.*]] = mul i32 [[SA1]], [[SB1]]
; CHECK-NEXT:    [[SUM:%.*]] = add i32 [[M0]], [[M1]]
; CHECK-NEXT:    ret i32 [[SUM]]
;
entry:
  %a0 = load i8, ptr %a, align 4
  %a1p = getelementptr i8, ptr %a, i64 1
  %a1 = load i8, ptr %a1p, align 1
  %b0 = load i8, ptr %b, align 4
  %b1p = getelementptr i8, ptr %b, i64 1
  %b1 = load i8, ptr %b1p, align 1
  %sa0 = sext i8 %a0 to i32
  %sa1 = sext i8 %a1 to i32
  %sb0 = sext i8 %b0 to i32
  %sb1 = sext i8 %b1 to i32
  %m0 = mul i32 %sa0, %sb0
  %m1 = mul i32 %sa1, %sb1
  %sum = add i32 %m0, %m1
  ret i32 %sum
}
