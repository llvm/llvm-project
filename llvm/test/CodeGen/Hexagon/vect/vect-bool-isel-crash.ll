; RUN: llc -march=hexagon < %s | FileCheck %s

; Check for a successful compilation.
; CHECK: jumpr r31

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define void @fred(ptr %a0, ptr %a1) #0 {
b0:
  %v1 = icmp sgt <8 x i32> undef, undef
  %v2 = extractelement <8 x i1> %v1, i32 4
  %v3 = select i1 %v2, i32 0, i32 undef
  %v4 = add nsw i32 %v3, 0
  %v5 = add nsw i32 0, %v4
  %v6 = extractelement <8 x i1> %v1, i32 6
  %v7 = select i1 %v6, i32 0, i32 undef
  %v8 = add nsw i32 %v7, %v5
  %v9 = add nsw i32 0, %v8
  %v10 = add nsw i32 0, %v9
  %v11 = load i32, ptr %a0, align 4
  %v12 = mul nsw i32 %v11, %v10
  %v13 = add nsw i32 %v12, 16384
  %v14 = ashr i32 %v13, 15
  %v15 = select i1 undef, i32 %v14, i32 255
  %v16 = trunc i32 %v15 to i8
  store i8 %v16, ptr %a1, align 1
  ret void
}

attributes #0 = { norecurse nounwind }
