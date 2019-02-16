; RUN: llc -march=hexagon < %s | FileCheck %s

; Check that both functions compile successfully.


target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

; CHECK-LABEL: danny:
; CHECK: vmem
define void @danny(i16* %a0) #0 {
b0:
  %v1 = load i16, i16* %a0, align 2
  %v2 = insertelement <8 x i16> undef, i16 %v1, i32 6
  %v3 = insertelement <8 x i16> %v2, i16 undef, i32 7
  %v4 = sext <8 x i16> %v3 to <8 x i32>
  %v5 = mul <8 x i32> %v4, <i32 -36410, i32 -36410, i32 -36410, i32 -36410, i32 -36410, i32 -36410, i32 -36410, i32 -36410>
  %v6 = add <8 x i32> %v5, <i32 32768, i32 32768, i32 32768, i32 32768, i32 32768, i32 32768, i32 32768, i32 32768>
  %v7 = add <8 x i32> %v6, zeroinitializer
  %v8 = ashr <8 x i32> %v7, <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  %v9 = add nsw <8 x i32> zeroinitializer, %v8
  %v10 = shl <8 x i32> %v9, <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  %v11 = ashr exact <8 x i32> %v10, <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  %v12 = sub nsw <8 x i32> zeroinitializer, %v11
  %v13 = trunc <8 x i32> %v12 to <8 x i16>
  %v14 = extractelement <8 x i16> %v13, i32 7
  store i16 %v14, i16* %a0, align 2
  ret void
}

; CHECK-LABEL: sammy:
; CHECK: vmem
define void @sammy(i16* %a0) #1 {
b0:
  %v1 = load i16, i16* %a0, align 2
  %v2 = insertelement <16 x i16> undef, i16 %v1, i32 14
  %v3 = insertelement <16 x i16> %v2, i16 undef, i32 15
  %v4 = sext <16 x i16> %v3 to <16 x i32>
  %v5 = mul <16 x i32> %v4, <i32 -36410, i32 -36410, i32 -36410, i32 -36410, i32 -36410, i32 -36410, i32 -36410, i32 -36410, i32 -36410, i32 -36410, i32 -36410, i32 -36410, i32 -36410, i32 -36410, i32 -36410, i32 -36410>
  %v6 = add <16 x i32> %v5, <i32 32768, i32 32768, i32 32768, i32 32768, i32 32768, i32 32768, i32 32768, i32 32768, i32 32768, i32 32768, i32 32768, i32 32768, i32 32768, i32 32768, i32 32768, i32 32768>
  %v7 = add <16 x i32> %v6, zeroinitializer
  %v8 = ashr <16 x i32> %v7, <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  %v9 = add nsw <16 x i32> zeroinitializer, %v8
  %v10 = shl <16 x i32> %v9, <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  %v11 = ashr exact <16 x i32> %v10, <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  %v12 = sub nsw <16 x i32> zeroinitializer, %v11
  %v13 = trunc <16 x i32> %v12 to <16 x i16>
  %v14 = extractelement <16 x i16> %v13, i32 15
  store i16 %v14, i16* %a0, align 2
  ret void
}

attributes #0 = { norecurse nounwind "target-cpu"="hexagonv60" "target-features"="+hvx-length64b,+hvxv60" }
attributes #1 = { norecurse nounwind "target-cpu"="hexagonv60" "target-features"="+hvx-length128b,+hvxv60" }
