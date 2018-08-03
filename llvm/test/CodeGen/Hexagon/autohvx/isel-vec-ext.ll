; RUN: llc -march=hexagon < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

; CHECK-LABEL: danny:
; CHECK: vunpack
; CHECK-NOT: vinsert
define void @danny(<16 x i16>* %a0, <16 x i32>* %a1) #0 {
b2:
  %v16 = load <16 x i16>, <16 x i16>* %a0, align 128
  %v17 = sext <16 x i16> %v16 to <16 x i32>
  store <16 x i32> %v17, <16 x i32>* %a1, align 128
  ret void
}

; CHECK-LABEL: sammy:
; CHECK: vunpack
; CHECK-NOT: vinsert
define void @sammy(<32 x i16>* %a0, <32 x i32>* %a1) #1 {
b2:
  %v16 = load <32 x i16>, <32 x i16>* %a0, align 128
  %v17 = sext <32 x i16> %v16 to <32 x i32>
  store <32 x i32> %v17, <32 x i32>* %a1, align 128
  ret void
}


attributes #0 = { noinline norecurse nounwind "target-cpu"="hexagonv60" "target-features"="+hvx-length64b,+hvxv60" }
attributes #1 = { noinline norecurse nounwind "target-cpu"="hexagonv60" "target-features"="+hvx-length128b,+hvxv60" }
