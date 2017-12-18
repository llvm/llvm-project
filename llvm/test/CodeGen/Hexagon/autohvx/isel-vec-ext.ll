; RUN: llc -march=hexagon < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

; CHECK-LABEL: danny:
; CHECK: vsxt
; CHECK-NOT: vinsert
define void @danny() local_unnamed_addr #0 {
b2:
  %v16 = select <16 x i1> undef, <16 x i16> undef, <16 x i16> zeroinitializer
  %v17 = sext <16 x i16> %v16 to <16 x i32>
  store <16 x i32> %v17, <16 x i32>* undef, align 128
  unreachable
}

; CHECK-LABEL: sammy:
; CHECK: vsxt
; CHECK-NOT: vinsert
define void @sammy() local_unnamed_addr #1 {
b2:
  %v16 = select <32 x i1> undef, <32 x i16> undef, <32 x i16> zeroinitializer
  %v17 = sext <32 x i16> %v16 to <32 x i32>
  store <32 x i32> %v17, <32 x i32>* undef, align 128
  unreachable
}


attributes #0 = { noinline norecurse nounwind "target-cpu"="hexagonv60" "target-features"="+hvx-length64b,+hvxv60" }
attributes #1 = { noinline norecurse nounwind "target-cpu"="hexagonv60" "target-features"="+hvx-length128b,+hvxv60" }
