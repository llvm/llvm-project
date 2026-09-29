; RUN: llc -mtriple=hexagon < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

; Check that this code only spills a single vector.
; CHECK-NOT: vmem(#r29+{{[^0]}})

%struct.descr = type opaque

define inreg <64 x i32> @danny(ptr %desc, i32 %xy0, i32 %xy1) #0 {
entry:
  %call = tail call inreg <32 x i32> @sammy(ptr %desc, i32 %xy0) #3
  %call1 = tail call inreg <32 x i32> @kirby(ptr %desc, i32 %xy1) #3
  %0 = tail call <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32> %call1, <32 x i32> %call)
  ret <64 x i32> %0
}

declare inreg <32 x i32> @sammy(ptr, i32) #1
declare inreg <32 x i32> @kirby(ptr, i32) #1
declare <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32>, <32 x i32>) #2

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx-length128b,+hvxv60" }
attributes #1 = { "target-cpu"="hexagonv60" "target-features"="+hvx-length128b,+hvxv60" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }
