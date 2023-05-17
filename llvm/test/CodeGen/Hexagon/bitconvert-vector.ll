; RUN: llc -march=hexagon < %s | FileCheck %s

; This testcase would fail on a bitcast from v64i16 to v32i32. Check that
; is compiles without errors.
; CHECK: valign
; CHECK: vshuff

target triple = "hexagon"

declare <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32>) #0
declare <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32>, <32 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.valignbi.128B(<32 x i32>, <32 x i32>, i32) #0

define void @fred(ptr %a0, ptr %a1) #1 {
entry:
  %t0 = bitcast <64 x i16> zeroinitializer to <32 x i32>
  %t1 = tail call <32 x i32> @llvm.hexagon.V6.valignbi.128B(<32 x i32> %t0, <32 x i32> undef, i32 2)
  %t2 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> undef, <32 x i32> %t1, i32 -2)
  %t3 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %t2)
  store <64 x i16> zeroinitializer, ptr %a0, align 128
  store <32 x i32> %t3, ptr %a1, align 128
  ret void
}


attributes #0 = { nounwind readnone }
attributes #1 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length128b" }
