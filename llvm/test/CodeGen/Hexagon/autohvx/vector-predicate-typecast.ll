; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Check that this compiles successfully.

declare <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32>, i32)
declare <32 x i32> @llvm.hexagon.V6.vandqrt.128B(<128 x i1>, i32)

; The overloaded intrinsic @llvm.hexagon.V6.pred.typecast.128B changes
; the type of the vector predicate. Each intended application needs to be
; declared individually, and they are distinguished by unique suffixes.
; These suffixes don't mean anything.

declare <32 x i1> @llvm.hexagon.V6.pred.typecast.128B.s1(<128 x i1>)
declare <128 x i1> @llvm.hexagon.V6.pred.typecast.128B.s2(<32 x i1>)

; CHECK-LABEL: fred:

; CHECK: r[[R0:[0-9]+]] = #-1
; CHECK: q[[Q0:[0-9]+]] = vand(v0,r[[R0]])
; CHECK: v0 = vand(q[[Q0]],r[[R0]])

define <32 x i32> @fred(<32 x i32> %a0) #0 {
  %q0 = call <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32> %a0, i32 -1)
  %q1 = call <32 x i1> @llvm.hexagon.V6.pred.typecast.128B.s1(<128 x i1> %q0)
  %q2 = call <128 x i1> @llvm.hexagon.V6.pred.typecast.128B.s2(<32 x i1> %q1)
  %v0 = call <32 x i32> @llvm.hexagon.V6.vandqrt.128B(<128 x i1> %q2, i32 -1)
  ret <32 x i32> %v0
}

attributes #0 = { readnone nounwind "target-cpu"="hexagonv66" "target-features"="+hvxv66,+hvx-length128b" }

