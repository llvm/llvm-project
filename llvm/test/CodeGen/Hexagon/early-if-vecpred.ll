; RUN: llc -march=hexagon < %s | FileCheck %s
; REQUIRES: asserts

; Hexagon early if-conversion used to crash on this testcase due to not
; recognizing vector predicate registers.

target triple = "hexagon"

; Check that the early if-conversion has not happened.

; CHECK-LABEL: fred
; CHECK: q{{[0-3]}} = not
; CHECK: LBB
; CHECK: if (q{{[0-3]}}) vmem
define void @fred(i32 %a0) #0 {
b1:
  %v2 = tail call <1024 x i1> @llvm.hexagon.V6.pred.scalar2.128B(i32 %a0) #2
  br i1 undef, label %b3, label %b5

b3:                                               ; preds = %b1
  %v4 = tail call <1024 x i1> @llvm.hexagon.V6.pred.not.128B(<1024 x i1> %v2) #2
  br label %b5

b5:                                               ; preds = %b3, %b1
  %v6 = phi <1024 x i1> [ %v4, %b3 ], [ %v2, %b1 ]
  %v7 = bitcast <1024 x i1> %v6 to <32 x i32>
  tail call void asm sideeffect "if ($0) vmem($1) = $2;", "q,r,v,~{memory}"(<32 x i32> %v7, <32 x i32>* undef, <32 x i32> undef) #2
  ret void
}

declare <1024 x i1> @llvm.hexagon.V6.pred.scalar2.128B(i32) #1
declare <1024 x i1> @llvm.hexagon.V6.pred.not.128B(<1024 x i1>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-double" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }

