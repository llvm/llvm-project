; RUN: llc -march=hexagon < %s
; REQUIRES: asserts

; Splitting live ranges of vector predicate registers (in hexagon-peephole)
; moved a PHI instruction into the middle of another basic block causing a
; crash later on. Make sure this does not happen and that the testcase
; compiles successfully.

target triple = "hexagon"

; Function Attrs: nounwind
define void @f0() local_unnamed_addr #0 {
b0:
  %v0 = icmp eq i32 undef, 0
  br i1 %v0, label %b1, label %b2

b1:                                               ; preds = %b0
  %v1 = tail call <128 x i1> @llvm.hexagon.V6.pred.not.128B(<128 x i1> undef) #2
  br label %b2

b2:                                               ; preds = %b1, %b0
  %v2 = phi <128 x i1> [ %v1, %b1 ], [ undef, %b0 ]
  br label %b3

b3:                                               ; preds = %b3, %b2
  %v3 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %v2, <32 x i32> undef, <32 x i32> undef) #2
  %v4 = tail call <32 x i32> @llvm.hexagon.V6.vor.128B(<32 x i32> undef, <32 x i32> %v3) #2
  %v5 = tail call <32 x i32> @llvm.hexagon.V6.vor.128B(<32 x i32> %v4, <32 x i32> undef) #2
  %v6 = tail call <128 x i1> @llvm.hexagon.V6.vgtub.128B(<32 x i32> %v5, <32 x i32> undef) #2
  %v7 = tail call <128 x i1> @llvm.hexagon.V6.pred.or.128B(<128 x i1> %v6, <128 x i1> undef) #2
  %v8 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %v7, <32 x i32> undef, <32 x i32> undef) #2
  tail call void asm sideeffect "if($0) vmem($1)=$2;", "q,r,v,~{memory}"(<128 x i1> undef, <32 x i32>* undef, <32 x i32> %v8) #2
  br label %b3
}

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1>, <32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <128 x i1> @llvm.hexagon.V6.vgtub.128B(<32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <128 x i1> @llvm.hexagon.V6.pred.or.128B(<128 x i1>, <128 x i1>) #1

; Function Attrs: nounwind readnone
declare <128 x i1> @llvm.hexagon.V6.pred.not.128B(<128 x i1>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vor.128B(<32 x i32>, <32 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length128b" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
