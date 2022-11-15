; RUN: llc -march=hexagon -mcpu=hexagonv68 -mattr=+hvxv68,+hvx-length128b < %s | FileCheck %s

; Do not generate selectI1,Q,Q.
; CHECK: q[[Q:[0-9]+]] = vsetq(r{{[0-9]+}})
; CHECK: q{{[0-9]+}} = and(q{{[0-9]+}},q[[Q]])
; CHECK-NOT: v{{[0-9]+}} = vand(q{{[0-9]+}},r{{[0-9]+}})

target triple = "hexagon"

declare void @llvm.hexagon.V6.vS32b.qpred.ai.128B(<128 x i1>, i8*, <32 x i32>) #0
declare <128 x i1> @llvm.hexagon.V6.pred.scalar2.128B(i32) #1
declare <128 x i1> @llvm.hexagon.V6.pred.and.128B(<128 x i1>, <128 x i1>) #1

define void @libjit_convertFromD32_sm_hf_wrap_3_specialized(i16* %0) local_unnamed_addr #2 {
entry:
  %arrayidx55.i.i = getelementptr inbounds i16, i16* %0, i32 undef
  %1 = ptrtoint i16* %arrayidx55.i.i to i32
  %and.i5.i.i = and i32 %1, 127
  %2 = icmp eq i32 %and.i5.i.i, 127
  %.sroa.speculated.i13.i.i = zext i1 %2 to i32
  %3 = tail call <128 x i1> @llvm.hexagon.V6.pred.scalar2.128B(i32 %.sroa.speculated.i13.i.i) #3
  %4 = tail call <128 x i1> @llvm.hexagon.V6.pred.and.128B(<128 x i1> undef, <128 x i1> %3) #3
  tail call void @llvm.hexagon.V6.vS32b.qpred.ai.128B(<128 x i1> %4, i8* nonnull undef, <32 x i32> undef) #3
  ret void
}

attributes #0 = { nounwind writeonly }
attributes #1 = { nounwind readnone }
attributes #2 = { "use-soft-float"="false" }
attributes #3 = { nounwind }
