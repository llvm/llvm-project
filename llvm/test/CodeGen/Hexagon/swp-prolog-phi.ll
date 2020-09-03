; RUN: llc -march=hexagon -rdf-opt=0 < %s -pipeliner-experimental-cg=true | FileCheck %s

; Test that we generate the correct name for a value in a prolog block. The
; pipeliner was using an incorrect value for an instruction in the 2nd prolog
; block for a value defined by a Phi. The result was that an instruction in
; the 1st and 2nd prolog blocks contain the same operands.

; CHECK: vcmp.gt([[VREG:(v[0-9]+)]].uh,v{{[0-9]+}}.uh)
; CHECK-NOT: vcmp.gt([[VREG]].uh,v{{[0-9]+}}.uh)
; CHECK: loop0

define void @f0(<64 x i32> %a0, <32 x i32> %a1, i32 %a2) #0 {
b0:
  br i1 undef, label %b1, label %b5

b1:                                               ; preds = %b0
  %v0 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %a0)
  br label %b2

b2:                                               ; preds = %b4, %b1
  %v1 = phi <32 x i32> [ %a1, %b1 ], [ %v7, %b4 ]
  br label %b3

b3:                                               ; preds = %b3, %b2
  %v2 = phi i32 [ 0, %b2 ], [ %v8, %b3 ]
  %v3 = phi <32 x i32> [ zeroinitializer, %b2 ], [ %v0, %b3 ]
  %v4 = phi <32 x i32> [ %v1, %b2 ], [ %v7, %b3 ]
  %v5 = tail call <128 x i1> @llvm.hexagon.V6.vgtuh.128B(<32 x i32> %v3, <32 x i32> undef)
  %v6 = tail call <128 x i1> @llvm.hexagon.V6.veqh.and.128B(<128 x i1> %v5, <32 x i32> undef, <32 x i32> undef)
  %v7 = tail call <32 x i32> @llvm.hexagon.V6.vaddhq.128B(<128 x i1> %v6, <32 x i32> %v4, <32 x i32> undef)
  %v8 = add nsw i32 %v2, 1
  %v9 = icmp slt i32 %v8, %a2
  br i1 %v9, label %b3, label %b4

b4:                                               ; preds = %b3
  br i1 undef, label %b5, label %b2

b5:                                               ; preds = %b4, %b0
  ret void
}

; Function Attrs: nounwind readnone
declare <128 x i1> @llvm.hexagon.V6.vgtuh.128B(<32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <128 x i1> @llvm.hexagon.V6.veqh.and.128B(<128 x i1>, <32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vaddhq.128B(<128 x i1>, <32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length128b" }
attributes #1 = { nounwind readnone }
