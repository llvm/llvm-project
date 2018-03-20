; RUN: llc -march=hexagon -enable-pipeliner -debug-only=pipeliner < %s -o - 2>&1 > /dev/null | FileCheck %s
; REQUIRES: asserts

; Fix bug when pipelining xxh benchmark at O3, mv55, and with vectorization.
; The problem is choosing the correct name for the Phis in the epilog.

; CHECK: New block
; CHECK: %{{.*}}, %[[REG:([0-9]+)]]{{.*}} = L2_loadri_pi
; CHECK: epilog:
; CHECK: = PHI
; CHECK-NOT: = PHI %{{[0-9]+}}, {{.*}}, %[[REG]]
; CHECK: = PHI

; Function Attrs: nounwind
define void @f0(i32 %a0, i32* %a1) #0 {
b0:
  %v0 = ashr i32 %a0, 1
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v1 = phi i64 [ %v8, %b1 ], [ undef, %b0 ]
  %v2 = phi i32 [ %v9, %b1 ], [ 0, %b0 ]
  %v3 = phi i32 [ %v7, %b1 ], [ undef, %b0 ]
  %v4 = inttoptr i32 %v3 to i32*
  %v5 = load i32, i32* %v4, align 4, !tbaa !0
  %v6 = tail call i64 @llvm.hexagon.S2.packhl(i32 %v5, i32 undef)
  %v7 = add nsw i32 %v3, -16
  %v8 = tail call i64 @llvm.hexagon.M2.vdmacs.s0(i64 %v1, i64 undef, i64 %v6)
  %v9 = add nsw i32 %v2, 1
  %v10 = icmp eq i32 %v9, %v0
  br i1 %v10, label %b2, label %b1

b2:                                               ; preds = %b1
  %v11 = trunc i64 %v8 to i32
  %v12 = getelementptr inbounds i32, i32* %a1, i32 8
  store i32 %v11, i32* %v12, align 4, !tbaa !0
  call void @llvm.trap()
  unreachable
}

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.M2.vdmacs.s0(i64, i64, i64) #1

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.S2.packhl(i32, i32) #1

; Function Attrs: noreturn nounwind
declare void @llvm.trap() #2

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
attributes #1 = { nounwind readnone }
attributes #2 = { noreturn nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
