; RUN: llc -mtriple=hexagon -hexagon-expand-condsets=0 < %s | FileCheck %s

; CHECK: cmp.gt
; CHECK-NOT: r1 = p0
; CHECK-NOT: p0 = r1
; CHECK: mux

%s.0 = type { i32 }
%s.1 = type { i64 }

@g0 = common global i16 0, align 2

; Function Attrs: nounwind
define void @f0(ptr nocapture %a0, ptr nocapture %a1, ptr nocapture %a2) #0 {
b0:
  %v0 = load i16, ptr @g0, align 2, !tbaa !0
  %v1 = icmp eq i16 %v0, 3
  %v2 = select i1 %v1, i32 -1, i32 34
  %v4 = load i32, ptr %a0, align 4
  %v5 = zext i32 %v4 to i64
  %v6 = getelementptr inbounds %s.0, ptr %a0, i32 1, i32 0
  %v7 = load i32, ptr %v6, align 4
  %v8 = zext i32 %v7 to i64
  %v9 = shl nuw i64 %v8, 32
  %v10 = or i64 %v9, %v5
  %v12 = load i64, ptr %a1, align 8, !tbaa !4
  %v13 = tail call i64 @llvm.hexagon.M2.vrcmpyr.s0(i64 %v10, i64 %v12)
  %v14 = tail call i64 @llvm.hexagon.S2.asr.i.p(i64 %v13, i32 14)
  %v15 = lshr i64 %v14, 32
  %v16 = trunc i64 %v15 to i32
  %v17 = tail call i32 @llvm.hexagon.C2.cmpgti(i32 %v16, i32 0)
  %v18 = trunc i64 %v14 to i32
  %v19 = tail call i32 @llvm.hexagon.C2.mux(i32 %v17, i32 %v2, i32 %v18)
  %v20 = zext i32 %v19 to i64
  %v21 = getelementptr inbounds %s.1, ptr %a2, i32 2, i32 0
  store i64 %v20, ptr %v21, align 8
  ret void
}

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.M2.vrcmpyr.s0(i64, i64) #1

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.S2.asr.i.p(i64, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.C2.cmpgti(i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.C2.mux(i32, i32, i32) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!0 = !{!1, !1, i64 0}
!1 = !{!"short", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"long long", !2}
