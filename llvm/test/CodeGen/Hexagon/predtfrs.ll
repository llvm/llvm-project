; RUN: llc -march=hexagon -hexagon-expand-condsets=0 < %s | FileCheck %s

; CHECK: cmp.gt
; CHECK-NOT: r1 = p0
; CHECK-NOT: p0 = r1
; CHECK: mux

%s.0 = type { i32 }
%s.1 = type { i64 }

@g0 = common global i16 0, align 2

; Function Attrs: nounwind
define void @f0(%s.0* nocapture %a0, %s.1* nocapture %a1, %s.1* nocapture %a2) #0 {
b0:
  %v0 = load i16, i16* @g0, align 2, !tbaa !0
  %v1 = icmp eq i16 %v0, 3
  %v2 = select i1 %v1, i32 -1, i32 34
  %v3 = getelementptr inbounds %s.0, %s.0* %a0, i32 0, i32 0
  %v4 = load i32, i32* %v3, align 4
  %v5 = zext i32 %v4 to i64
  %v6 = getelementptr inbounds %s.0, %s.0* %a0, i32 1, i32 0
  %v7 = load i32, i32* %v6, align 4
  %v8 = zext i32 %v7 to i64
  %v9 = shl nuw i64 %v8, 32
  %v10 = or i64 %v9, %v5
  %v11 = getelementptr inbounds %s.1, %s.1* %a1, i32 0, i32 0
  %v12 = load i64, i64* %v11, align 8, !tbaa !4
  %v13 = tail call i64 @llvm.hexagon.M2.vrcmpyr.s0(i64 %v10, i64 %v12)
  %v14 = tail call i64 @llvm.hexagon.S2.asr.i.p(i64 %v13, i32 14)
  %v15 = lshr i64 %v14, 32
  %v16 = trunc i64 %v15 to i32
  %v17 = tail call i32 @llvm.hexagon.C2.cmpgti(i32 %v16, i32 0)
  %v18 = trunc i64 %v14 to i32
  %v19 = tail call i32 @llvm.hexagon.C2.mux(i32 %v17, i32 %v2, i32 %v18)
  %v20 = zext i32 %v19 to i64
  %v21 = getelementptr inbounds %s.1, %s.1* %a2, i32 2, i32 0
  store i64 %v20, i64* %v21, align 8
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
