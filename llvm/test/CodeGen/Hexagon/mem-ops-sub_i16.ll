; RUN: llc -O2 -march=hexagon < %s | FileCheck %s
; Test that we do generate max #u5 in memops.
; CHECK: memh(r{{[0-9]+}}+#0) -= #31

@g0 = unnamed_addr global i16 -32, align 2

; Function Attrs: norecurse nounwind
define fastcc void @f0() unnamed_addr #0 {
b0:
  %v0 = load i16, ptr @g0, align 1, !tbaa !4
  %v1 = zext i16 %v0 to i32
  %v2 = mul nuw nsw i32 %v1, 9625
  %v3 = and i32 %v2, 255
  %v4 = mul nuw nsw i32 %v3, 9625
  %v5 = and i32 %v4, 255
  %v6 = trunc i32 %v5 to i16
  store i16 %v6, ptr @g0, align 2, !tbaa !4
  ret void
}

define i32 @f1() {
b0:
  %v0 = load i16, ptr @g0, align 2, !tbaa !4
  %v1 = zext i16 %v0 to i32
  %v2 = add nuw nsw i32 %v1, 65505
  %v3 = trunc i32 %v2 to i16
  store i16 %v3, ptr @g0, align 2, !tbaa !4
  tail call fastcc void @f0()
  %v4 = load i16, ptr @g0, align 2, !tbaa !4
  %v5 = zext i16 %v4 to i32
  ret i32 %v5
}

attributes #0 = { norecurse nounwind "target-cpu"="hexagonv55" }

!llvm.module.flags = !{!0, !2}

!0 = !{i32 6, !"Target CPU", !1}
!1 = !{!"hexagonv55"}
!2 = !{i32 6, !"Target Features", !3}
!3 = !{!"-hvx"}
!4 = !{!5, !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
