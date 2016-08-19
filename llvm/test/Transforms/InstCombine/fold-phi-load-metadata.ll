; RUN: opt -instcombine -S < %s | FileCheck %s

@g1 = common global i32* null, align 8

%struct.S1 = type { i32, float }
%struct.S2 = type { float, i32 }

; Check that instcombine preserves metadata when it merges two loads.
;
; CHECK: return:
; CHECK: load i32*, i32** %{{[a-z0-9.]+}}, align 8, !nonnull ![[EMPTYNODE:[0-9]+]]
; CHECK: load i32, i32* %{{[a-z0-9.]+}}, align 4, !tbaa ![[TBAA:[0-9]+]], !range ![[RANGE:[0-9]+]], !invariant.load ![[EMPTYNODE:[0-9]+]], !alias.scope ![[ALIAS_SCOPE:[0-9]+]], !noalias ![[NOALIAS:[0-9]+]]

; Function Attrs: nounwind ssp uwtable
define i32 @phi_load_metadata(%struct.S1* %s1, %struct.S2* %s2, i32 %c, i32** %x0, i32 **%x1) #0 {
entry:
  %tobool = icmp eq i32 %c, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %i = getelementptr inbounds %struct.S2, %struct.S2* %s2, i64 0, i32 1
  %val = load i32, i32* %i, align 4, !tbaa !0, !alias.scope !13, !noalias !14, !invariant.load !17, !range !18
  %p0 = load i32*, i32** %x0, align 8, !nonnull !17
  br label %return

if.end:                                           ; preds = %entry
  %i2 = getelementptr inbounds %struct.S1, %struct.S1* %s1, i64 0, i32 0
  %val2 = load i32, i32* %i2, align 4, !tbaa !2, !alias.scope !15, !noalias !16, !invariant.load !17, !range !19
  %p1 = load i32*, i32** %x1, align 8, !nonnull !17
  br label %return

return:                                           ; preds = %if.end, %if.then
  %retval = phi i32 [ %val, %if.then ], [ %val2, %if.end ]
  %pval = phi i32* [ %p0, %if.then ], [ %p1, %if.end ]
  store i32* %pval, i32** @g1, align 8
  ret i32 %retval
}

; CHECK: ![[EMPTYNODE]] = !{}
; CHECK: ![[TBAA]] = !{![[TAG1:[0-9]+]], ![[TAG1]], i64 0}
; CHECK: ![[TAG1]] = !{!"int", !{{[0-9]+}}, i64 0}
; CHECK: ![[RANGE]] = !{i32 10, i32 25}
; CHECK: ![[ALIAS_SCOPE]] = !{![[SCOPE0:[0-9]+]], ![[SCOPE2:[0-9]+]], ![[SCOPE1:[0-9]+]]}
; CHECK: ![[SCOPE0]] = distinct !{![[SCOPE0]], !{{[0-9]+}}, !"scope0"}
; CHECK: ![[SCOPE2]] = distinct !{![[SCOPE2]], !{{[0-9]+}}, !"scope2"}
; CHECK: ![[SCOPE1]] = distinct !{![[SCOPE1]], !{{[0-9]+}}, !"scope1"}
; CHECK: ![[NOALIAS]] = !{![[SCOPE3:[0-9]+]]}
; CHECK: ![[SCOPE3]] = distinct !{![[SCOPE3]], !{{[0-9]+}}, !"scope3"}

!0 = !{!1, !4, i64 4}
!1 = !{!"", !7, i64 0, !4, i64 4}
!2 = !{!3, !4, i64 0}
!3 = !{!"", !4, i64 0, !7, i64 4}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!7 = !{!"float", !5, i64 0}
!8 = !{!8, !"some domain"}
!9 = !{!9, !8, !"scope0"}
!10 = !{!10, !8, !"scope1"}
!11 = !{!11, !8, !"scope2"}
!12 = !{!12, !8, !"scope3"}
!13 = !{!9, !10}
!14 = !{!11, !12}
!15 = !{!9, !11}
!16 = !{!10, !12}
!17 = !{}
!18 = !{i32 10, i32 20}
!19 = !{i32 15, i32 25}
