; RUN: llc -mtriple=hexagon -mcpu=hexagonv85 -mattr=+hlx -O0 -stop-after=hexagon-lower-hlx-pair-allocas -o - < %s | FileCheck %s
; RUN: llc -mtriple=hexagon -mcpu=hexagonv85 -mattr=+hlx -O0 -o /dev/null < %s

target triple = "hexagon"

; CHECK-LABEL: define <1024 x i32> @direct
; CHECK: %pair = alloca <1024 x i32>, align 2048
; CHECK: call void @llvm.hexagon.LX.SS16b.new(ptr %pair, <1024 x i32> %value), !tbaa ![[TBAA:[0-9]+]], !alias.scope ![[ALIAS:[0-9]+]], !noalias ![[NOALIAS:[0-9]+]], !llvm.access.group ![[GROUP:[0-9]+]]
; CHECK: %[[RESULT:.*]] = call <1024 x i32> @llvm.hexagon.LX.LL16b.tmp(ptr %pair), !tbaa ![[TBAA]], !alias.scope ![[ALIAS]], !noalias ![[NOALIAS]], !llvm.access.group ![[GROUP]]
; CHECK: ret <1024 x i32> %[[RESULT]]
;
define <1024 x i32> @direct(<1024 x i32> %value) {
entry:
  %pair = alloca <1024 x i32>, align 2048
  store <1024 x i32> %value, ptr %pair, align 2048, !tbaa !14, !alias.scope !8, !noalias !10, !llvm.access.group !12
  %result = load <1024 x i32>, ptr %pair, align 2048, !tbaa !14, !alias.scope !8, !noalias !10, !llvm.access.group !12
  ret <1024 x i32> %result
}

; CHECK-LABEL: define <1024 x i32> @derived
; CHECK: %pair = alloca <1024 x i32>, align 2048
; CHECK: call void @llvm.hexagon.LX.SS16b.new(ptr %pair, <1024 x i32> %value)
; CHECK: %[[DERIVED_RESULT:.*]] = call <1024 x i32> @llvm.hexagon.LX.LL16b.tmp(ptr %pair)
; CHECK: ret <1024 x i32> %[[DERIVED_RESULT]]
define <1024 x i32> @derived(<1024 x i32> %value) {
entry:
  %pair = alloca <1024 x i32>, align 2048
  %derived = getelementptr <1024 x i32>, ptr %pair, i32 0
  store <1024 x i32> %value, ptr %derived, align 2048
  %result = load <1024 x i32>, ptr %derived, align 2048
  ret <1024 x i32> %result
}

!7 = distinct !{!7, !"domain"}
!8 = !{!9}
!9 = distinct !{!9, !7, !"alias"}
!10 = !{!11}
!11 = distinct !{!11, !7, !"noalias"}
!12 = distinct !{}
!14 = !{!15, !15, i64 0}
!15 = !{!"omnipotent char", !16, i64 0}
!16 = !{!"Simple C/C++ TBAA"}
