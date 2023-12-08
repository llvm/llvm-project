; RUN: llvm-as < %s -o - | llvm-dis - | FileCheck %s

; The loop metadata in this test is similar to the output from clang for
; code like the this:
; 
;     #pragma clang loop unroll_count(4)
;     #pragma clang loop vectorize(assume_safety)
;     for(int i = 0; i < n; i++)
;       ...
;
; strip debug should not remove the useful metadata in the followup part.

declare hidden void @g() local_unnamed_addr align 2

define hidden void @f() {
; CHECK: tail call {{.*}} !llvm.loop [[LID:![0-9]+]]
  tail call void @g(), !llvm.loop !2
  ret void
}

!llvm.module.flags = !{!0, !1}

; CHECK-NOT: DILocation
; CHECK: [[LID]] = distinct !{[[LID]], [[VE:![0-9]+]], [[VF:![0-9]+]]}
; CHECK-NOT: DILocation
; CHECK: [[VE]] = !{!"llvm.loop.vectorize.enable", i1 true}
; CHECK: [[VF]] = !{!"llvm.loop.vectorize.followup_all", [[FU:![0-9]+]]}
; CHECK: [[FU]] = distinct !{[[FU]], [[VD:![0-9]+]], [[UC:![0-9]+]]}
; CHECK: [[VD]] = !{!"llvm.loop.isvectorized"}
; CHECK: [[UC]] = !{!"llvm.loop.unroll.count", i32 4}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !{!2, !3, !10, !11, !12}
!3 = !DILocation(line: 5, column: 12, scope: !4)
!4 = distinct !DILexicalBlock(scope: !6, file: !5, line: 105, column: 3)
!5 = !DIFile(filename: "/", directory: "f.cpp")
!6 = distinct !DISubprogram(name: "f", scope: !5, file: !5, line: 1324, type: !7, scopeLine: 1324, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !9)
!7 = !DISubroutineType(types: !8)
!8 = !{}
!9 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !5, isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug)
!10 = !DILocation(line: 6, column: 1, scope: !4)
!11 = !{!"llvm.loop.vectorize.enable", i1 true}
!12 = !{!"llvm.loop.vectorize.followup_all", !13}
!13 = distinct !{!13, !3, !10, !14, !15}
!14 = !{!"llvm.loop.isvectorized"}
!15 = !{!"llvm.loop.unroll.count", i32 4}
