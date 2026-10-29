; RUN: opt -S -passes=loop-rotate %s | FileCheck %s

; The old latch ID contains only source locations, while the branch that
; becomes the new latch carries mustprogress.

define void @simplify_latch(ptr %end, ptr %begin) !dbg !9 {
entry:
  br label %header

header:
  %iter = phi ptr [ null, %entry ], [ %next, %latch ]
  %done = icmp eq ptr %iter, %end
  br i1 %done, label %exit, label %latch, !llvm.loop !4

latch:
  %next = getelementptr i8, ptr %begin, i64 8
  br label %header, !llvm.loop !7

exit:
  ret void
}

; CHECK-LABEL: define void @simplify_latch(
; CHECK: br i1 %done, label %exit, label %header, !llvm.loop ![[LOOP:[0-9]+]]
; CHECK: ![[LOOP]] = distinct !{![[LOOP]], {{.*}}![[MUSTPROGRESS:[0-9]+]]}
; CHECK: ![[MUSTPROGRESS]] = !{!"llvm.loop.mustprogress"}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "loop.cpp", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !{!4, !5, !6, !10}
!5 = !DILocation(line: 1, column: 1, scope: !9)
!6 = !DILocation(line: 4, column: 1, scope: !9)
!7 = distinct !{!7, !8, !11}
!8 = !DILocation(line: 2, column: 1, scope: !9)
!9 = distinct !DISubprogram(name: "simplify_latch", scope: !1, file: !1, line: 1, type: !12, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!10 = !{!"llvm.loop.mustprogress"}
!11 = !DILocation(line: 3, column: 1, scope: !9)
!12 = !DISubroutineType(types: !2)
