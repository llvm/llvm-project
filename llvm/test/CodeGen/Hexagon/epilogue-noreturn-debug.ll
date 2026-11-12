; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; CHECK-LABEL: CuSuiteAddSuite:
; CHECK: call _Assert
; CHECK-NOT: deallocframe

source_filename = "epilogue-noreturn-debug.ll"

%struct.CuSuite.2.29.32.38.41.44.53.56.68.86.112 = type { i32, [1024 x ptr], i32 }

@.str24 = external unnamed_addr constant [140 x i8], align 8

declare void @_Assert()

; Function Attrs: nounwind null_pointer_is_valid
define void @CuSuiteAddSuite(ptr %p, i1 %cond) #0 !dbg !5 {
entry:
  br i1 %cond, label %for.body.us, label %for.end, !dbg !16

for.body.us:                                      ; preds = %entry
  %0 = load ptr, ptr %p, align 4, !dbg !17
  %1 = load i32, ptr %p, align 4, !dbg !18
  %cmp.i.us = icmp slt i32 %1, 1024, !dbg !19
  br i1 %cmp.i.us, label %CuSuiteAdd.exit.us, label %cond.false6.i.us, !dbg !20

cond.false6.i.us:                                 ; preds = %for.body.us
    #dbg_value(ptr getelementptr inbounds (i8, ptr @.str24, i64 8), !9, !DIExpression(), !17)
  tail call void @_Assert() #2, !dbg !21
  unreachable, !dbg !22

CuSuiteAdd.exit.us:                               ; preds = %for.body.us
  %arrayidx.i.us = getelementptr inbounds %struct.CuSuite.2.29.32.38.41.44.53.56.68.86.112, ptr null, i32 0, i32 1, i32 %1, !dbg !23
  store ptr %0, ptr %arrayidx.i.us, align 4, !dbg !24
  call void @llvm.trap(), !dbg !25
  unreachable, !dbg !26

for.end:                                          ; preds = %entry
  ret void, !dbg !27
}

; Function Attrs: cold noreturn nounwind memory(inaccessiblemem: write)
declare void @llvm.trap() #1

attributes #0 = { nounwind null_pointer_is_valid }
attributes #1 = { cold noreturn nounwind memory(inaccessiblemem: write) }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "epilogue-noreturn-debug.ll", directory: "/")
!2 = !{i32 12}
!3 = !{i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "CuSuiteAddSuite", linkageName: "CuSuiteAddSuite", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !{!9, !11, !13, !15}
!9 = !DILocalVariable(name: "1", scope: !5, file: !1, line: 2, type: !10)
!10 = !DIBasicType(name: "ty64", size: 64, encoding: DW_ATE_unsigned)
!11 = !DILocalVariable(name: "2", scope: !5, file: !1, line: 3, type: !12)
!12 = !DIBasicType(name: "ty32", size: 32, encoding: DW_ATE_unsigned)
!13 = !DILocalVariable(name: "3", scope: !5, file: !1, line: 4, type: !14)
!14 = !DIBasicType(name: "ty8", size: 8, encoding: DW_ATE_unsigned)
!15 = !DILocalVariable(name: "4", scope: !5, file: !1, line: 8, type: !10)
!16 = !DILocation(line: 1, column: 1, scope: !5)
!17 = !DILocation(line: 2, column: 1, scope: !5)
!18 = !DILocation(line: 3, column: 1, scope: !5)
!19 = !DILocation(line: 4, column: 1, scope: !5)
!20 = !DILocation(line: 5, column: 1, scope: !5)
!21 = !DILocation(line: 6, column: 1, scope: !5)
!22 = !DILocation(line: 7, column: 1, scope: !5)
!23 = !DILocation(line: 8, column: 1, scope: !5)
!24 = !DILocation(line: 9, column: 1, scope: !5)
!25 = !DILocation(line: 10, column: 1, scope: !5)
!26 = !DILocation(line: 11, column: 1, scope: !5)
!27 = !DILocation(line: 12, column: 1, scope: !5)
