; RUN: opt < %s -passes='pseudo-probe-update' -S  | FileCheck %s

declare i32 @f1()

declare i32 @f2()

declare void @f3()

define i32 @foo(i1 %cond, i1 %cond2) !dbg !4 !prof !10 {
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 1, i32 0, i64 -1)
  br i1 %cond, label %T1, label %Merge, !prof !11

T1:                                               ; preds = %0
  %v1 = call i32 @f1(), !prof !12
  %cond3 = icmp eq i32 %v1, 412
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 2, i32 0, i64 -1)
;; The distribution factor -9223372036854775808 stands for 53.85%, whic is from 7/6+7.
; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID:]], i64 4, i32 0, i64 -9223372036854775808)
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 4, i32 0, i64 -1), !dbg !13
;; Probe 7 has two copies, since they don't share the same inline context, they are not
;; considered sharing samples, thus their distribution factors are not fixed up.
; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID]], i64 7, i32 0, i64 -1)
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 7, i32 0, i64 -1), !dbg !13
;; Similar to Probe 7, one copy of Probe 8 doesn't have inline context.
; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID]], i64 8, i32 0, i64 -1)
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 8, i32 0, i64 -1), !dbg !13
  br i1 %cond3, label %T2, label %F2, !prof !11

Merge:                                            ; preds = %0
  %v2 = call i32 @f2(), !prof !12
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 3, i32 0, i64 -1)
;; The distribution factor  -9223372036854775808 stands for 46.25%, which is from 6/6+7.
; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID]], i64 4, i32 0, i64  -9223372036854775808)
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 4, i32 0, i64 8513881922462547968), !dbg !13
; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID]], i64 7, i32 0, i64 -1)
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 7, i32 0, i64 -1), !dbg !18
; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID]], i64 8, i32 0, i64 -1)
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 8, i32 0, i64 -1), !dbg !19 
  br i1 %cond2, label %T2, label %F2, !prof !11

T2:                                               ; preds = %Merge, %T1
  %B1 = phi i32 [ %v1, %T1 ], [ %v2, %Merge ]
  call void @f3(), !prof !12
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 5, i32 0, i64 -1)
  ret i32 %B1

F2:                                               ; preds = %Merge, %T1
  %B2 = phi i32 [ %v1, %T1 ], [ %v2, %Merge ]
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 6, i32 0, i64 -1)
  ret i32 %B2
}

; Function Attrs: inaccessiblememonly nounwind willreturn
declare void @llvm.pseudoprobe(i64, i64, i32, i64) #0

attributes #0 = { inaccessiblememonly nounwind willreturn }

!llvm.module.flags = !{!0, !1}
!llvm.pseudo_probe_desc = !{!2, !3}

!0 = !{i32 7, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !{i64 6699318081062747564, i64 281479271677951, !"foo", null}
!3 = !{i64 6468398850841090686, i64 138828622701, !"zen", null}
!4 = distinct !DISubprogram(name: "foo", scope: !5, file: !5, line: 9, type: !6, scopeLine: 9, spFlags: DISPFlagDefinition, unit: !9)
!5 = !DIFile(filename: "test.cpp", directory: "test")
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !8}
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !5, isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug)
!10 = !{!"function_entry_count", i64 14}
!11 = !{!"branch_weights", i32 8, i32 7}
!12 = !{!"branch_weights", i32 7}
!13 = !DILocation(line: 39, column: 9, scope: !14, inlinedAt: !16)
!14 = distinct !DILexicalBlock(scope: !15, file: !5, line: 39, column: 7)
!15 = distinct !DISubprogram(name: "zen", scope: !5, file: !5, line: 37, type: !6, scopeLine: 38, spFlags: DISPFlagDefinition, unit: !9)
!16 = distinct !DILocation(line: 10, column: 11, scope: !17)
!17 = !DILexicalBlockFile(scope: !4, file: !5, discriminator: 186646551)
!18 = !DILocation(line: 53, column: 3, scope: !15, inlinedAt: !19)
!19 = !DILocation(line: 12, column: 3, scope: !4)
