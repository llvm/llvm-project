; RUN: llc -stop-before=expand-isel-pseudos -pre-RA-sched=linearize < %s | FileCheck %s
source_filename = "linear-dbg-value.ll"

; Function Attrs: nounwind readonly uwtable
define i32 @foo(i32* nocapture readonly %a, i32 %N) local_unnamed_addr #0 !dbg !6 {
entry:
  %cmp6 = icmp sgt i32 %N, 0, !dbg !11
  br i1 %cmp6, label %for.body.preheader, label %for.cond.cleanup, !dbg !15

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %N to i64
  br label %for.body, !dbg !17

for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup, !dbg !19

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  %x.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.cond.cleanup.loopexit ]
  ret i32 %x.0.lcssa, !dbg !19

for.body:                                         ; preds = %for.body, %for.body.preheader
; CHECK: ![[X:[0-9]+]] = !DILocalVariable(name: "x",
; CHECK-LABEL: bb.3.for.body:
; CHECK: DBG_VALUE {{.*}} ![[X]], !DIExpression()
; CHECK: DBG_VALUE {{.*}} ![[X]], !DIExpression()
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %x.07 = phi i32 [ %add, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv, !dbg !17
  %0 = load i32, i32* %arrayidx, align 4, !dbg !17
  %add = add nsw i32 %0, %x.07, !dbg !17
  call void @llvm.dbg.value(metadata i32 %add, metadata !9, metadata !DIExpression()), !dbg !20
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !21
  call void @llvm.dbg.value(metadata i32 %add, metadata !9, metadata !DIExpression()), !dbg !20
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count, !dbg !11
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body, !dbg !15
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind readonly uwtable }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 4.0.1 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "foo.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 4.0.1 "}
!6 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 2, type: !7, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !8)
!7 = !DISubroutineType(types: !2)
!8 = !{!9}
!9 = !DILocalVariable(name: "x", scope: !6, file: !1, line: 3, type: !10)
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocation(line: 4, scope: !12)
!12 = !DILexicalBlockFile(scope: !13, file: !1, discriminator: 1)
!13 = distinct !DILexicalBlock(scope: !14, file: !1, line: 4, column: 3)
!14 = distinct !DILexicalBlock(scope: !6, file: !1, line: 4, column: 3)
!15 = !DILocation(line: 4, scope: !16)
!16 = !DILexicalBlockFile(scope: !14, file: !1, discriminator: 1)
!17 = !DILocation(line: 5, scope: !18)
!18 = distinct !DILexicalBlock(scope: !13, file: !1, line: 4, column: 31)
!19 = !DILocation(line: 7, scope: !6)
!20 = !DILocation(line: 3, scope: !6)
!21 = !DILocation(line: 4, scope: !22)
!22 = !DILexicalBlockFile(scope: !13, file: !1, discriminator: 3)
