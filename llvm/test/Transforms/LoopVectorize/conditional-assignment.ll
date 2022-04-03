; RUN: opt < %s -enable-cond-stores-vec=false -loop-vectorize -S -pass-remarks-missed='loop-vectorize' -pass-remarks-analysis='loop-vectorize' 2>&1 | FileCheck %s
; RUN: opt < %s -enable-cond-stores-vec=false -passes=loop-vectorize -S -pass-remarks-missed='loop-vectorize' -pass-remarks-analysis='loop-vectorize' 2>&1 | FileCheck %s

; CHECK: remark: source.c:2:8: the cost-model indicates that vectorization is not beneficial

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind ssp uwtable
define void @conditional_store(i32* noalias nocapture %indices) #0 !dbg !4 {
entry:
  br label %for.body, !dbg !10

for.body:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.inc ]
  %arrayidx = getelementptr inbounds i32, i32* %indices, i64 %indvars.iv, !dbg !12
  %0 = load i32, i32* %arrayidx, align 4, !dbg !12, !tbaa !14
  %cmp1 = icmp eq i32 %0, 1024, !dbg !12
  br i1 %cmp1, label %if.then, label %for.inc, !dbg !12

if.then:                                          ; preds = %for.body
  store i32 0, i32* %arrayidx, align 4, !dbg !18, !tbaa !14
  br label %for.inc, !dbg !18

for.inc:                                          ; preds = %for.body, %if.then
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !10
  %exitcond = icmp eq i64 %indvars.iv.next, 4096, !dbg !10
  br i1 %exitcond, label %for.end, label %for.body, !dbg !10

for.end:                                          ; preds = %for.inc
  ret void, !dbg !19
}

attributes #0 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.6.0", isOptimized: true, emissionKind: LineTablesOnly, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "source.c", directory: ".")
!2 = !{}
!4 = distinct !DISubprogram(name: "conditional_store", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 1, file: !1, scope: !5, type: !6, retainedNodes: !2)
!5 = !DIFile(filename: "source.c", directory: ".")
!6 = !DISubroutineType(types: !2)
!7 = !{i32 2, !"Dwarf Version", i32 2}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{!"clang version 3.6.0"}
!10 = !DILocation(line: 2, column: 8, scope: !11)
!11 = distinct !DILexicalBlock(line: 2, column: 3, file: !1, scope: !4)
!12 = !DILocation(line: 3, column: 9, scope: !13)
!13 = distinct !DILexicalBlock(line: 3, column: 9, file: !1, scope: !11)
!14 = !{!15, !15, i64 0}
!15 = !{!"int", !16, i64 0}
!16 = !{!"omnipotent char", !17, i64 0}
!17 = !{!"Simple C/C++ TBAA"}
!18 = !DILocation(line: 3, column: 29, scope: !13)
!19 = !DILocation(line: 4, column: 1, scope: !4)
