; RUN: llc -march=hexagon < %s | FileCheck %s
; Check for some sane output (original problem was a crash).
; CHECK: DEBUG_VALUE: fred:Count <- 0

target triple = "hexagon"

define i32 @fred(i32 %p) local_unnamed_addr #0 !dbg !6 {
entry:
  br label %cond.end

cond.end:                                         ; preds = %entry
  br i1 undef, label %cond.false.i, label %for.body.lr.ph.i

for.body.lr.ph.i:                                 ; preds = %cond.end
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !10, metadata !12) #0, !dbg !13
  br label %for.body.i

cond.false.i:                                     ; preds = %cond.end
  unreachable

for.body.i:                                       ; preds = %for.inc.i, %for.body.lr.ph.i
  %inc.sink37.i = phi i32 [ 0, %for.body.lr.ph.i ], [ %inc.i, %for.inc.i ]
  %call.i = tail call i8* undef(i32 12, i8* undef) #0
  br label %for.inc.i

for.inc.i:                                        ; preds = %for.body.i
  %inc.i = add nuw i32 %inc.sink37.i, 1
  %cmp1.i = icmp ult i32 %inc.i, %p
  br i1 %cmp1.i, label %for.body.i, label %PQ_AllocMem.exit.loopexit

PQ_AllocMem.exit.loopexit:                        ; preds = %for.inc.i
  unreachable
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 4.0.0 (http://llvm.org/git/clang.git 37afcb099ac2b001f4c826da7ca1d077b67a508c) (http://llvm.org/git/llvm.git 5887f1c75b3ba216850c834b186efdd3e54b7d4f)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2)
!1 = !DIFile(filename: "file.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 4.0.0 (http://llvm.org/git/clang.git 37afcb099ac2b001f4c826da7ca1d077b67a508c) (http://llvm.org/git/llvm.git 5887f1c75b3ba216850c834b186efdd3e54b7d4f)"}
!6 = distinct !DISubprogram(name: "fred", scope: !1, file: !1, line: 116, type: !7, isLocal: false, isDefinition: true, scopeLine: 121, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !9)
!7 = !DISubroutineType(types: !2)
!8 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !{!10}
!10 = !DILocalVariable(name: "Count", scope: !6, file: !1, line: 1, type: !8)
!11 = distinct !DILocation(line: 1, column: 1, scope: !6)
!12 = !DIExpression()
!13 = !DILocation(line: 1, column: 1, scope: !6, inlinedAt: !11)
