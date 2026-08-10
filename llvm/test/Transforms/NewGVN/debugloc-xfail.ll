; XFAIL: *
; RUN: opt < %s -passes=newgvn -S | FileCheck %s
; CHECK: {{^}}for.body:
; CHECK-NEXT: [[VREG1:%[^ ]+]] = phi{{.*}}[[VREG2:%[^ ]+]],{{.*}}%.sink,
; CHECK-NOT: !dbg
; CHECK-SAME: {{$}}
; CHECK: {{^}}for.inc:
; CHECK-NEXT: [[VREG2]] = phi{{.*}}%inc,{{.*}}[[VREG1]]

target triple = "x86_64-unknown-linux-gnu"

@g = external local_unnamed_addr global i32, align 4

; Function Attrs: nounwind uwtable
define void @foo(i32 %x, i32 %y, i32 %z) local_unnamed_addr #0 !dbg !4 {
entry:
  %not.tobool = icmp eq i32 %x, 0, !dbg !8
  %.sink = zext i1 %not.tobool to i32, !dbg !8
  store i32 %.sink, ptr @g, align 4, !tbaa !9
  %cmp8 = icmp sgt i32 %y, 0, !dbg !13
  br i1 %cmp8, label %for.body.preheader, label %for.end, !dbg !17

for.body.preheader:                               ; preds = %entry
  br label %for.body, !dbg !19

for.body:                                         ; preds = %for.body.preheader, %for.inc
  %i.09 = phi i32 [ %inc4, %for.inc ], [ 0, %for.body.preheader ]
  %cmp1 = icmp sgt i32 %i.09, %z, !dbg !19
  br i1 %cmp1, label %if.then2, label %for.inc, !dbg !21

if.then2:                                         ; preds = %for.body
  %0 = load i32, ptr @g, align 4, !dbg !22, !tbaa !9
  %inc = add nsw i32 %0, 1, !dbg !22
  store i32 %inc, ptr @g, align 4, !dbg !22, !tbaa !9
  br label %for.inc, !dbg !23

for.inc:                                          ; preds = %for.body, %if.then2
  %inc4 = add nuw nsw i32 %i.09, 1, !dbg !24
  %exitcond = icmp ne i32 %inc4, %y, !dbg !13
  br i1 %exitcond, label %for.body, label %for.end.loopexit, !dbg !17

for.end.loopexit:                                 ; preds = %for.inc
  br label %for.end, !dbg !26

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void, !dbg !26
}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1)
!1 = !DIFile(filename: "foo.c", directory: "b/")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 3, type: !5, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{null, !7, !7, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !DILocation(line: 4, column: 7, scope: !4)
!9 = !{!10, !10, i64 0}
!10 = !{!"int", !11, i64 0}
!11 = !{!"omnipotent char", !12, i64 0}
!12 = !{!"Simple C/C++ TBAA"}
!13 = !DILocation(line: 10, column: 13, scope: !14)
!14 = !DILexicalBlockFile(scope: !15, file: !1, discriminator: 1)
!15 = distinct !DILexicalBlock(scope: !16, file: !1, line: 10, column: 3)
!16 = distinct !DILexicalBlock(scope: !4, file: !1, line: 10, column: 3)
!17 = !DILocation(line: 10, column: 3, scope: !18)
!18 = !DILexicalBlockFile(scope: !16, file: !1, discriminator: 1)
!19 = !DILocation(line: 11, column: 11, scope: !20)
!20 = distinct !DILexicalBlock(scope: !15, file: !1, line: 11, column: 9)
!21 = !DILocation(line: 11, column: 9, scope: !15)
!22 = !DILocation(line: 12, column: 8, scope: !20)
!23 = !DILocation(line: 12, column: 7, scope: !20)
!24 = !DILocation(line: 10, column: 20, scope: !25)
!25 = !DILexicalBlockFile(scope: !15, file: !1, discriminator: 2)
!26 = !DILocation(line: 13, column: 1, scope: !4)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
