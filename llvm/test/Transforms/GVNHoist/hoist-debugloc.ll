; RUN: opt < %s -gvn-hoist -S | FileCheck %s
; CHECK: [[VREG:%[^ ]+]] = getelementptr{{.*}}@s1
; CHECK-NOT: !dbg
; CHECK-SAME: {{$}}
; CHECK-NEXT: load{{.*}}[[VREG]]
; CHECK-NOT: !dbg
; CHECK-SAME: {{$}}

target triple = "x86_64-unknown-linux-gnu"

%struct.S1 = type { [10 x i32] }

@s1 = external local_unnamed_addr global [10 x %struct.S1*], align 16

define void @foo(i32 %x, i32 %y, i32 %i) !dbg !4 {
entry:
  %tobool = icmp ne i32 %y, 0, !dbg !8
  br i1 %tobool, label %if.then, label %if.else, !dbg !10

if.then:                                          ; preds = %entry
  %add = add nsw i32 %x, %y, !dbg !11
  %idxprom = sext i32 %i to i64, !dbg !12
  %arrayidx = getelementptr inbounds [10 x %struct.S1*], [10 x %struct.S1*]* @s1, i64 0, i64 %idxprom, !dbg !12
  %0 = load %struct.S1*, %struct.S1** %arrayidx, align 8, !dbg !12, !tbaa !13
  %a = getelementptr inbounds %struct.S1, %struct.S1* %0, i32 0, i32 0, !dbg !17
  br label %if.end, !dbg !12

if.else:                                          ; preds = %entry
  %idxprom3 = sext i32 %i to i64, !dbg !18
  %arrayidx4 = getelementptr inbounds [10 x %struct.S1*], [10 x %struct.S1*]* @s1, i64 0, i64 %idxprom3, !dbg !18
  %1 = load %struct.S1*, %struct.S1** %arrayidx4, align 8, !dbg !18, !tbaa !13
  %a5 = getelementptr inbounds %struct.S1, %struct.S1* %1, i32 0, i32 0, !dbg !19
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %a5.sink = phi [10 x i32]* [ %a5, %if.else ], [ %a, %if.then ]
  %.sink = phi i32 [ %x, %if.else ], [ %add, %if.then ]
  %idxprom6 = sext i32 %i to i64
  %arrayidx7 = getelementptr inbounds [10 x i32], [10 x i32]* %a5.sink, i64 0, i64 %idxprom6
  store i32 %.sink, i32* %arrayidx7, align 4, !tbaa !20
  ret void, !dbg !22
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1)
!1 = !DIFile(filename: "foo.c", directory: "b/")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 7, type: !5, isLocal: false, isDefinition: true, scopeLine: 7, flags: DIFlagPrototyped, isOptimized: true, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{null, !7, !7, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !DILocation(line: 8, column: 7, scope: !9)
!9 = distinct !DILexicalBlock(scope: !4, file: !1, line: 8, column: 7)
!10 = !DILocation(line: 8, column: 7, scope: !4)
!11 = !DILocation(line: 9, column: 18, scope: !9)
!12 = !DILocation(line: 9, column: 5, scope: !9)
!13 = !{!14, !14, i64 0}
!14 = !{!"any pointer", !15, i64 0}
!15 = !{!"omnipotent char", !16, i64 0}
!16 = !{!"Simple C/C++ TBAA"}
!17 = !DILocation(line: 9, column: 9, scope: !9)
!18 = !DILocation(line: 11, column: 5, scope: !9)
!19 = !DILocation(line: 11, column: 9, scope: !9)
!20 = !{!21, !21, i64 0}
!21 = !{!"int", !15, i64 0}
!22 = !DILocation(line: 12, column: 1, scope: !4)
