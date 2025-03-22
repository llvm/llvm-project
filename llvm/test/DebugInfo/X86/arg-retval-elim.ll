; RUN: opt -S -passes=deadargelim < %s | FileCheck %s
;
; Source code:
;   int tar(int a);
;   __attribute__((noinline)) static int foo(int a, int b)
;   {
;     return tar(a) + tar(a + 1);
;   }
;   int bar(int a)
;   {
;     foo(a, 1);
;     return 0;
;   }

define dso_local noundef i32 @bar(i32 noundef %a) local_unnamed_addr #0 !dbg !10 {
    #dbg_value(i32 %a, !15, !DIExpression(), !16)
  %2 = tail call fastcc i32 @foo(i32 noundef %a, i32 noundef 1), !dbg !17
  ret i32 0, !dbg !18
}

define internal fastcc i32 @foo(i32 noundef %a, i32 noundef %b) unnamed_addr #1 !dbg !19 {
    #dbg_value(i32 %a, !23, !DIExpression(), !25)
    #dbg_value(i32 %b, !24, !DIExpression(), !25)
  %x = tail call i32 @tar(i32 noundef %a), !dbg !26
  %t = add nsw i32 %a, 1, !dbg !27
  %y = tail call i32 @tar(i32 noundef %t), !dbg !28
  %z = add nsw i32 %y, %x, !dbg !29
  ret i32 %z, !dbg !30
}

; CHECK: define internal fastcc void @foo(i32 noundef %a)

declare !dbg !31 i32 @tar(i32 noundef) local_unnamed_addr

attributes #0 = { nounwind }
attributes #1 = { noinline nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 21.0.0git (git@github.com:yonghong-song/llvm-project.git 25cfee009e78194d1f7ca70779d63ef1936cc7b9)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/tests/sig-change/deadret", checksumkind: CSK_MD5, checksum: "728d225e6425c104712ae21cee1db99b")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!9 = !{!"clang version 21.0.0git (git@github.com:yonghong-song/llvm-project.git 25cfee009e78194d1f7ca70779d63ef1936cc7b9)"}
!10 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 6, type: !11, scopeLine: 7, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !14)
!11 = !DISubroutineType(types: !12)
!12 = !{!13, !13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{!15}
!15 = !DILocalVariable(name: "a", arg: 1, scope: !10, file: !1, line: 6, type: !13)
!16 = !DILocation(line: 0, scope: !10)
!17 = !DILocation(line: 8, column: 3, scope: !10)
!18 = !DILocation(line: 9, column: 3, scope: !10)
!19 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 2, type: !20, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !22)

; CHECK: distinct !DISubprogram(name: "foo", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[#]], scopeLine: [[#]], flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized | DISPFlagArgChanged | DISPFlagRetvalRemoved, unit: !0, retainedNodes: ![[#]])

!20 = !DISubroutineType(types: !21)
!21 = !{!13, !13, !13}
!22 = !{!23, !24}
!23 = !DILocalVariable(name: "a", arg: 1, scope: !19, file: !1, line: 2, type: !13)
!24 = !DILocalVariable(name: "b", arg: 2, scope: !19, file: !1, line: 2, type: !13)
!25 = !DILocation(line: 0, scope: !19)
!26 = !DILocation(line: 4, column: 10, scope: !19)
!27 = !DILocation(line: 4, column: 25, scope: !19)
!28 = !DILocation(line: 4, column: 19, scope: !19)
!29 = !DILocation(line: 4, column: 17, scope: !19)
!30 = !DILocation(line: 4, column: 3, scope: !19)
!31 = !DISubprogram(name: "tar", scope: !1, file: !1, line: 1, type: !11, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
