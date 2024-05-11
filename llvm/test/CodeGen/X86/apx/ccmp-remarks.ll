; RUN: llc < %s -mattr=+ccmp -pass-remarks=x86-ccmp -pass-remarks-analysis=x86-ccmp -o /dev/null 2>&1 | FileCheck %s
; CHECK-COUNT-3: remark: tmp.c:4:13: convert CMP into conditional CMP
; CHECK: remark: tmp.c:3:0: generate 3 CCMP to eliminate JCC in function foo

define i32 @foo(i32 noundef %a, i32 noundef %b, i32 noundef %c, i32 noundef %d) !dbg !9 {
entry:
  %cmp = icmp slt i32 %a, 1, !dbg !12
  %cmp1 = icmp slt i32 %b, 2
  %or.cond = and i1 %cmp, %cmp1, !dbg !13
  %cmp3 = icmp slt i32 %c, 3
  %or.cond5 = and i1 %or.cond, %cmp3, !dbg !14
  %tobool = icmp ne i32 %d, 0
  %or.cond6 = and i1 %or.cond5, %tobool, !dbg !15
  br i1 %or.cond6, label %if.then, label %return, !dbg !13

if.then:
  %call = tail call i32 (...) @g()
  br label %return

return:
  %retval.0 = phi i32 [ %call, %if.then ], [ 6, %entry ]
  ret i32 %retval.0
}

declare i32 @g(...)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "tmp.c", directory: "/llvm")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{!"clang"}
!9 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 3, type: !10, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!10 = !DISubroutineType(types: !11)
!11 = !{}
!12 = !DILocation(line: 4, column: 9, scope: !9)
!13 = !DILocation(line: 4, column: 13, scope: !9)
!14 = !DILocation(line: 7, column: 12, scope: !9)
!15 = !DILocation(line: 7, column: 5, scope: !9)
