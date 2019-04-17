; RUN: opt < %s -S -partial-inliner -skip-partial-inlining-cost-analysis=true | FileCheck %s

; CHECK-LABEL: @callee
; CHECK: %mul = mul nsw i32 %v, 10, !dbg ![[DBG1:[0-9]+]]
define i32 @callee(i32 %v, ...) !dbg !16 {
entry:
  %cmp = icmp sgt i32 %v, 2000, !dbg !17
  br i1 %cmp, label %if.then, label %if.end, !dbg !19

if.then:                                          ; preds = %entry
  %mul = mul nsw i32 %v, 10, !dbg !20
  br label %if.end, !dbg !21

if.end:                                           ; preds = %if.then, %entry
  %v2 = phi i32 [ %v, %entry ], [ %mul, %if.then ]
  %add = add nsw i32 %v2, 200, !dbg !22
  ret i32 %add, !dbg !23
}

; CHECK-LABEL: @caller
; CHECK: codeRepl.i:
; CHECK-NOT: br label
; CHECK: call void (i32, i32*, ...) @callee.1.if.then(i32 %v, i32* %mul.loc.i, i32 99), !dbg ![[DBG2:[0-9]+]]
define i32 @caller(i32 %v) !dbg !8 {
entry:
  %call = call i32 (i32, ...) @callee(i32 %v, i32 99), !dbg !14
  ret i32 %call, !dbg !15
}

; CHECK-LABEL: define internal void @callee.1.if.then
; CHECK: br label %if.then, !dbg ![[DBG3:[0-9]+]]

; CHECK: ![[DBG1]] = !DILocation(line: 10, column: 7,
; CHECK: ![[DBG2]] = !DILocation(line: 10, column: 7,
; CHECK: ![[DBG3]] = !DILocation(line: 10, column: 7,


!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0 (trunk 177881)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 1, !"min_enum_size", i32 4}
!7 = !{!"clang version 6.0.0"}
!8 = distinct !DISubprogram(name: "caller", scope: !1, file: !1, line: 3, type: !9, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !12)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !11}
!11 = !DIBasicType(name: "int", size: 19, encoding: DW_ATE_signed)
!12 = !{!13}
!13 = !DILocalVariable(name: "v", arg: 1, scope: !8, file: !1, line: 3, type: !11)
!14 = !DILocation(line: 5, column: 10, scope: !8)
!15 = !DILocation(line: 5, column: 3, scope: !8)
!16 = distinct !DISubprogram(name: "callee", scope: !1, file: !1, line: 8, type: !9, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !26)
!26 = !{!27}
!27 = !DILocalVariable(name: "v", arg: 1, scope: !16, file: !1, line: 8, type: !11)
!17 = !DILocation(line: 9, column: 9, scope: !18)
!18 = distinct !DILexicalBlock(scope: !16, file: !1, line: 9, column: 7)
!19 = !DILocation(line: 9, column: 7, scope: !16)
!20 = !DILocation(line: 10, column: 7, scope: !18)
!21 = !DILocation(line: 10, column: 5, scope: !18)
!22 = !DILocation(line: 11, column: 5, scope: !16)
!36 = !DILocation(line: 12, column: 10, scope: !16)
!23 = !DILocation(line: 12, column: 3, scope: !16)
