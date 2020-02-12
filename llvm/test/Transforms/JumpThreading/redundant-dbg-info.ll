; RUN: opt -jump-threading -S < %s | FileCheck %s

define dso_local i32 @_Z3fooi(i32 %a) !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata i32 %a, metadata !12, metadata !DIExpression()), !dbg !13
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !13
  %tobool = icmp ne i32 %a, 0, !dbg !15
  br i1 %tobool, label %if.then, label %if.end, !dbg !17

if.then:                                          ; preds = %entry
  call void @_Z4callv(), !dbg !18
  call void @llvm.dbg.value(metadata i32 0, metadata !14, metadata !DIExpression()), !dbg !13
  br label %if.end, !dbg !20

if.end:                                           ; preds = %if.then, %entry
  %c.0 = phi i32 [ 0, %if.then ], [ 1, %entry ], !dbg !13
  call void @llvm.dbg.value(metadata i32 %c.0, metadata !14, metadata !DIExpression()), !dbg !13
  %tobool1 = icmp ne i32 %c.0, 0, !dbg !21
  br i1 %tobool1, label %if.else, label %if.then2, !dbg !23

; CHECK-LABEL: if.then2:
; CHECK: call void @llvm.dbg.value({{.+}}, metadata ![[B:[0-9]+]], metadata !DIExpression())
; CHECK: call void @llvm.dbg.value({{.+}}, metadata ![[B:[0-9]+]], metadata !DIExpression())
; CHECK-NOT: call void @llvm.dbg.value({{.+}}, metadata ![[B]], metadata !DIExpression())
if.then2:                                         ; preds = %if.end
  call void @llvm.dbg.value(metadata i32 4, metadata !24, metadata !DIExpression()), !dbg !13
  br label %if.end3, !dbg !25

; CHECK-LABEL: if.end3:
if.else:                                          ; preds = %if.end
  call void @llvm.dbg.value(metadata i32 6, metadata !24, metadata !DIExpression()), !dbg !13
  br label %if.end3

if.end3:                                          ; preds = %if.else, %if.then2
  %b.0 = phi i32 [ 6, %if.else ], [ 4, %if.then2 ], !dbg !27
  call void @llvm.dbg.value(metadata i32 %b.0, metadata !24, metadata !DIExpression()), !dbg !13
  ret i32 %b.0, !dbg !28
}
; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare dso_local void @_Z4callv()
; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

; CHECK: ![[B]] = !DILocalVariable(name: "b"
!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 10.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 10.0.0"}
!7 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooi", scope: !8, file: !8, line: 3, type: !9, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DIFile(filename: "./test.cpp", directory: "/")
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DILocalVariable(name: "a", arg: 1, scope: !7, file: !8, line: 3, type: !11)
!13 = !DILocation(line: 0, scope: !7)
!14 = !DILocalVariable(name: "c", scope: !7, file: !8, line: 4, type: !11)
!15 = !DILocation(line: 5, column: 7, scope: !16)
!16 = distinct !DILexicalBlock(scope: !7, file: !8, line: 5, column: 7)
!17 = !DILocation(line: 5, column: 7, scope: !7)
!18 = !DILocation(line: 6, column: 5, scope: !19)
!19 = distinct !DILexicalBlock(scope: !16, file: !8, line: 5, column: 10)
!20 = !DILocation(line: 8, column: 3, scope: !19)
!21 = !DILocation(line: 9, column: 8, scope: !22)
!22 = distinct !DILexicalBlock(scope: !7, file: !8, line: 9, column: 7)
!23 = !DILocation(line: 9, column: 7, scope: !7)
!24 = !DILocalVariable(name: "b", scope: !7, file: !8, line: 4, type: !11)
!25 = !DILocation(line: 11, column: 3, scope: !26)
!26 = distinct !DILexicalBlock(scope: !22, file: !8, line: 9, column: 11)
!27 = !DILocation(line: 0, scope: !22)
!28 = !DILocation(line: 14, column: 3, scope: !7)
