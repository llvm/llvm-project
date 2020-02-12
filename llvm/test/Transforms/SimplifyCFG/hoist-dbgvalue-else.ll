; RUN: opt -simplifycfg -S < %s | FileCheck %s
; Checks that when the if.then and if.else blocks are both eliminated by
; SimplifyCFG, as the common code is hoisted out, the variables with a
; dbg.value in either of those blocks are set undef just before the
; conditional branch instruction.

define i32 @"?fn@@YAHH@Z"(i32 %foo) !dbg !8 {
; CHECK-LABEL: entry:
entry:
  call void @llvm.dbg.value(metadata i32 %foo, metadata !13, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.value(metadata i32 0, metadata !14, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.value(metadata i32 0, metadata !15, metadata !DIExpression()), !dbg !16
  %cmp = icmp eq i32 %foo, 4, !dbg !17
; CHECK: call void @llvm.dbg.value(metadata i32 undef, metadata [[BEARDS:![0-9]+]]
; CHECK: call void @llvm.dbg.value(metadata i32 undef, metadata [[BIRDS:![0-9]+]]
  br i1 %cmp, label %if.then, label %if.else, !dbg !17

if.then:                                          ; preds = %entry
  call void @llvm.dbg.value(metadata i32 8, metadata !14, metadata !DIExpression()), !dbg !16
  br label %if.end, !dbg !18

if.else:                                          ; preds = %entry
  call void @llvm.dbg.value(metadata i32 4, metadata !14, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.value(metadata i32 8, metadata !15, metadata !DIExpression()), !dbg !16
  br label %if.end, !dbg !21

if.end:                                           ; preds = %if.else, %if.then
  %beards.0 = phi i32 [ 8, %if.then ], [ 4, %if.else ], !dbg !23
  call void @llvm.dbg.value(metadata i32 %beards.0, metadata !14, metadata !DIExpression()), !dbg !16
  ret i32 %beards.0, !dbg !24
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

; CHECK-LABEL: }
; CHECK: [[BEARDS]] = !DILocalVariable(name: "beards"
; CHECK: [[BIRDS]] = !DILocalVariable(name: "birds"

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 10.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "C:\5Cdev\5Cllvm-project", checksumkind: CSK_MD5, checksum: "64604a72fdf5b6db8aa2328236bedd6b")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 10.0.0 "}
!8 = distinct !DISubprogram(name: "fn", linkageName: "?fn@@YAHH@Z", scope: !1, file: !1, line: 1, type: !9, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !12)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{!13, !14, !15}
!13 = !DILocalVariable(name: "foo", arg: 1, scope: !8, file: !1, line: 1, type: !11)
!14 = !DILocalVariable(name: "beards", scope: !8, file: !1, line: 2, type: !11)
!15 = !DILocalVariable(name: "birds", scope: !8, file: !1, line: 3, type: !11)
!16 = !DILocation(line: 0, scope: !8)
!17 = !DILocation(line: 5, scope: !8)
!18 = !DILocation(line: 8, scope: !19)
!19 = distinct !DILexicalBlock(scope: !20, file: !1, line: 5)
!20 = distinct !DILexicalBlock(scope: !8, file: !1, line: 5)
!21 = !DILocation(line: 11, scope: !22)
!22 = distinct !DILexicalBlock(scope: !20, file: !1, line: 8)
!23 = !DILocation(line: 0, scope: !20)
!24 = !DILocation(line: 13, scope: !8)
