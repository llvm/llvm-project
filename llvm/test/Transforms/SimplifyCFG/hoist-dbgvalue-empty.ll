; RUN: opt -simplifycfg -S < %s | FileCheck %s
; Checks that when the if.then block is eliminated due to containing no
; instructions, the debug intrinsics are hoisted out of the block before its
; deletion. The hoisted intrinsics should have undef values as the branch
; behaviour is unknown to the intrinsics after hoisting.

define dso_local i32 @"?fn@@YAHH@Z"(i32 %foo) local_unnamed_addr !dbg !8 {
entry:
  call void @llvm.dbg.value(metadata i32 %foo, metadata !13, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.value(metadata i32 0, metadata !14, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.value(metadata i32 0, metadata !15, metadata !DIExpression()), !dbg !16
  %cmp = icmp eq i32 %foo, 4, !dbg !17
  br i1 %cmp, label %if.then, label %if.else, !dbg !17

if.then:                                          ; preds = %entry
  call void @llvm.dbg.value(metadata i32 8, metadata !14, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.value(metadata i32 3, metadata !15, metadata !DIExpression()), !dbg !16
  br label %if.end, !dbg !18

if.else:                                          ; preds = %entry
  call void @"?side@@YAXXZ"(), !dbg !21
  call void @llvm.dbg.value(metadata i32 4, metadata !14, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.value(metadata i32 6, metadata !15, metadata !DIExpression()), !dbg !16
  br label %if.end, !dbg !23

; CHECK-LABEL: if.end:
if.end:                                           ; preds = %if.else, %if.then
; CHECK: call void @llvm.dbg.value(metadata i32 undef, metadata [[BEARDS:![0-9]+]]
; CHECK: call void @llvm.dbg.value(metadata i32 undef, metadata [[BIRDS:![0-9]+]]
  %beards.0 = phi i32 [ 8, %if.then ], [ 4, %if.else ], !dbg !24
  call void @llvm.dbg.value(metadata i32 %beards.0, metadata !14, metadata !DIExpression()), !dbg !16
  ret i32 %beards.0, !dbg !25
}

declare dso_local void @"?side@@YAXXZ"() local_unnamed_addr

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

; CHECK: [[BEARDS]] = !DILocalVariable(name: "beards"
; CHECK: [[BIRDS]] = !DILocalVariable(name: "birds"

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 10.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "test2.cpp", directory: "C:\5Cdev\5Cllvm-project", checksumkind: CSK_MD5, checksum: "8ac5d40fcc9914d6479c1a770dfdc176")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 10.0.0 "}
!8 = distinct !DISubprogram(name: "fn", linkageName: "?fn@@YAHH@Z", scope: !1, file: !1, line: 3, type: !9, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !12)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{!13, !14, !15}
!13 = !DILocalVariable(name: "foo", arg: 1, scope: !8, file: !1, line: 3, type: !11)
!14 = !DILocalVariable(name: "beards", scope: !8, file: !1, line: 4, type: !11)
!15 = !DILocalVariable(name: "birds", scope: !8, file: !1, line: 5, type: !11)
!16 = !DILocation(line: 0, scope: !8)
!17 = !DILocation(line: 7, scope: !8)
!18 = !DILocation(line: 10, scope: !19)
!19 = distinct !DILexicalBlock(scope: !20, file: !1, line: 7)
!20 = distinct !DILexicalBlock(scope: !8, file: !1, line: 7)
!21 = !DILocation(line: 11, scope: !22)
!22 = distinct !DILexicalBlock(scope: !20, file: !1, line: 10)
!23 = !DILocation(line: 14, scope: !22)
!24 = !DILocation(line: 0, scope: !20)
!25 = !DILocation(line: 16, scope: !8)
