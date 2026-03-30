; RUN: llc -mtriple=x86_64-unknown-linux-gnu -stop-after=branch-folder -o - %s | FileCheck %s

; CHECK: TCRETURNdi64cc {{.*}} debug-location !DILocation(line: 0, scope:

define dso_local void @_Z3fooi(i32 noundef %x) !dbg !13 {
entry:
    #dbg_value(i32 %x, !18, !DIExpression(), !19)
  %cmp = icmp eq i32 %x, 42, !dbg !20
  br i1 %cmp, label %if.then, label %if.end, !dbg !22

if.then:                                          ; preds = %entry
  tail call void @_Z3barv(), !dbg !23
  br label %if.end, !dbg !25

if.end:                                           ; preds = %if.then, %entry
  ret void, !dbg !26
}

declare !dbg !27 void @_Z3barv() 
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1)
!1 = !DIFile(filename: "test.cpp", directory: "/tmp", checksumkind: CSK_MD5, checksum: "bcd632cd2764dd78c5bf358224f7e3ba")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!13 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooi", scope: !1, file: !1, line: 3, type: !14, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !17, keyInstructions: true)
!14 = !DISubroutineType(types: !15)
!15 = !{null, !16}
!16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!17 = !{!18}
!18 = !DILocalVariable(name: "x", arg: 1, scope: !13, file: !1, line: 3, type: !16)
!19 = !DILocation(line: 0, scope: !13)
!20 = !DILocation(line: 4, column: 9, scope: !21, atomGroup: 1, atomRank: 2)
!21 = distinct !DILexicalBlock(scope: !13, file: !1, line: 4, column: 7)
!22 = !DILocation(line: 4, column: 9, scope: !21, atomGroup: 1, atomRank: 1)
!23 = !DILocation(line: 5, column: 5, scope: !24)
!24 = distinct !DILexicalBlock(scope: !21, file: !1, line: 4, column: 16)
!25 = !DILocation(line: 6, column: 3, scope: !24)
!26 = !DILocation(line: 7, column: 1, scope: !13, atomGroup: 2, atomRank: 1)
!27 = !DISubprogram(name: "bar", linkageName: "_Z3barv", scope: !1, file: !1, line: 1, type: !28, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!28 = !DISubroutineType(types: !29)
!29 = !{null}
