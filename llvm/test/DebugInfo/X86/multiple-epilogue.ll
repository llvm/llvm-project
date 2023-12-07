; RUN: llc -O1 %s -mtriple=x86_64 -filetype=obj -o %t && llvm-dwarfdump  -debug-line %t | FileCheck -v --check-prefix=LINE-TABLE %s

; based on
;
;   1:   int bar();
;   2:   int baz();
;   3;
;   4:   int foo(int a) {
;   5:     if (a > 20)
;   6:       return bar();
;   7:     else
;   8:       return baz();
;   9:   }
;
; compiled with -g -S -emit-llvm -O1


; LINE-TABLE: .debug_line contents:
; LINE-TABLE: Line table prologue:
; LINE-TABLE:     total_length: 0x00000069
; LINE-TABLE:           format: DWARF32
; LINE-TABLE:          version: 5
; LINE-TABLE:     address_size: 8
; LINE-TABLE:  seg_select_size: 0
; LINE-TABLE:  prologue_length: 0x00000037

; LINE-TABLE:      Address            Line   Column File   ISA Discriminator OpIndex Flags
; LINE-TABLE-NEXT: ------------------ ------ ------ ------ --- ------------- ------- -------------
; LINE-TABLE-NEXT: 0x0000000000000000      4      0      0   0             0       0  is_stmt
; LINE-TABLE-NEXT: 0x0000000000000001      5      9      0   0             0       0  is_stmt prologue_end
; LINE-TABLE-NEXT: 0x0000000000000004      5      7      0   0             0       0
; LINE-TABLE-NEXT: 0x0000000000000006      6     12      0   0             0       0  is_stmt
; LINE-TABLE-NEXT: 0x000000000000000b      9      1      0   0             0       0  is_stmt epilogue_begin
; LINE-TABLE-NEXT: 0x000000000000000d      8     12      0   0             0       0  is_stmt
; LINE-TABLE-NEXT: 0x0000000000000012      9      1      0   0             0       0  is_stmt epilogue_begin
; LINE-TABLE-NEXT: 0x0000000000000014      9      1      0   0             0       0  is_stmt end_sequence




; Function Attrs: mustprogress uwtable
define dso_local noundef i32 @_Z3fooi(i32 noundef %a) local_unnamed_addr #0 !dbg !8 {
entry:
  call void @llvm.dbg.value(metadata i32 %a, metadata !13, metadata !DIExpression()), !dbg !14
  %cmp = icmp sgt i32 %a, 20, !dbg !15
  br i1 %cmp, label %if.then, label %if.else, !dbg !17

if.then:                                          ; preds = %entry
  %call = call noundef i32 @_Z3barv(), !dbg !18
  br label %return, !dbg !19

if.else:                                          ; preds = %entry
  %call1 = call noundef i32 @_Z3bazv(), !dbg !20
  br label %return, !dbg !21

return:                                           ; preds = %if.else, %if.then
  %retval.0 = phi i32 [ %call, %if.then ], [ %call1, %if.else ], !dbg !22
  ret i32 %retval.0, !dbg !23
}

declare !dbg !24 dso_local noundef i32 @_Z3barv() local_unnamed_addr #1

declare !dbg !28 dso_local noundef i32 @_Z3bazv() local_unnamed_addr #1

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 15.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "q.cpp", directory: "/", checksumkind: CSK_MD5, checksum: "6b5f6118d466ba2f6510ac0790b09bef")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 2}
!6 = !{i32 7, !"frame-pointer", i32 2}
!7 = !{!"clang version 15.0.0"}
!8 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooi", scope: !1, file: !1, line: 4, type: !9, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !12)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{!13}
!13 = !DILocalVariable(name: "a", arg: 1, scope: !8, file: !1, line: 4, type: !11)
!14 = !DILocation(line: 0, scope: !8)
!15 = !DILocation(line: 5, column: 9, scope: !16)
!16 = distinct !DILexicalBlock(scope: !8, file: !1, line: 5, column: 7)
!17 = !DILocation(line: 5, column: 7, scope: !8)
!18 = !DILocation(line: 6, column: 12, scope: !16)
!19 = !DILocation(line: 6, column: 5, scope: !16)
!20 = !DILocation(line: 8, column: 12, scope: !16)
!21 = !DILocation(line: 8, column: 5, scope: !16)
!22 = !DILocation(line: 0, scope: !16)
!23 = !DILocation(line: 9, column: 1, scope: !8)
!24 = !DISubprogram(name: "bar", linkageName: "_Z3barv", scope: !1, file: !1, line: 1, type: !25, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !27)
!25 = !DISubroutineType(types: !26)
!26 = !{!11}
!27 = !{}
!28 = !DISubprogram(name: "baz", linkageName: "_Z3bazv", scope: !1, file: !1, line: 2, type: !25, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !27)
