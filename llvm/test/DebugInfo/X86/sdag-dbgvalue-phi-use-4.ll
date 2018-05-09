; RUN: llc -start-after=codegenprepare -stop-before expand-isel-pseudos -o - %s | FileCheck %s

; This is a reproducer based on the test case from PR37321.

; We verify that the fragment for the last DBG_VALUE is limited depending
; on the size of the original fragment (and that we do not emit more
; DBG_VALUE instructions than needed in case we cover the whole original
; fragment with just a few DBG_VALUE instructions).

; CHECK-LABEL: bb.{{.*}}.if.end36:
; CHECK:      [[REG1:%[0-9]+]]:gr32 = PHI
; CHECK-NEXT: [[REG2:%[0-9]+]]:gr32 = PHI
; CHECK-NEXT: [[REG3:%[0-9]+]]:gr32 = PHI
; CHECK-NEXT: DBG_VALUE debug-use [[REG1]], debug-use $noreg, !13, !DIExpression(DW_OP_LLVM_fragment, 0, 32)
; CHECK-NEXT: DBG_VALUE debug-use [[REG2]], debug-use $noreg, !13, !DIExpression(DW_OP_LLVM_fragment, 32, 32)
; CHECK-NEXT: DBG_VALUE debug-use [[REG3]], debug-use $noreg, !13, !DIExpression(DW_OP_LLVM_fragment, 64, 16)
; CHECK-NEXT: DBG_VALUE debug-use [[REG1]], debug-use $noreg, !12, !DIExpression(DW_OP_LLVM_fragment, 10, 32)
; CHECK-NEXT: DBG_VALUE debug-use [[REG2]], debug-use $noreg, !12, !DIExpression(DW_OP_LLVM_fragment, 42, 13)
; CHECK-NOT:  DBG_VALUE

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-w64-windows-gnu"

; Function Attrs: nounwind readnone
define dso_local i64 @nextafterl(i80 %a) local_unnamed_addr #0 !dbg !6 {
entry:
  br i1 undef, label %if.else, label %if.then13, !dbg !28

if.then13:                                        ; preds = %entry
  %u.sroa.0.8.insert.insert = or i80 %a, 2222, !dbg !29
  br label %if.end36, !dbg !33

if.else:                                          ; preds = %entry
  br label %if.end36

if.end36:                                         ; preds = %if.else, %if.then13
  %u.sroa.0.1.in = phi i80 [ %u.sroa.0.8.insert.insert, %if.then13 ], [ 1234567, %if.else ]
  call void @llvm.dbg.value(metadata i80 %u.sroa.0.1.in, metadata !13, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 80)), !dbg !34
  call void @llvm.dbg.value(metadata i80 %u.sroa.0.1.in, metadata !12, metadata !DIExpression(DW_OP_LLVM_fragment, 10, 45)), !dbg !34
  %u.sroa.0.0.extract.ashr = ashr i80 %u.sroa.0.1.in, 8, !dbg !35
  %u.sroa.0.0.extract.trunc = trunc i80 %u.sroa.0.0.extract.ashr to i64, !dbg !35
  ret i64 %u.sroa.0.0.extract.trunc
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!26, !27}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 7.0.0 (trunk 330808) (llvm/trunk 330813)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3)
!1 = !DIFile(filename: "pr37321.c", directory: "")
!2 = !{}
!3 = !{!4}
!4 = !DIGlobalVariableExpression(var: !5, expr: !DIExpression())
!5 = distinct !DIGlobalVariable(name: "normal_bit", scope: !6, file: !1, line: 31, type: !25, isLocal: true, isDefinition: true)
!6 = distinct !DISubprogram(name: "nextafterl", scope: !1, file: !1, line: 17, type: !7, isLocal: false, isDefinition: true, scopeLine: 18, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !10)
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !9, !9}
!9 = !DIBasicType(name: "long double", size: 96, encoding: DW_ATE_float)
!10 = !{!11, !12, !13}
!11 = !DILocalVariable(name: "x", arg: 1, scope: !6, file: !1, line: 17, type: !9)
!12 = !DILocalVariable(name: "y", arg: 2, scope: !6, file: !1, line: 17, type: !9)
!13 = !DILocalVariable(name: "u", scope: !6, file: !1, line: 27, type: !14)
!14 = distinct !DICompositeType(tag: DW_TAG_union_type, scope: !6, file: !1, line: 19, size: 128, elements: !15)
!15 = !{!16, !17}
!16 = !DIDerivedType(tag: DW_TAG_member, name: "ld", scope: !14, file: !1, line: 20, baseType: !9, size: 96)
!17 = !DIDerivedType(tag: DW_TAG_member, name: "parts", scope: !14, file: !1, line: 26, baseType: !18, size: 128)
!18 = distinct !DICompositeType(tag: DW_TAG_structure_type, scope: !14, file: !1, line: 21, size: 128, elements: !19)
!19 = !{!20, !22, !24}
!20 = !DIDerivedType(tag: DW_TAG_member, name: "mantissa", scope: !18, file: !1, line: 23, baseType: !21, size: 64)
!21 = !DIBasicType(name: "long long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!22 = !DIDerivedType(tag: DW_TAG_member, name: "expn", scope: !18, file: !1, line: 24, baseType: !23, size: 16, offset: 64)
!23 = !DIBasicType(name: "unsigned short", size: 16, encoding: DW_ATE_unsigned)
!24 = !DIDerivedType(tag: DW_TAG_member, name: "pad", scope: !18, file: !1, line: 25, baseType: !23, size: 16, offset: 80)
!25 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !21)
!26 = !{i32 2, !"Debug Info Version", i32 3}
!27 = !{i32 1, !"wchar_size", i32 2}
!28 = !DILocation(line: 47, column: 7, scope: !6)
!29 = !DILocation(line: 51, column: 14, scope: !30)
!30 = distinct !DILexicalBlock(scope: !31, file: !1, line: 50, column: 11)
!31 = distinct !DILexicalBlock(scope: !32, file: !1, line: 48, column: 5)
!32 = distinct !DILexicalBlock(scope: !6, file: !1, line: 47, column: 7)
!33 = !DILocation(line: 51, column: 2, scope: !30)
!34 = !DILocation(line: 27, column: 5, scope: !6)
!35 = !DILocation(line: 63, column: 22, scope: !36)
!36 = distinct !DILexicalBlock(scope: !6, file: !1, line: 62, column: 7)
