; RUN: llc < %s -mtriple=armv7-linux-gnueabihf -O1 -stop-after=finalize-isel | FileCheck %s

define dso_local i32 @h(i64 %j) local_unnamed_addr !dbg !8 {
entry:
  call void @llvm.dbg.value(metadata i64 %j, metadata !14, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.value(metadata i64 %j, metadata !15, metadata !DIExpression(DW_OP_LLVM_convert, 64, DW_ATE_unsigned, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_stack_value, DW_OP_LLVM_fragment, 0, 32)), !dbg !29
  call void @llvm.dbg.value(metadata i64 %j, metadata !15, metadata !DIExpression(DW_OP_constu, 32, DW_OP_shr, DW_OP_LLVM_convert, 64, DW_ATE_unsigned, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_stack_value, DW_OP_LLVM_fragment, 32, 32)), !dbg !29
  %tobool = icmp ult i64 %j, 4294967296, !dbg !30
  br i1 %tobool, label %cleanup, label %if.then, !dbg !31

if.then:                                          ; preds = %entry
  call void @llvm.dbg.value(metadata i64 %j, metadata !15, metadata !DIExpression(DW_OP_LLVM_convert, 64, DW_ATE_unsigned, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_stack_value, DW_OP_LLVM_fragment, 0, 32)), !dbg !29
  %conv = sitofp i64 %j to double, !dbg !32
  %add = fadd double %conv, 0x43F0000000000000, !dbg !33
  call void @llvm.dbg.value(metadata double %add, metadata !25, metadata !DIExpression()), !dbg !34
  %conv2 = fptosi double %add to i32, !dbg !35
  br label %cleanup

cleanup:                                          ; preds = %entry, %if.then
  %retval.0 = phi i32 [ %conv2, %if.then ], [ undef, %entry ]
  ret i32 %retval.0, !dbg !36
}

; CHECK-LABEL: bb.0.entry:
; CHECK: DBG_VALUE [[REG1:%[0-9]+]], $noreg, !14, !DIExpression(DW_OP_LLVM_fragment, 32, 32
; CHECK: DBG_VALUE [[REG2:%[0-9]+]], $noreg, !14, !DIExpression(DW_OP_LLVM_fragment, 0, 32
; CHECK: DBG_VALUE [[REG2]], $noreg, !15, !DIExpression({{.+}}DW_OP_LLVM_fragment, 0, 32
; CHECK: DBG_VALUE $noreg, $noreg, !15, !DIExpression({{.+}}DW_OP_LLVM_fragment, 32, 32

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "tif_aux.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 1, !"min_enum_size", i32 4}
!7 = !{!"clang version 10.0.0 "}
!8 = distinct !DISubprogram(name: "h", scope: !1, file: !1, line: 10, type: !9, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !13)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !12}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DIBasicType(name: "long long int", size: 64, encoding: DW_ATE_signed)
!13 = !{!14, !15, !25}
!14 = !DILocalVariable(name: "j", arg: 1, scope: !8, file: !1, line: 10, type: !12)
!15 = !DILocalVariable(name: "i", scope: !8, file: !1, line: 11, type: !16)
!16 = !DIDerivedType(tag: DW_TAG_typedef, name: "g", file: !1, line: 8, baseType: !17)
!17 = distinct !DICompositeType(tag: DW_TAG_union_type, file: !1, line: 5, size: 64, elements: !18)
!18 = !{!19, !24}
!19 = !DIDerivedType(tag: DW_TAG_member, name: "e", scope: !17, file: !1, line: 6, baseType: !20, size: 64)
!20 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "b", file: !1, line: 1, size: 64, elements: !21)
!21 = !{!22, !23}
!22 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !20, file: !1, line: 2, baseType: !11, size: 32)
!23 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !20, file: !1, line: 2, baseType: !11, size: 32, offset: 32)
!24 = !DIDerivedType(tag: DW_TAG_member, name: "f", scope: !17, file: !1, line: 7, baseType: !12, size: 64)
!25 = !DILocalVariable(name: "a", scope: !26, file: !1, line: 14, type: !28)
!26 = distinct !DILexicalBlock(scope: !27, file: !1, line: 13, column: 14)
!27 = distinct !DILexicalBlock(scope: !8, file: !1, line: 13, column: 7)
!28 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!29 = !DILocation(line: 0, scope: !8)
!30 = !DILocation(line: 13, column: 7, scope: !27)
!31 = !DILocation(line: 13, column: 7, scope: !8)
!32 = !DILocation(line: 14, column: 16, scope: !26)
!33 = !DILocation(line: 14, column: 20, scope: !26)
!34 = !DILocation(line: 0, scope: !26)
!35 = !DILocation(line: 15, column: 12, scope: !26)
!36 = !DILocation(line: 17, column: 1, scope: !8)
