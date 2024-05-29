; RUN: llc -mtriple=x86_64-unknown-linux-gnu %s -o %t -filetype=obj
; RUN: llvm-dwarfdump --debug-info %t | FileCheck %s

%struct.struct_t = type { i8 }

@g = dso_local global %struct.struct_t zeroinitializer, align 1, !dbg !0

; CHECK-LABEL: DW_TAG_subprogram
; CHECK: DW_AT_name ("test1")
; CHECK: DW_TAG_variable
; CHECK: DW_AT_location (DW_OP_fbreg -1, DW_OP_deref, DW_OP_constu 0x3d, DW_OP_shl, DW_OP_constu 0x3d, DW_OP_shr, DW_OP_stack_value)
; CHECK: DW_AT_name ("x")
; CHECK: DW_TAG_variable
; CHECK: DW_AT_location (DW_OP_fbreg -1, DW_OP_deref, DW_OP_constu 0x39, DW_OP_shl, DW_OP_constu 0x3c, DW_OP_shra, DW_OP_stack_value)
; CHECK: DW_AT_name ("y")

define i32 @test1() !dbg !13 {
entry:
  %0 = alloca %struct.struct_t, align 1
  tail call void @llvm.dbg.declare(metadata ptr %0, metadata !17, metadata !DIExpression(DW_OP_LLVM_extract_bits, 0, 3, DW_ATE_unsigned)), !dbg !18
  tail call void @llvm.dbg.declare(metadata ptr %0, metadata !19, metadata !DIExpression(DW_OP_LLVM_extract_bits, 3, 4, DW_ATE_signed)), !dbg !21
  ret i32 0, !dbg !22
}

; CHECK-LABEL: DW_TAG_subprogram
; CHECK: DW_AT_name ("test2")
; CHECK: DW_TAG_variable
; CHECK: DW_AT_location (DW_OP_breg0 RAX+0, DW_OP_constu 0xff, DW_OP_and, DW_OP_constu 0x3d, DW_OP_shl, DW_OP_constu 0x3d, DW_OP_shr, DW_OP_stack_value)
; CHECK: DW_AT_name ("x")
; CHECK: DW_TAG_variable
; CHECK: DW_AT_location (DW_OP_breg0 RAX+0, DW_OP_constu 0xff, DW_OP_and, DW_OP_constu 0x39, DW_OP_shl, DW_OP_constu 0x3c, DW_OP_shra, DW_OP_stack_value)
; CHECK: DW_AT_name ("y")

define i8 @test2() !dbg !23 {
entry:
  %0 = load i8, ptr @g, align 1
  tail call void @llvm.dbg.value(metadata i8 %0, metadata !24, metadata !DIExpression(DW_OP_LLVM_extract_bits, 0, 3, DW_ATE_unsigned)), !dbg !25
  tail call void @llvm.dbg.value(metadata i8 %0, metadata !26, metadata !DIExpression(DW_OP_LLVM_extract_bits, 3, 4, DW_ATE_signed)), !dbg !27
  ret i8 %0, !dbg !28
}

; CHECK-LABEL: DW_TAG_subprogram
; CHECK: DW_AT_name ("test3")
; CHECK: DW_TAG_variable
; CHECK: DW_AT_location (DW_OP_breg0 RAX+0, DW_OP_constu 0x3f, DW_OP_shr, DW_OP_stack_value)
; CHECK: DW_AT_name ("x")
; CHECK: DW_TAG_variable
; CHECK: DW_AT_location (DW_OP_breg0 RAX+0, DW_OP_constu 0x3f, DW_OP_shra, DW_OP_stack_value)
; CHECK: DW_AT_name ("y")

define i64 @test3(ptr %p) !dbg !29 {
entry:
  %0 = load i64, ptr %p, align 8
  tail call void @llvm.dbg.value(metadata i64 %0, metadata !33, metadata !DIExpression(DW_OP_LLVM_extract_bits, 63, 1, DW_ATE_unsigned)), !dbg !30
  tail call void @llvm.dbg.value(metadata i64 %0, metadata !34, metadata !DIExpression(DW_OP_LLVM_extract_bits, 63, 1, DW_ATE_signed)), !dbg !31
  ret i64 %0, !dbg !32
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!11, !12}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !3, line: 6, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "DW_OP_bit_piece.cpp", directory: "./")
!4 = !{!0}
!5 = !DIDerivedType(tag: DW_TAG_typedef, name: "struct_t", file: !3, line: 4, baseType: !6)
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !3, line: 1, size: 8, flags: DIFlagTypePassByValue, elements: !7, identifier: "_ZTS8struct_t")
!7 = !{!8, !10}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !6, file: !3, line: 2, baseType: !9, size: 3, flags: DIFlagBitField, extraData: i64 0)
!9 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !6, file: !3, line: 3, baseType: !9, size: 4, offset: 3, flags: DIFlagBitField, extraData: i64 0)
!11 = !{i32 7, !"Dwarf Version", i32 5}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = distinct !DISubprogram(name: "test1", linkageName: "test1", scope: !3, file: !3, line: 8, type: !14, scopeLine: 8, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !16)
!14 = !DISubroutineType(types: !15)
!15 = !{!9}
!16 = !{}
!17 = !DILocalVariable(name: "x", scope: !13, file: !3, line: 9, type: !9)
!18 = !DILocation(line: 9, column: 9, scope: !13)
!19 = !DILocalVariable(name: "y", scope: !13, file: !3, line: 9, type: !20)
!20 = !DIBasicType(name: "signed int", size: 32, encoding: DW_ATE_signed)
!21 = !DILocation(line: 9, column: 12, scope: !13)
!22 = !DILocation(line: 10, column: 3, scope: !13)
!23 = distinct !DISubprogram(name: "test2", linkageName: "test2", scope: !3, file: !3, line: 8, type: !14, scopeLine: 8, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !16)
!24 = !DILocalVariable(name: "x", scope: !23, file: !3, line: 9, type: !9)
!25 = !DILocation(line: 9, column: 9, scope: !23)
!26 = !DILocalVariable(name: "y", scope: !23, file: !3, line: 9, type: !20)
!27 = !DILocation(line: 9, column: 12, scope: !23)
!28 = !DILocation(line: 10, column: 3, scope: !23)
!29 = distinct !DISubprogram(name: "test3", linkageName: "test3", scope: !3, file: !3, line: 8, type: !14, scopeLine: 8, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !16)
!30 = !DILocation(line: 9, column: 9, scope: !29)
!31 = !DILocation(line: 9, column: 12, scope: !29)
!32 = !DILocation(line: 10, column: 3, scope: !29)
!33 = !DILocalVariable(name: "x", scope: !29, file: !3, line: 9, type: !9)
!34 = !DILocalVariable(name: "y", scope: !29, file: !3, line: 9, type: !20)
