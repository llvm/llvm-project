; RUN: opt -S -simplifycfg -strip-debug < %s | FileCheck %s
; RUN: opt -S -simplifycfg < %s | FileCheck %s

; Test case for BUG-27615
; Test that simplify cond branch produce same result for debug and non-debug builds
; CHECK: select i1 %or.cond, i32 -1, i32 5
; CHECK-NOT: bb1:

; ModuleID = './csmith107.i.debug.ll'
source_filename = "./csmith107.i.debug.ll"

@a = global i16 0, !dbg !0
@b = global i32 0, !dbg !4
@c = global i16* null, !dbg !9

define i16 @fn1() !dbg !17 {
bb2:
  store i32 -1, i32* @b, align 1
  %_tmp1.pre = load i16, i16* @a, align 1, !dbg !20
  %_tmp2.pre = load i16*, i16** @c, align 1
  tail call void @llvm.dbg.value(metadata i16 6, metadata !22, metadata !23), !dbg !24
  tail call void @llvm.dbg.value(metadata i16 %_tmp1.pre, metadata !25, metadata !23), !dbg !20
  %_tmp3 = load i16, i16* %_tmp2.pre, align 1
  %_tmp4 = icmp ne i16 %_tmp3, 0
  %_tmp6 = icmp ne i16 %_tmp1.pre, 0
  %or.cond = and i1 %_tmp6, %_tmp4
  br i1 %or.cond, label %bb5, label %bb1

bb1:                                              ; preds = %bb2
  store i32 5, i32* @b, align 1
  br label %bb5

bb5:                                              ; preds = %bb1, %bb2
  ret i16 0
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

attributes #0 = { nounwind readnone }

!llvm.dbg.cu = !{!12}
!llvm.module.flags = !{!15, !16}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "a", scope: null, file: !2, line: 2, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "csmith107.i.c", directory: "/tmp")
!3 = !DIBasicType(name: "int", size: 16, align: 16, encoding: DW_ATE_signed)
!4 = !DIGlobalVariableExpression(var: !5, expr: !DIExpression())
!5 = !DIGlobalVariable(name: "b", scope: null, file: !2, line: 3, type: !6, isLocal: false, isDefinition: true)
!6 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint32_t", file: !2, line: 1, baseType: !7)
!7 = !DIDerivedType(tag: DW_TAG_typedef, name: "__u32_t", file: !2, baseType: !8)
!8 = !DIBasicType(name: "unsigned long", size: 32, align: 16, encoding: DW_ATE_unsigned)
!9 = !DIGlobalVariableExpression(var: !10, expr: !DIExpression())
!10 = !DIGlobalVariable(name: "c", scope: null, file: !2, line: 4, type: !11, isLocal: false, isDefinition: true)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 16, align: 16)
!12 = distinct !DICompileUnit(language: DW_LANG_C, file: !2, producer: "FlexC Compiler v6.36 (LLVM)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !13, retainedTypes: !13, globals: !14)
!13 = !{}
!14 = !{!0, !4, !9}
!15 = !{i32 2, !"Dwarf Version", i32 4}
!16 = !{i32 2, !"Debug Info Version", i32 3}
!17 = distinct !DISubprogram(name: "fn1", scope: !2, file: !2, line: 5, type: !18, isLocal: false, isDefinition: true, scopeLine: 5, isOptimized: false, unit: !12, retainedNodes: !13)
!18 = !DISubroutineType(types: !19)
!19 = !{!3}
!20 = !DILocation(line: 8, column: 16, scope: !21)
!21 = !DILexicalBlock(scope: !17, file: !2, line: 7, column: 29)
!22 = !DILocalVariable(name: "d", scope: !21, line: 8, type: !3)
!23 = !DIExpression()
!24 = !DILocation(line: 8, column: 9, scope: !21)
!25 = !DILocalVariable(name: "e", scope: !21, line: 8, type: !3)

