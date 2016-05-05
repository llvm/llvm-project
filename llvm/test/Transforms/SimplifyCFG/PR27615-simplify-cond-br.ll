; RUN: opt -S -simplifycfg -strip-debug < %s | FileCheck %s
; RUN: opt -S -simplifycfg < %s | FileCheck %s

; Test case for BUG-27615
; Test that simplify cond branch produce same result for debug and non-debug builds
; CHECK: select i1 %or.cond, i32 -1, i32 5
; CHECK-NOT: bb1:

; ModuleID = './csmith107.i.debug.ll'
source_filename = "./csmith107.i.debug.ll"

@a = global i16 0
@b = global i32 0
@c = global i16* null


; Function Attrs: nounwind
define i16 @fn1() #3 !dbg !15 {
bb2:
  store i32 -1, i32* @b, align 1
  %_tmp1.pre = load i16, i16* @a, align 1, !dbg !19
  %_tmp2.pre = load i16*, i16** @c, align 1
  tail call void @llvm.dbg.value(metadata i16 6, i64 0, metadata !22, metadata !23), !dbg !24
  tail call void @llvm.dbg.value(metadata i16 %_tmp1.pre, i64 0, metadata !25, metadata !23), !dbg !19
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
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #4

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12, !13}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "FlexC Compiler v6.36 (LLVM)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !3)
!1 = !DIFile(filename: "csmith107.i.c", directory: "/tmp")
!2 = !{}
!3 = !{!4, !6, !10}
!4 = !DIGlobalVariable(name: "a", scope: null, file: !1, line: 2, type: !5, isLocal: false, isDefinition: true, variable: i16* @a)
!5 = !DIBasicType(name: "int", size: 16, align: 16, encoding: DW_ATE_signed)
!6 = !DIGlobalVariable(name: "b", scope: null, file: !1, line: 3, type: !7, isLocal: false, isDefinition: true, variable: i32* @b)
!7 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint32_t", file: !1, line: 1, baseType: !8)
!8 = !DIDerivedType(tag: DW_TAG_typedef, name: "__u32_t", file: !1, baseType: !9)
!9 = !DIBasicType(name: "unsigned long", size: 32, align: 16, encoding: DW_ATE_unsigned)
!10 = !DIGlobalVariable(name: "c", scope: null, file: !1, line: 4, type: !11, isLocal: false, isDefinition: true, variable: i16** @c)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 16, align: 16)
!12 = !{i32 2, !"Dwarf Version", i32 4}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!15 = distinct !DISubprogram(name: "fn1", scope: !1, file: !1, line: 5, type: !16, isLocal: false, isDefinition: true, scopeLine: 5, isOptimized: false, unit: !0, variables: !2)
!16 = !DISubroutineType(types: !17)
!17 = !{!5}
!19 = !DILocation(line: 8, column: 16, scope: !20)
!20 = !DILexicalBlock(scope: !15, file: !1, line: 7, column: 29)
!22 = !DILocalVariable(name: "d", scope: !20, line: 8, type: !5)
!23 = !DIExpression()
!24 = !DILocation(line: 8, column: 9, scope: !20)
!25 = !DILocalVariable(name: "e", scope: !20, line: 8, type: !5)

