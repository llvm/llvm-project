; RUN: llc -mtriple=x86_64-pc-linux-gnu %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; ModuleID = 'test.c'

@GLB = common global i32 0, align 4

define i32 @f() nounwind !dbg !5 {
  %LOC = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %LOC, metadata !15, metadata !DIExpression()), !dbg !17
  %1 = load i32, i32* @GLB, align 4, !dbg !18
  store i32 %1, i32* %LOC, align 4, !dbg !18
  %2 = load i32, i32* @GLB, align 4, !dbg !19
  ret i32 %2, !dbg !19
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.0 (trunk)", isOptimized: false, emissionKind: FullDebug, file: !20, enums: !1, retainedTypes: !1, globals: !12, imports:  !1)
!1 = !{}
!5 = distinct !DISubprogram(name: "f", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !0, file: !6, scope: !6, type: !7)
!6 = !DIFile(filename: "test.c", directory: "/work/llvm/vanilla/test/DebugInfo")
!7 = !DISubroutineType(types: !8)
!8 = !{!9}
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!12 = !{!14}
!14 = !DIGlobalVariable(name: "GLB", line: 1, isLocal: false, isDefinition: true, scope: null, file: !6, type: !9, variable: i32* @GLB)
!15 = !DILocalVariable(name: "LOC", line: 4, scope: !16, file: !6, type: !9)
!16 = distinct !DILexicalBlock(line: 3, column: 9, file: !20, scope: !5)
!17 = !DILocation(line: 4, column: 9, scope: !16)
!18 = !DILocation(line: 4, column: 23, scope: !16)
!19 = !DILocation(line: 5, column: 5, scope: !16)
!20 = !DIFile(filename: "test.c", directory: "/work/llvm/vanilla/test/DebugInfo")

; CHECK: DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name [DW_FORM_strp]       ( .debug_str[0x{{[0-9a-f]*}}] = "GLB")
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_decl_file [DW_FORM_data1] ("/work/llvm/vanilla/test/DebugInfo{{[/\\]}}test.c")
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_decl_line [DW_FORM_data1] (1)

; CHECK: DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name [DW_FORM_strp]   ( .debug_str[0x{{[0-9a-f]*}}] = "LOC")
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_decl_file [DW_FORM_data1]     ("/work/llvm/vanilla/test/DebugInfo{{[/\\]}}test.c")
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_decl_line [DW_FORM_data1]     (4)

!21 = !{i32 1, !"Debug Info Version", i32 3}
