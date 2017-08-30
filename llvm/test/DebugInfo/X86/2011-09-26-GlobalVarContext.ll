; RUN: llc -mtriple=x86_64-pc-linux-gnu %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; ModuleID = 'test.c'

source_filename = "test/DebugInfo/X86/2011-09-26-GlobalVarContext.ll"

@GLB = common global i32 0, align 4, !dbg !0

; Function Attrs: nounwind
define i32 @f() #0 !dbg !8 {
  %LOC = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %LOC, metadata !11, metadata !13), !dbg !14
  %1 = load i32, i32* @GLB, align 4, !dbg !15
  store i32 %1, i32* %LOC, align 4, !dbg !15
  %2 = load i32, i32* @GLB, align 4, !dbg !16
  ret i32 %2, !dbg !16
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!7}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "GLB", scope: null, file: !2, line: 1, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "test.c", directory: "/work/llvm/vanilla/test/DebugInfo")
!3 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!4 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang version 3.0 (trunk)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !5)
!5 = !{}
!6 = !{!0}
!7 = !{i32 1, !"Debug Info Version", i32 3}
!8 = distinct !DISubprogram(name: "f", scope: !2, file: !2, line: 3, type: !9, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !4)
!9 = !DISubroutineType(types: !10)
!10 = !{!3}
!11 = !DILocalVariable(name: "LOC", scope: !12, file: !2, line: 4, type: !3)
!12 = distinct !DILexicalBlock(scope: !8, file: !2, line: 3, column: 9)
!13 = !DIExpression()
!14 = !DILocation(line: 4, column: 9, scope: !12)
!15 = !DILocation(line: 4, column: 23, scope: !12)
!16 = !DILocation(line: 5, column: 5, scope: !12)

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

