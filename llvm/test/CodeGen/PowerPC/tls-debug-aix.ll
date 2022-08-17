; RUN: llc -mtriple powerpc-ibm-aix-xcoff < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple powerpc64-ibm-aix-xcoff < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple powerpc-ibm-aix-xcoff -filetype=obj < %s | \
; RUN:   llvm-dwarfdump -i - | FileCheck %s --check-prefix=OBJ
; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -filetype=obj < %s | \
; RUN:   llvm-dwarfdump -i - | FileCheck %s --check-prefix=OBJ

; ASM:         .dwsect 0x10000
; ASM-NEXT:    .dwinfo:
; ASM:         .byte{{.*}}DW_TAG_variable
; ASM-NOT:     .byte{{.*}}DW_AT_location
; ASM-NOT:     .vbyte{{.*}}i[TL]
; ASM:         .byte{{.*}}DW_TAG_

; OBJ:        DW_TAG_variable
; OBJ-NEXT:     DW_AT_name{{.*}}("i")
; OBJ-NEXT:     DW_AT_type
; OBJ-NEXT:     DW_AT_external
; OBJ-NEXT:     DW_AT_decl_file
; OBJ-NEXT:     DW_AT_decl_line
; OBJ-NOT:      DW_AT_location{{.*}}DW_OP_form_tls_address
; OBJ:        DW_TAG_

@i = thread_local global i32 20, align 4, !dbg !0

define i32 @foo() !dbg !12 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  %0 = load i32, i32* @i, align 4, !dbg !16
  ret i32 %0, !dbg !16
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!6, !7, !8, !9, !10}
!llvm.ident = !{!11}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "i", scope: !2, file: !3, line: 2, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "IBM Open XL C/C++ for AIX", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "t.c", directory: ".")
!4 = !{!0}
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !{i32 7, !"Dwarf Version", i32 3}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 2}
!9 = !{i32 7, !"PIC Level", i32 2}
!10 = !{i32 7, !"frame-pointer", i32 2}
!11 = !{!"IBM Open XL C/C++ for AIX"}
!12 = distinct !DISubprogram(name: "foo", scope: !3, file: !3, line: 3, type: !13, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !15)
!13 = !DISubroutineType(types: !14)
!14 = !{!5}
!15 = !{}
!16 = !DILocation(line: 4, scope: !12)
