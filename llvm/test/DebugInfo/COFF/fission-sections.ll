; RUN: llc -split-dwarf-file=baz.dwo -split-dwarf-output=%t.dwo -O0 %s -mtriple=x86_64-unknown-windows-msvc -filetype=obj -o %t
; RUN: llvm-objdump -ht %t | FileCheck --check-prefix=OBJ %s
; RUN: llvm-objdump -ht %t.dwo | FileCheck --check-prefix=DWO %s

; This test is derived from test/DebugInfo/X86/fission-cu.ll
; But it checks that the output objects have the expected sections

source_filename = "test/DebugInfo/X86/fission-cu.ll"

@a = common global i32 0, align 4, !dbg !0

!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!7}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "a", scope: null, file: !2, line: 1, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "baz.c", directory: "e:\\llvm-project\\tmp")
!3 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!4 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang version 3.3 (trunk 169021) (llvm/trunk 169020)", isOptimized: false, runtimeVersion: 0, splitDebugFilename: "baz.dwo", emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !5)
!5 = !{}
!6 = !{!0}
!7 = !{i32 1, !"Debug Info Version", i32 3}

; CHECK-LABEL: Sections:

; OBJ:     Idx Name
; OBJ-NEXT:  0 .text
; OBJ-NEXT:  1 .data
; OBJ-NEXT:  2 .bss
; OBJ-NEXT:  3 .debug$S
; OBJ-NEXT:  4 .debug_abbrev
; OBJ-NEXT:  5 .debug_info
; OBJ-NEXT:  6 .debug_str
; OBJ-NEXT:  7 .debug_addr
; OBJ-NEXT:  8 .debug_pubnames
; OBJ-NEXT:  9 .debug_pubtypes
; OBJ-NEXT: 10 .debug_line

; OBJ:     .debug_abbrev
; OBJ:     .debug_info
; OBJ-NOT: .dwo

; DWO:      Idx Name
; DWO-NEXT:   0 .debug_str.dwo
; DWO-NEXT:   1 .debug_str_offsets.dwo
; DWO-NEXT:   2 .debug_info.dwo
; DWO-NEXT:   3 .debug_abbrev.dwo

; DWO:      SYMBOL TABLE:
; DWO-NEXT: [ 0](sec  1)(fl 0x00)(ty   0)(scl   3) (nx 1) 0x00000000 .debug_str.dwo
; DWO-NEXT: AUX scnlen 0x49 nreloc 0 nlnno 0 checksum 0xa4983874 assoc 1 comdat 0
; DWO-NEXT: [ 2](sec  2)(fl 0x00)(ty   0)(scl   3) (nx 1) 0x00000000 .debug_str_offsets.dwo
; DWO-NEXT: AUX scnlen 0x14 nreloc 0 nlnno 0 checksum 0x9392f0f0 assoc 2 comdat 0
; DWO-NEXT: [ 4](sec  3)(fl 0x00)(ty   0)(scl   3) (nx 1) 0x00000000 .debug_info.dwo
; DWO-NEXT: AUX scnlen 0x29 nreloc 0 nlnno 0 checksum 0x{{[0-9a-f]*}} assoc 3 comdat 0
; DWO-NEXT: [ 6](sec  4)(fl 0x00)(ty   0)(scl   3) (nx 1) 0x00000000 .debug_abbrev.dwo
; DWO-NEXT: AUX scnlen 0x33 nreloc 0 nlnno 0 checksum 0x8056e5e4 assoc 4 comdat 0
; DWO-EMPTY:
