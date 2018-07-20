; RUN: llc -filetype=asm -O0 -mtriple=x86_64-linux-gnu < %s -dwarf-sections-as-references=Enable -dwarf-inlined-strings=Enable -no-dwarf-pub-sections -no-dwarf-ranges-section -dwarf-version 2 -debugger-tune=gdb | FileCheck %s

; CHECK:      .file

; CHECK-NOT:  .L

; CHECK:      .section .debug_abbrev
; CHECK-NOT:  DW_FORM_str{{p|x}}
; CHECK-NOT: .L

; CHECK:      .section .debug_info
; CHECK-NOT:  .L
; CHECK:      .short 2             # DWARF version number
; CHECK-NOT:  .L
; CHECK:      .long .debug_abbrev  # Offset Into Abbrev. Section
; CHECK-NOT:  .L
; CHECK:      .long .debug_line    # DW_AT_stmt_list
; CHECK-NOT:  .L
; CHECK:      .long .debug_abbrev  # Offset Into Abbrev. Section
; CHECK-NOT:  .L
; CHECK:      .long .debug_line    # DW_AT_stmt_list
; CHECK-NOT:  .L
; CHECK:      .quad .debug_info+{{[0-9]+}} # DW_AT_type
; CHECK-NOT:  .L
; CHECK:      .byte 0              # End Of Children Mark
; CHECK-NOT:  .L

source_filename = "test/DebugInfo/X86/sections_as_references.ll"

%struct.foo = type { i8 }

@f = global %struct.foo zeroinitializer, align 1, !dbg !0
@g = global %struct.foo zeroinitializer, align 1, !dbg !6

!llvm.dbg.cu = !{!9, !12}
!llvm.module.flags = !{!14, !15}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "f", scope: null, file: !2, line: 2, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "tu1.cpp", directory: "/dir")
!3 = !DICompositeType(tag: DW_TAG_structure_type, name: "foo", file: !4, line: 1, size: 8, align: 8, elements: !5, identifier: "_ZTS3foo")
!4 = !DIFile(filename: "./hdr.h", directory: "/dir")
!5 = !{}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = !DIGlobalVariable(name: "g", scope: null, file: !8, line: 2, type: !3, isLocal: false, isDefinition: true)
!8 = !DIFile(filename: "tu2.cpp", directory: "/dir")
!9 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !10, globals: !11, imports: !5)
!10 = !{!3}
!11 = !{!0}
!12 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !8, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !10, globals: !13, imports: !5)
!13 = !{!6}
!14 = !{i32 2, !"Dwarf Version", i32 2}
!15 = !{i32 1, !"Debug Info Version", i32 3}

