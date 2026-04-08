; RUN: opt -passes='print<module-debuginfo>' -disable-output 2>&1 < %s \
; RUN:   | FileCheck %s

; Macro hierarchy graph:
; CompileUnit
;   ├── MacroFile: ./def.c (!2)
;   │   ├── Define Macro: 'SIZE' = '5' (!6)
;   │   ├── Undef Macro: 'SIZE' (!7)
;   │   └── MacroFile: ./def.nested.c (!4)
;   │       └── Undef Macro: 'BAZ' (!9)
;   └── Define Macro: 'BAZ' (!8)

; CHECK: Macro: DW_MACINFO_define 'SIZE' = '5' from ./def.c
; CHECK: Macro: DW_MACINFO_undef 'SIZE' from ./def.c
; CHECK: Macro: DW_MACINFO_undef 'BAZ' from ./def.nested.c
; CHECK: Macro: DW_MACINFO_define 'BAZ' at line 1

!llvm.module.flags = !{!0, !10}
!llvm.dbg.cu = !{!1}

!0 = !{i32 7, !"Dwarf Version", i32 0}
!1 = distinct !DICompileUnit(language: DW_LANG_OpenCL, file: !3, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, macros: !{!2, !8})
!2 = !DIMacroFile(file: !3, nodes: !{!6, !7, !4})
!3 = !DIFile(filename: "def.c", directory: ".")
!4 = !DIMacroFile(file: !5, nodes: !{!9})
!5 = !DIFile(filename: "def.nested.c", directory: ".")
!6 = !DIMacro(type: DW_MACINFO_define, line: 1, name: "SIZE", value: "5")
!7 = !DIMacro(type: DW_MACINFO_undef, line: 1, name: "SIZE")
!8 = !DIMacro(type: DW_MACINFO_define, line: 1, name: "BAZ")
!9 = !DIMacro(type: DW_MACINFO_undef, line: 1, name: "BAZ")
!10 = !{i32 2, !"Debug Info Version", i32 3}
