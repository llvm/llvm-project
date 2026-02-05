; RUN: opt -passes='print<module-debuginfo>' -disable-output 2>&1 < %s \
; RUN:   | FileCheck %s

; Macro hierarchy graph:
; CompileUnit
;   ├── MacroFile: ./def.c (!3)
;   │   ├── Define Macro: 'SIZE' = '5' (!7)
;   │   ├── Undef Macro: 'SIZE' (!8)
;   │   └── MacroFile: ./def.nested.c (!5)
;   │       └── Undef Macro: 'BAZ' (!10)
;   └── Define Macro: 'BAZ' (!9)

; CHECK: Macro: DW_MACINFO_define 'SIZE' = '5' from ./def.c
; CHECK: Macro: DW_MACINFO_undef 'SIZE' from ./def.c
; CHECK: Macro: DW_MACINFO_undef 'BAZ' from ./def.nested.c
; CHECK: Macro: DW_MACINFO_define 'BAZ' at line 1

!llvm.module.flags = !{!0, !11}
!llvm.dbg.cu = !{!1}

!0 = !{i32 7, !"Dwarf Version", i32 0}
!1 = distinct !DICompileUnit(language: DW_LANG_OpenCL, file: !2, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, macros: !{!3, !9})
!2 = !DIFile(filename: "def.c", directory: "/tmp")
!3 = !DIMacroFile(file: !4, nodes: !{!7, !8, !5})
!4 = !DIFile(filename: "def.c", directory: ".")
!5 = !DIMacroFile(file: !6, nodes: !{!10})
!6 = !DIFile(filename: "def.nested.c", directory: ".")
!7 = !DIMacro(type: DW_MACINFO_define, line: 1, name: "SIZE", value: "5")
!8 = !DIMacro(type: DW_MACINFO_undef, line: 1, name: "SIZE")
!9 = !DIMacro(type: DW_MACINFO_define, line: 1, name: "BAZ")
!10 = !DIMacro(type: DW_MACINFO_undef, line: 1, name: "BAZ")
!11 = !{i32 2, !"Debug Info Version", i32 3}
