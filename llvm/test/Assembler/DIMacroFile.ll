; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

@foo = global i32 0

; CHECK: !named = !{!0, !1, !2, !3, !4, !5}
!named = !{!0, !1, !2, !3, !4, !5}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!6, !7}
 
!0 = distinct !DICompileUnit(language: DW_LANG_C89, file: !1, macros: !2)
!1 = !DIFile(filename: "a.c", directory: "/")
!2 = !{!3}
; CHECK: !3 = !DIMacroFile(file: !1, nodes: !4)
!3 = !DIMacroFile(line: 0, file: !1, nodes: !4)
!4 = !{!5}
; CHECK: !5 = !DIMacro(type: DW_MACINFO_define, name: "X", value: "5")
!5 = !DIMacro(type: DW_MACINFO_define, line: 0, name: "X", value: "5")
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
