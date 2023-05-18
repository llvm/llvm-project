; RUN: llvm-link %s %s -S -o -| FileCheck %s

; This test checks that DIMacro and DIMacroFile comparison works correctly.

; CHECK: !llvm.dbg.cu = !{[[CU1:![0-9]*]], [[CU2:![0-9]*]]}

; CHECK: [[CU1]] = distinct !DICompileUnit
; CHECK-SAME: macros: [[MS1:![0-9]*]]
; CHECK: [[F1:![0-9]*]] = !DIFile(filename: "t.c"
; CHECK: [[MS1]] = !{[[MF1:![0-9]*]]}
; CHECK: [[MF1]] = !DIMacroFile(
; CHECK-SAME: file: [[F1]], nodes: [[MS2:![0-9]*]])
; CHECK: [[MS2]] = !{[[M1:![0-9]*]]}
; CHECK: [[M1]] = !DIMacro(type: DW_MACINFO_define, line: 3, name: "X", value: "5")
; CHECK: [[CU2]] = distinct !DICompileUnit
; CHECK-SAME: macros: [[MS1]]

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 4.0.0 (trunk 276746)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, macros: !3)
!1 = !DIFile(filename: "t.c", directory: "/")
!2 = !{}
!3 = !{!4}
!4 = !DIMacroFile(line: 0, file: !1, nodes: !5)
!5 = !{!6}
!6 = !DIMacro(type: DW_MACINFO_define, line: 3, name: "X", value: "5")
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{!"clang version 4.0.0 (trunk 276746)"}
