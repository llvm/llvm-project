; RUN: llc -mtriple=hexagon -mcpu=hexagonv85 -mattr=+hlx < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=hexagon -mcpu=hexagonv85 -mattr=+hlx -filetype=obj < %s \
; RUN:   | llvm-dwarfdump --debug-info - | FileCheck %s --check-prefix=DWARF

; The DW_CC values of the Hexagon HLX calling conventions are emitted in
; objects, so they must not change: DW_CC_LLVM_HexagonHLXCall = 0xd1 and
; DW_CC_LLVM_HexagonPreserveHLX = 0xd2.

; ASM: .byte 209 // DW_AT_calling_convention
; ASM: .byte 210 // DW_AT_calling_convention

; DWARF: DW_AT_name ("f_hlxcc")
; DWARF: DW_AT_calling_convention (DW_CC_LLVM_HexagonHLXCall)
; DWARF: DW_AT_name ("f_preserve_hlx")
; DWARF: DW_AT_calling_convention (DW_CC_LLVM_HexagonPreserveHLX)

define hexagon_hlxcc void @f_hlxcc() !dbg !6 {
  ret void, !dbg !12
}

define hexagon_preserve_hlx void @f_preserve_hlx() !dbg !9 {
  ret void, !dbg !13
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "cc.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "f_hlxcc", scope: !1, file: !1, line: 1, type: !7, spFlags: DISPFlagDefinition, unit: !0)
!7 = !DISubroutineType(cc: DW_CC_LLVM_HexagonHLXCall, types: !8)
!8 = !{null}
!9 = distinct !DISubprogram(name: "f_preserve_hlx", scope: !1, file: !1, line: 2, type: !10, spFlags: DISPFlagDefinition, unit: !0)
!10 = !DISubroutineType(cc: DW_CC_LLVM_HexagonPreserveHLX, types: !8)
!12 = !DILocation(line: 1, scope: !6)
!13 = !DILocation(line: 2, scope: !9)
