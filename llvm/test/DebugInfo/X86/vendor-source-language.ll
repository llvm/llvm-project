; DW_LANG_lo_user and DW_LANG_hi_user are enumerated by dwarf::isC(), but the
; vendor-defined codes between them are not, and used to fall through to the
; trailing llvm_unreachable. Emitting a prototyped subprogram in such a
; language calls isC() and must not crash.

; RUN: llc -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o %t.o < %s
; RUN: llvm-dwarfdump -debug-info %t.o | FileCheck %s

; CHECK: DW_TAG_compile_unit
; CHECK:   DW_AT_language (0x8003)

define void @f() !dbg !4 {
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

; 32771 == DW_LANG_lo_user (0x8000) + 3, a vendor-defined source language.
!0 = distinct !DICompileUnit(language: 32771, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "a.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 2, !"Dwarf Version", i32 5}
!4 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !5, unit: !0, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
