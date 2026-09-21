; dwarf::isC() enumerates every named DW_LANG_* code, but a DICompileUnit may
; name any source language, including ones no case covers: standard codes not
; yet added to Dwarf.def, and vendor codes strictly between DW_LANG_lo_user and
; DW_LANG_hi_user (both of which *are* named). Emitting a prototyped subprogram
; queries isC(), which used to fall through to a trailing llvm_unreachable for
; all of them. Check that each kind is emitted unrecognised rather than
; crashing, and that neither is treated as C.

; RUN: rm -rf %t
; RUN: mkdir %t

; 0x7fff: a standard-range code above the highest assigned one (0x0048). If it
; is ever assigned a name in Dwarf.def, replace it with another unassigned one.
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o %t/test.o < %s
; RUN: llvm-dwarfdump -debug-info %t/test.o | FileCheck %s -DLANG=0x7fff

; 0x8003: a vendor code strictly between DW_LANG_lo_user and DW_LANG_hi_user.
; RUN: sed -e "s/language: 32767/language: 32771/" %s > %t/test.ll
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o %t/test.o < %t/test.ll
; RUN: llvm-dwarfdump -debug-info %t/test.o | FileCheck %s -DLANG=0x8003

; CHECK:     DW_TAG_compile_unit
; CHECK:       DW_AT_language ([[LANG]])

; An unrecognised language is not C, so DIFlagPrototyped must not become
; DW_AT_prototyped.
; CHECK:     DW_TAG_subprogram
; CHECK-NOT:   DW_AT_prototyped
; CHECK:       DW_AT_name ("f")
; CHECK-NOT:   DW_AT_prototyped
; CHECK:     NULL

define void @f() !dbg !4 {
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: 32767, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "a.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 2, !"Dwarf Version", i32 5}
!4 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !5, unit: !0, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
