; dwarf::isC() enumerates every named DW_LANG_* code, but a DICompileUnit may
; name any source languages including those not defined in Dwarf.def: gaps in
; the standard range and vendor codes strictly between DW_LANG_lo_user and
; DW_LANG_hi_user. Emitting a prototyped subprogram queries isC(), which used
; to reach a trailing llvm_unreachable for unknown language codes (including
; the vendor codes). Check that each kind is emitted unrecognised rather than
; crashing, and that none is treated as C.
;
; If a code below is ever assigned a name in Dwarf.def, this test will start
; failing; replace it with another unassigned one.

; RUN: rm -rf %t && mkdir %t

; 0x0029: a gap in the standard range (Dwarf.def jumps 0x0028 -> 0x002a).
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o %t/test.o < %s
; RUN: llvm-dwarfdump -debug-info %t/test.o | FileCheck %s -DLANG=0x0029

; 0x7fff: a standard-range code above the highest assigned one (0x0048).
; RUN: sed -e "s/language: 41/language: 32767/" %s > %t/test.ll
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o %t/test.o < %t/test.ll
; RUN: llvm-dwarfdump -debug-info %t/test.o | FileCheck %s -DLANG=0x7fff

; 0x8003: a vendor code strictly between DW_LANG_lo_user and DW_LANG_hi_user.
; RUN: sed -e "s/language: 41/language: 32771/" %s > %t/test.ll
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

!0 = distinct !DICompileUnit(language: 41, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "a.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 2, !"Dwarf Version", i32 5}
!4 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !5, unit: !0, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
