; CELQMAIN must be emitted while the text section is still open. With debug
; info, the DWARF aranges end the text section before emitEndOfAsmFile.
;
; RUN: llc < %s -mtriple=s390x-ibm-zos -generate-arange-section | FileCheck %s
; RUN: llc < %s -mtriple=s390x-ibm-zos -generate-arange-section \
; RUN:   -filetype=obj -o - | od -Ax -tx1 | FileCheck --check-prefix=OBJ %s

; CHECK:      CELQMAIN DS 0H
; CHECK-NEXT: * CELQMAIN, RENT format
; CHECK-NEXT:  DC XL4'04000001'
; CHECK:      main DS 0H

; OBJ: 04 00 00 01 00 00 00 00

define signext i32 @main() !dbg !4 {
  ret i32 42, !dbg !7
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "celqmain.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !DILocation(line: 1, column: 1, scope: !4)
