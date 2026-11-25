; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: llc -mtriple=x86_64-linux-gnu -filetype=obj middle.ll -o middle.o
; RUN: llvm-dwarfdump --debug-line middle.o 2>&1 | FileCheck %s --check-prefix=MIDDLE --implicit-check-not=warning:
; RUN: llvm-dwarfdump --verify middle.o | FileCheck %s --check-prefix=VERIFY
; RUN: llc -mtriple=x86_64-linux-gnu -filetype=obj last.ll -o last.o
; RUN: llvm-dwarfdump --debug-line last.o 2>&1 | FileCheck %s --check-prefix=LAST --implicit-check-not=warning:
; RUN: llvm-dwarfdump --verify last.o | FileCheck %s --check-prefix=VERIFY

; VERIFY: No errors.

;; A line entry created in subsection 1, here by a .loc directive in inline asm,
;; has an address after the end of the function, where DwarfDebug adds an end
;; entry to terminate the CU's range. It must be emitted after that end entry,
;; in a separate sequence that is terminated at the end of the section.

;; Generated from:
;; int g;
;; void f(void) {
;;   g = 1;
;;   asm volatile(".subsection 1\n.loc 1 100 0\nnop\n.subsection 0");
;;   g = 2;
;; }

; MIDDLE:      Address            Line   Column File   ISA Discriminator OpIndex Flags
; MIDDLE-NEXT: ------------------ ------ ------ ------ --- ------------- ------- -------------
; MIDDLE-NEXT: 0x0000000000000000      3      5      1   0             0       0  is_stmt prologue_end
; MIDDLE-NEXT: 0x000000000000000a      5      5      1   0             0       0  is_stmt
; MIDDLE-NEXT: 0x0000000000000014      6      1      1   0             0       0  is_stmt
; MIDDLE-NEXT: 0x0000000000000015      6      1      1   0             0       0  is_stmt end_sequence
; MIDDLE-NEXT: 0x0000000000000015    100      0      1   0             0       0  is_stmt
; MIDDLE-NEXT: 0x0000000000000016    100      0      1   0             0       0  is_stmt end_sequence
; MIDDLE-EMPTY:

;; The last line entry before the end entry is created in subsection 1, as ret
;; shares the location of the asm statement and gets no .loc of its own. The
;; end entry, cloned from that entry, nevertheless belongs to subsection 0,
;; where its label is, so the subsection 1 code at 0xb-0xd keeps its line info.

;; Generated from:
;; int g;
;; void f(void) {
;;   g = 1;
;;   asm volatile(".subsection 1\n.loc 1 100 0\nnop\nnop\n.subsection 0");
;; }

; LAST:      Address            Line   Column File   ISA Discriminator OpIndex Flags
; LAST-NEXT: ------------------ ------ ------ ------ --- ------------- ------- -------------
; LAST-NEXT: 0x0000000000000000      3      5      1   0             0       0  is_stmt prologue_end
; LAST-NEXT: 0x000000000000000b      3      5      1   0             0       0  is_stmt end_sequence
; LAST-NEXT: 0x000000000000000b    100      0      1   0             0       0  is_stmt
; LAST-NEXT: 0x000000000000000d    100      0      1   0             0       0  is_stmt end_sequence
; LAST-EMPTY:

;--- middle.ll
@g = dso_local global i32 0, align 4

define dso_local void @f() !dbg !5 {
  store i32 1, ptr @g, align 4, !dbg !8
  call void asm sideeffect ".subsection 1\0A.loc 1 100 0\0Anop\0A.subsection 0", ""(), !dbg !9
  store i32 2, ptr @g, align 4, !dbg !10
  ret void, !dbg !11
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 2, type: !6, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !DILocation(line: 3, column: 5, scope: !5)
!9 = !DILocation(line: 4, column: 3, scope: !5)
!10 = !DILocation(line: 5, column: 5, scope: !5)
!11 = !DILocation(line: 6, column: 1, scope: !5)

;--- last.ll
@g = dso_local global i32 0, align 4

define dso_local void @f() !dbg !5 {
  store i32 1, ptr @g, align 4, !dbg !8
  call void asm sideeffect ".subsection 1\0A.loc 1 100 0\0Anop\0Anop\0A.subsection 0", ""(), !dbg !9
  ret void, !dbg !9
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 2, type: !6, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !DILocation(line: 3, column: 5, scope: !5)
!9 = !DILocation(line: 4, column: 3, scope: !5)
