;; The "MachO Relocatable DWARF" module flag has the same effect on LTO code
;; generation as -macho-relocatable-dwarf: references from one DWARF section
;; into another are relocations, and every __DWARF section is at address 0.
;; a.ll has the flag and b.ll doesn't.

; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: llvm-as a.ll -o a.bc
; RUN: llvm-as b.ll -o b.bc
; RUN: opt -module-summary a.ll -o a.thin.bc
; RUN: opt -module-summary b.ll -o b.thin.bc

;; Full LTO merges the flag with Max, so the object uses relocations even though
;; b.bc doesn't have the flag.
; RUN: llvm-lto -exported-symbol=_a_entry -exported-symbol=_b_sum \
; RUN:   a.bc b.bc -o legacy.o
; RUN: llvm-objdump -r --section=__debug_info legacy.o \
; RUN:   | FileCheck %s --check-prefix=RELOC
; RUN: llvm-readobj --sections legacy.o | FileCheck %s --check-prefix=ZERO

; RUN: llvm-lto2 run a.bc b.bc -o full -r a.bc,_a_entry,plx -r a.bc,_b_sum, \
; RUN:   -r b.bc,_b_sum,plx
; RUN: llvm-objdump -r --section=__debug_info full.0 \
; RUN:   | FileCheck %s --check-prefix=RELOC
; RUN: llvm-readobj --sections full.0 | FileCheck %s --check-prefix=ZERO

;; Without the flag, nothing changes.
; RUN: llvm-lto -exported-symbol=_b_sum b.bc -o legacy-plain.o
; RUN: llvm-objdump -r --section=__debug_info legacy-plain.o \
; RUN:   | FileCheck %s --check-prefix=PLAIN
; RUN: llvm-readobj --sections legacy-plain.o \
; RUN:   | FileCheck %s --check-prefix=NONZERO

;; In ThinLTO, each module uses its own flag, merged with the flags of the
;; modules it imports from. b.thin.bc imports nothing, so it stays plain.
; RUN: llvm-lto -thinlto-action=run -exported-symbol=_a_entry \
; RUN:   -exported-symbol=_b_sum a.thin.bc b.thin.bc
; RUN: llvm-objdump -r --section=__debug_info a.thin.bc.thinlto.o \
; RUN:   | FileCheck %s --check-prefix=RELOC
; RUN: llvm-readobj --sections a.thin.bc.thinlto.o \
; RUN:   | FileCheck %s --check-prefix=ZERO
; RUN: llvm-objdump -r --section=__debug_info b.thin.bc.thinlto.o \
; RUN:   | FileCheck %s --check-prefix=PLAIN
; RUN: llvm-readobj --sections b.thin.bc.thinlto.o \
; RUN:   | FileCheck %s --check-prefix=NONZERO

;; The code-generation-only action creates its TargetMachine separately.
; RUN: llvm-lto -thinlto-action=codegen a.thin.bc -o a.codegen.o
; RUN: llvm-objdump -r --section=__debug_info a.codegen.o \
; RUN:   | FileCheck %s --check-prefix=RELOC
; RUN: llvm-readobj --sections a.codegen.o | FileCheck %s --check-prefix=ZERO
; RUN: llvm-lto -thinlto-action=codegen b.thin.bc -o b.codegen.o
; RUN: llvm-objdump -r --section=__debug_info b.codegen.o \
; RUN:   | FileCheck %s --check-prefix=PLAIN
; RUN: llvm-readobj --sections b.codegen.o | FileCheck %s --check-prefix=NONZERO

; RUN: llvm-lto2 run a.thin.bc b.thin.bc -o thin \
; RUN:   -r a.thin.bc,_a_entry,plx -r a.thin.bc,_b_sum, \
; RUN:   -r b.thin.bc,_b_sum,plx
; RUN: llvm-objdump -r --section=__debug_info thin.1 \
; RUN:   | FileCheck %s --check-prefix=RELOC
; RUN: llvm-readobj --sections thin.1 | FileCheck %s --check-prefix=ZERO
; RUN: llvm-objdump -r --section=__debug_info thin.2 \
; RUN:   | FileCheck %s --check-prefix=PLAIN
; RUN: llvm-readobj --sections thin.2 | FileCheck %s --check-prefix=NONZERO

; RELOC:     RELOCATION RECORDS FOR [__debug_info]:
; RELOC-DAG: ARM64_RELOC_UNSIGNED __debug_abbrev
; RELOC-DAG: ARM64_RELOC_UNSIGNED __debug_line
; RELOC-DAG: ARM64_RELOC_UNSIGNED __debug_str_offs

; PLAIN:     file format mach-o arm64
; PLAIN-NOT: __debug_

; ZERO:      Name: __debug_abbrev (
; ZERO-NEXT: Segment: __DWARF
; ZERO-NEXT: Address: 0x0{{$}}
; ZERO:      Name: __debug_line (
; ZERO-NEXT: Segment: __DWARF
; ZERO-NEXT: Address: 0x0{{$}}

; NONZERO:      Name: __debug_abbrev (
; NONZERO-NEXT: Segment: __DWARF
; NONZERO-NEXT: Address: 0x{{[1-9A-F][0-9A-F]*$}}
; NONZERO:      Name: __debug_line (
; NONZERO-NEXT: Segment: __DWARF
; NONZERO-NEXT: Address: 0x{{[1-9A-F][0-9A-F]*$}}

;--- a.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx15.0.0"

define i32 @a_entry() !dbg !5 {
  %r = call i32 @b_sum(i32 3), !dbg !9
  ret i32 %r, !dbg !9
}

declare i32 @b_sum(i32)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "a.c", directory: "/tmp")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 7, !"MachO Relocatable DWARF", i32 1}
!5 = distinct !DISubprogram(name: "a_entry", scope: !1, file: !1, line: 2, type: !6, scopeLine: 2, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{!8}
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DILocation(line: 2, column: 28, scope: !5)

;--- b.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx15.0.0"

define i32 @b_sum(i32 %n) !dbg !5 {
  ret i32 %n, !dbg !9
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "b.c", directory: "/tmp")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "b_sum", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !8}
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DILocation(line: 1, column: 20, scope: !5)
