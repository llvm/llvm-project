;; Check a dbg_value of the address of an external global in a block that does
;; not itself use the global. A declaration cannot be named as a location (see
;; DebugInfo/X86/dbg-value-globaladdr-rejected.ll), so the variable has to be
;; described by a register holding the address. Such a register exists here:
;; the GOT load in the entry block also feeds the store in %use, so it is live
;; throughout %mid.
;;
;; FIXME: SelectionDAG drops the location in %mid, because it only finds
;; constants materialized in the current block.

; RUN: llc -O2 -mtriple=aarch64-apple-macosx -stop-after=finalize-isel < %s \
; RUN:   | FileCheck %s --check-prefix=SDAG
; RUN: llc -O2 -mtriple=aarch64-apple-macosx -global-isel \
; RUN:   -stop-after=ir-translator < %s | FileCheck %s --check-prefix=GISEL
; RUN: llc -O2 -mtriple=aarch64-apple-macosx < %s \
; RUN:   | FileCheck %s --check-prefixes=ASM,ASM-SDAG --implicit-check-not=DEBUG_VALUE
; RUN: llc -O2 -mtriple=aarch64-apple-macosx -global-isel < %s \
; RUN:   | FileCheck %s --check-prefixes=ASM,ASM-GISEL --implicit-check-not=DEBUG_VALUE

@ext = external global i64

;; SelectionDAG materializes the address separately in each block that uses it.
; SDAG-LABEL: bb.0.entry:
; SDAG: LOADgot target-flags(aarch64-got) @ext
; SDAG-LABEL: bb.1.mid:
; SDAG: DBG_VALUE $noreg, $noreg, ![[#]], !DIExpression()
; SDAG-LABEL: bb.2.use:
; SDAG: LOADgot target-flags(aarch64-got) @ext

;; GlobalISel materializes the address once, in the entry block, so the vreg
;; holding it dominates %mid.
; GISEL-LABEL: bb.1.entry:
; GISEL: %[[ADDR:[0-9]+]]:_(p0) = G_GLOBAL_VALUE @ext
; GISEL-LABEL: bb.2.mid:
; GISEL: DBG_VALUE %[[ADDR]](p0), $noreg, ![[#]], !DIExpression()
; GISEL-LABEL: bb.3.use:
; GISEL: G_STORE %[[ADDR]](p0)

;; MachineCSE folds the second GOT load into the first, so one register holds
;; the address from the entry block through to the store in %use.
; ASM-LABEL: _other_block:
; ASM: ldr x[[ADDR:[0-9]+]], [x[[ADDR]], _ext@GOTPAGEOFF]
; ASM-LABEL: ; %mid
; ASM-SDAG: ;DEBUG_VALUE: x <- undef
; ASM-GISEL: ;DEBUG_VALUE: x <- $x[[ADDR]]
; ASM-LABEL: ; %use
; ASM-GISEL-NEXT: ;DEBUG_VALUE: x <- $x[[ADDR]]
; ASM-NEXT: str x[[ADDR]], [x{{[0-9]+}}]
define void @other_block(ptr %p, i1 %c) !dbg !6 {
entry:
  store ptr @ext, ptr %p, align 8, !dbg !10
  br i1 %c, label %mid, label %exit, !dbg !10

mid:
    #dbg_value(ptr @ext, !9, !DIExpression(), !10)
  %d = tail call i1 @work(), !dbg !10
  br i1 %d, label %use, label %exit, !dbg !10

use:
  store ptr @ext, ptr %p, align 8, !dbg !10
  br label %exit, !dbg !10

exit:
  ret void, !dbg !10
}

declare i1 @work()

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 7, !"Dwarf Version", i32 5}
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang", isOptimized: true, emissionKind: FullDebug)
!3 = !DIFile(filename: "t.c", directory: "/")
!4 = !DISubroutineType(types: !{null})
!5 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!6 = distinct !DISubprogram(name: "other_block", scope: !3, file: !3, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64)
!8 = distinct !DILexicalBlock(scope: !6, file: !3, line: 2, column: 1)
!9 = !DILocalVariable(name: "x", scope: !8, file: !3, line: 2, type: !7)
!10 = !DILocation(line: 2, column: 1, scope: !8)
