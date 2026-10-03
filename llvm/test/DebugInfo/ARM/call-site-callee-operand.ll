;; Check that DW_TAG_call_site entries are emitted for direct and indirect
;; calls in both ARM and Thumb modes. On Thumb, the callee of tBL, tBLXi and
;; tBLXr is not operand 0 (operands 0 and 1 are the predicate), so this relies
;; on ARMBaseInstrInfo::getCalleeOperand returning the correct operand.

; RUN: llc -mtriple=armv7-unknown-linux-gnueabi -filetype=obj -o - %s \
; RUN:   | llvm-dwarfdump -debug-info - | FileCheck %s
; RUN: llc -mtriple=thumbv7m-unknown-none-eabi -filetype=obj -o - %s \
; RUN:   | llvm-dwarfdump -debug-info - | FileCheck %s
; RUN: llc -mtriple=thumbv6m-unknown-none-eabi -filetype=obj -o - %s \
; RUN:   | llvm-dwarfdump -debug-info - | FileCheck %s

; CHECK:      DW_TAG_subprogram
; CHECK:        DW_AT_call_all_calls (true)
; CHECK:        DW_AT_name ("caller")

;; Direct call: BL (ARM) / tBL (Thumb).
; CHECK:        DW_TAG_call_site
; CHECK-NEXT:     DW_AT_call_origin ({{.*}} "callee")
; CHECK-NEXT:     DW_AT_call_return_pc

;; Indirect call: BLX (ARM) / tBLXr (Thumb).
; CHECK:        DW_TAG_call_site
; CHECK-NEXT:     DW_AT_call_target (DW_OP_reg{{[0-9]+}} R{{[0-9]+}})
; CHECK-NEXT:     DW_AT_call_return_pc

declare !dbg !14 void @callee()

define void @caller(ptr %fp) !dbg !10 {
entry:
  call void @callee(), !dbg !15
  call void %fp(), !dbg !16
  ret void, !dbg !17
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/tmp")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!10 = distinct !DISubprogram(name: "caller", scope: !1, file: !1, line: 3, type: !11, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!11 = !DISubroutineType(types: !12)
!12 = !{null}
!14 = !DISubprogram(name: "callee", scope: !1, file: !1, line: 1, type: !11, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!15 = !DILocation(line: 4, column: 3, scope: !10)
!16 = !DILocation(line: 5, column: 3, scope: !10)
!17 = !DILocation(line: 6, column: 1, scope: !10)
