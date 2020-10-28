; RUN: llc %s -mtriple=x86_64-unknown-unknown -o - -stop-before=finalize-isel \
; RUN:   | FileCheck %s --check-prefix=NORMAL \
; RUN:     --implicit-check-not=debug-instr-number \
; RUN:     --implicit-check-not=DBG_INSTR_REF
; RUN: llc %s -mtriple=x86_64-unknown-unknown -o - -stop-before=finalize-isel \
; RUN:     -experimental-debug-variable-locations -verify-machineinstrs \
; RUN:   | FileCheck %s --check-prefix=INSTRREF \
; RUN:     --implicit-check-not=DBG_VALUE

; Test that SelectionDAG produces DBG_VALUEs normally, but DBG_INSTR_REFs when
; asked.

; NORMAL:      %[[REG0:[0-9]+]]:gr32 = ADD32rr
; NORMAL-NEXT: DBG_VALUE %[[REG0]]
; NORMAL-NEXT: %[[REG1:[0-9]+]]:gr32 = ADD32rr
; NORMAL-NEXT: DBG_VALUE %[[REG1]]

; Note that I'm baking in an assumption of one-based ordering here. We could
; capture and check for the instruction numbers, we'd rely on machine verifier
; ensuring there were no duplicates.

; INSTRREF:      ADD32rr
; INSTRREF-SAME: debug-instr-number 1
; INSTRREF-NEXT: DBG_INSTR_REF 1, 0
; INSTRREF-NEXT: ADD32rr
; INSTRREF-SAME: debug-instr-number 2
; INSTRREF-NEXT: DBG_INSTR_REF 2, 0

declare void @llvm.dbg.value(metadata, metadata, metadata)

define i32 @foo(i32 %bar, i32 %baz, i32 %qux) !dbg !7 {
entry:
  %0 = add i32 %bar, %baz, !dbg !14
  call void @llvm.dbg.value(metadata i32 %0, metadata !13, metadata !DIExpression()), !dbg !14
  %1 = add i32 %0, %qux
  call void @llvm.dbg.value(metadata i32 %1, metadata !13, metadata !DIExpression()), !dbg !14
  ret i32 %1, !dbg !14
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "exprconflict.c", directory: "/home/jmorse")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 5, type: !8, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!13}
!13 = !DILocalVariable(name: "baz", scope: !7, file: !1, line: 6, type: !10)
!14 = !DILocation(line: 1, scope: !7)
