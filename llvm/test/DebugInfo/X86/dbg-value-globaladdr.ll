;; Check that a dbg_value naming the address of a global describes the variable
;; directly, for the whole of its scope, rather than waiting for the address to
;; be materialized into a register.

; RUN: llc -O2 -mtriple=x86_64-unknown-linux-gnu -stop-after=finalize-isel < %s \
; RUN:   | FileCheck %s --check-prefix=MIR
;; The GlobalISel equivalent is DebugInfo/AArch64/dbg-value-globaladdr-gisel.ll.
; RUN: llc -O2 -mtriple=x86_64-unknown-linux-gnu -filetype=obj < %s \
; RUN:   | llvm-dwarfdump - | FileCheck %s --check-prefix=DWARF5
; RUN: llc -O2 -mtriple=x86_64-unknown-linux-gnu -dwarf-version=4 -filetype=obj < %s \
; RUN:   | llvm-dwarfdump - | FileCheck %s --check-prefix=DWARF4

@g = external global i64, align 8
@tls = external thread_local global i64, align 8

;; Nothing in the function materializes the address, so before this was
;; described the variable was dropped entirely.
; MIR-LABEL: name: global_only
; MIR: DBG_VALUE @g, $noreg, ![[#]], !DIExpression()
;
;; DW_OP_addrx indexes .debug_addr, which is relocated to @g.
; DWARF5-LABEL: DW_AT_name ("global_only")
; DWARF5: DW_TAG_variable
; DWARF5-NEXT: DW_AT_location (DW_OP_addrx 0x1, DW_OP_stack_value)
; DWARF5-NEXT: DW_AT_name ("x")
;
;; Before DWARF 5 there is no .debug_addr to index outside of split DWARF, so
;; the address is spelled as a relocated DW_OP_addr instead.
; DWARF4-LABEL: DW_AT_name ("global_only")
; DWARF4: DW_TAG_variable
; DWARF4-NEXT: DW_AT_location (DW_OP_addr 0x0, DW_OP_stack_value)
; DWARF4-NEXT: DW_AT_name ("x")
define void @global_only() !dbg !6 {
entry:
    #dbg_value(ptr @g, !10, !DIExpression(), !11)
  tail call void @sink(ptr null), !dbg !11
  ret void, !dbg !11
}

;; Here the address is materialized, but describing the variable by the
;; resulting register would leave the start of the scope uncovered.
; MIR-LABEL: name: global_stored
; MIR: DBG_VALUE @g, $noreg, ![[#]], !DIExpression()
; MIR-NOT: DBG_VALUE
;
; DWARF5-LABEL: DW_AT_name ("global_stored")
; DWARF5: DW_TAG_variable
; DWARF5-NEXT: DW_AT_location (DW_OP_addrx 0x1, DW_OP_stack_value)
; DWARF5-NEXT: DW_AT_name ("y")
define void @global_stored() !dbg !12 {
entry:
    #dbg_value(ptr @g, !13, !DIExpression(), !14)
  %box = tail call ptr @alloc(), !dbg !14
  store ptr @g, ptr %box, align 8, !dbg !14
  tail call void @sink(ptr %box), !dbg !14
  ret void, !dbg !14
}

;; A thread-local's address is not a link-time constant: describing it needs
;; DW_OP_form_tls_address, so it must not be named as a symbol here.
; MIR-LABEL: name: thread_local_global
; MIR-NOT: DBG_VALUE @tls
;; It still gets a register location recovered from the dangling debug info.
; MIR: DBG_INSTR_REF ![[#]], !DIExpression(DW_OP_LLVM_arg, 0)
define void @thread_local_global() !dbg !15 {
entry:
    #dbg_value(ptr @tls, !16, !DIExpression(), !17)
  tail call void @sink(ptr @tls), !dbg !17
  ret void, !dbg !17
}

declare void @sink(ptr)
declare ptr @alloc()

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 7, !"Dwarf Version", i32 5}
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang", isOptimized: true, emissionKind: FullDebug)
!3 = !DIFile(filename: "t.c", directory: "/")
!4 = !DISubroutineType(types: !5)
!5 = !{null}
!6 = distinct !DISubprogram(name: "global_only", scope: !3, file: !3, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64)
!8 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!10 = !DILocalVariable(name: "x", scope: !9, file: !3, line: 2, type: !7)
!9 = distinct !DILexicalBlock(scope: !6, file: !3, line: 2, column: 1)
!11 = !DILocation(line: 2, column: 1, scope: !9)
!12 = distinct !DISubprogram(name: "global_stored", scope: !3, file: !3, line: 5, type: !4, scopeLine: 5, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!13 = !DILocalVariable(name: "y", scope: !18, file: !3, line: 6, type: !7)
!18 = distinct !DILexicalBlock(scope: !12, file: !3, line: 6, column: 1)
!14 = !DILocation(line: 6, column: 1, scope: !18)
!15 = distinct !DISubprogram(name: "thread_local_global", scope: !3, file: !3, line: 10, type: !4, scopeLine: 10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!16 = !DILocalVariable(name: "z", scope: !19, file: !3, line: 11, type: !7)
!19 = distinct !DILexicalBlock(scope: !15, file: !3, line: 11, column: 1)
!17 = !DILocation(line: 11, column: 1, scope: !19)
