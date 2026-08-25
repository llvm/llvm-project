;; Check that GlobalISel describes a variable holding the address of a global by
;; naming the symbol, matching the SelectionDAG path in
;; DebugInfo/X86/dbg-value-globaladdr.ll. The IRTranslator builds the same
;; DBG_VALUE, and LiveDebugValues and the DWARF emitter are shared, so the
;; emitted DWARF is identical between the two pipelines.

; RUN: llc -O2 -mtriple=aarch64-apple-macosx -global-isel -stop-after=irtranslator \
; RUN:   < %s | FileCheck %s --check-prefix=MIR
; RUN: llc -O2 -mtriple=aarch64-apple-macosx -global-isel -filetype=obj < %s \
; RUN:   | llvm-dwarfdump - | FileCheck %s --check-prefix=DWARF

@g = global i64 0, align 8
@tls = thread_local global i64 0, align 8

;; Nothing in the function materializes the address. GlobalISel has no dangling
;; debug info recovery at all, so before this the variable was dropped outright.
; MIR-LABEL: name: global_only
; MIR: DBG_VALUE @g, $noreg, ![[#]], !DIExpression()
;
; DWARF-LABEL: DW_AT_name ("global_only")
; DWARF: DW_TAG_variable
; DWARF-NEXT: DW_AT_location (DW_OP_addrx 0x1, DW_OP_stack_value)
; DWARF-NEXT: DW_AT_name ("x")
define void @global_only() !dbg !6 {
entry:
    #dbg_value(ptr @g, !10, !DIExpression(), !11)
  tail call void @sink(ptr null), !dbg !11
  ret void, !dbg !11
}

;; Here the address is materialized into a vreg, but describing the variable by
;; that register would leave the start of the scope uncovered.
; MIR-LABEL: name: global_stored
; MIR: DBG_VALUE @g, $noreg, ![[#]], !DIExpression()
; MIR-NOT: DBG_VALUE
;
; DWARF-LABEL: DW_AT_name ("global_stored")
; DWARF: DW_TAG_variable
; DWARF-NEXT: DW_AT_location (DW_OP_addrx 0x1, DW_OP_stack_value)
; DWARF-NEXT: DW_AT_name ("y")
define void @global_stored() !dbg !12 {
entry:
    #dbg_value(ptr @g, !13, !DIExpression(), !14)
  %box = tail call ptr @alloc(), !dbg !14
  store ptr @g, ptr %box, align 8, !dbg !14
  tail call void @sink(ptr %box), !dbg !14
  ret void, !dbg !14
}

;; A thread-local's address is not a link-time constant: describing it needs
;; DW_OP_form_tls_address, so it must not be named as a symbol here. GlobalISel
;; then drops it, which is what it did for every global before this -- naming the
;; symbol does not regress that case, but does not fix it either.
; MIR-LABEL: name: thread_local_global
; MIR-NOT: DBG_VALUE @tls
; MIR: DBG_VALUE $noreg, 0, ![[#]], !DIExpression()
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
!9 = distinct !DILexicalBlock(scope: !6, file: !3, line: 2, column: 1)
!10 = !DILocalVariable(name: "x", scope: !9, file: !3, line: 2, type: !7)
!11 = !DILocation(line: 2, column: 1, scope: !9)
!12 = distinct !DISubprogram(name: "global_stored", scope: !3, file: !3, line: 5, type: !4, scopeLine: 5, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!13 = !DILocalVariable(name: "y", scope: !18, file: !3, line: 6, type: !7)
!14 = !DILocation(line: 6, column: 1, scope: !18)
!15 = distinct !DISubprogram(name: "thread_local_global", scope: !3, file: !3, line: 10, type: !4, scopeLine: 10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!16 = !DILocalVariable(name: "z", scope: !19, file: !3, line: 11, type: !7)
!17 = !DILocation(line: 11, column: 1, scope: !19)
!18 = distinct !DILexicalBlock(scope: !12, file: !3, line: 6, column: 1)
!19 = distinct !DILexicalBlock(scope: !15, file: !3, line: 11, column: 1)
