;; Under PIC a WebAssembly global lives at __memory_base plus its symbol's
;; address, so naming the symbol alone would describe a link-time offset where
;; the source variable holds a runtime address. Check that a variable holding
;; that address is left to the location the address is materialized into, and
;; that non-PIC, where the symbol does name the address, still uses it.

; RUN: llc -O2 -mtriple=wasm32-unknown-unknown -relocation-model=pic \
; RUN:   -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=PIC-MIR \
; RUN:       --implicit-check-not='DBG_VALUE @g'
; RUN: llc -O2 -mtriple=wasm32-unknown-unknown -relocation-model=pic \
; RUN:   -filetype=obj < %s | llvm-dwarfdump - | FileCheck %s --check-prefix=PIC
; RUN: llc -O2 -mtriple=wasm32-unknown-unknown -stop-after=finalize-isel < %s \
; RUN:   | FileCheck %s --check-prefix=STATIC-MIR
; RUN: llc -O2 -mtriple=wasm32-unknown-unknown -filetype=obj < %s \
; RUN:   | llvm-dwarfdump - | FileCheck %s --check-prefix=STATIC

target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-i128:128-n32:64-S128-ni:1:10:20"

@g = global i32 0, align 4, !dbg !0

; PIC-MIR-LABEL: name: global_stored
; PIC-MIR: DBG_VALUE %{{[0-9]+}}, $noreg, ![[#]], !DIExpression()
;
;; The variable keeps the location the address is computed into. The global
;; variable's own location shows what a description of that address takes: the
;; symbol is only half of it, and DW_OP_WASM_location needs a relocation that a
;; location list cannot carry.
; PIC-LABEL: DW_AT_name ("global_stored")
; PIC: DW_TAG_variable
; PIC-NEXT: DW_AT_location (indexed (0x0) loclist = 0x{{[0-9a-f]+}}:
; PIC-NEXT: DW_OP_WASM_location 0x0 0x0, DW_OP_stack_value)
; PIC-NEXT: DW_AT_name ("p")
; PIC: DW_AT_name ("g")
; PIC: DW_AT_location (DW_OP_WASM_location 0x3 0x1, DW_OP_addrx 0x1, DW_OP_plus)
;
; STATIC-MIR-LABEL: name: global_stored
; STATIC-MIR: DBG_VALUE @g, $noreg, ![[#]], !DIExpression()
;
; STATIC-LABEL: DW_AT_name ("global_stored")
; STATIC: DW_TAG_variable
; STATIC-NEXT: DW_AT_location (DW_OP_addrx 0x1, DW_OP_stack_value)
; STATIC-NEXT: DW_AT_name ("p")
define void @global_stored() !dbg !9 {
entry:
    #dbg_value(ptr @g, !12, !DIExpression(), !13)
  %box = tail call ptr @alloc(), !dbg !13
  store ptr @g, ptr %box, align 4, !dbg !13
  tail call void @sink(ptr %box), !dbg !13
  ret void, !dbg !13
}

declare void @sink(ptr)
declare ptr @alloc()

!llvm.module.flags = !{!5, !6}
!llvm.dbg.cu = !{!2}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !3, line: 1, type: !8, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang", isOptimized: true, emissionKind: FullDebug, globals: !4)
!3 = !DIFile(filename: "t.c", directory: "/")
!4 = !{!0}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 7, !"Dwarf Version", i32 5}
!7 = !DISubroutineType(types: !{null})
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = distinct !DISubprogram(name: "global_stored", scope: !3, file: !3, line: 5, type: !7, scopeLine: 5, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 32)
!11 = distinct !DILexicalBlock(scope: !9, file: !3, line: 6, column: 1)
!12 = !DILocalVariable(name: "p", scope: !11, file: !3, line: 6, type: !10)
!13 = !DILocation(line: 6, column: 1, scope: !11)
