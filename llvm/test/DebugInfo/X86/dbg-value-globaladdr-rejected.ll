;; Check the cases where the address of a global is not what its symbol names,
;; and so must not be used as a variable location. Each keeps describing the
;; variable by wherever the address is materialized instead.

;; Nothing anywhere in the output may name a global as a debug value.
; RUN: llc -O2 -mtriple=x86_64-unknown-linux-gnu -stop-after=finalize-isel \
; RUN:   < %s | FileCheck %s --check-prefixes=CHECK,ELF \
; RUN:       --implicit-check-not='DBG_VALUE @'
; RUN: llc -O2 -mtriple=x86_64-pc-windows-gnu -stop-after=finalize-isel \
; RUN:   < %s | FileCheck %s --check-prefixes=CHECK,COFF \
; RUN:       --implicit-check-not='DBG_VALUE @'

@declared = external global i64
@imported = external dllimport global i64
@chosen = ifunc void (), ptr @resolve_chosen

;; A declaration has no definition here for a symbol reference to name.
; CHECK-LABEL: name: external_declaration
; CHECK: DBG_INSTR_REF ![[#]], !DIExpression(DW_OP_LLVM_arg, 0)
define void @external_declaration() !dbg !6 {
entry:
    #dbg_value(ptr @declared, !9, !DIExpression(), !10)
  tail call void @sink(ptr @declared), !dbg !10
  ret void, !dbg !10
}

;; The address of a dllimport'd entity comes out of the import address table,
;; so it is only known once that load has happened. IR can only spell dllimport
;; on something that is a declaration for the linker anyway, so this is the
;; behaviour rather than a distinct branch of the predicate.
; COFF-LABEL: name: dllimport_address
; COFF: DBG_INSTR_REF ![[#]], !DIExpression(DW_OP_LLVM_arg, 0)
define void @dllimport_address() !dbg !11 {
entry:
    #dbg_value(ptr @imported, !12, !DIExpression(), !13)
  tail call void @sink(ptr @imported), !dbg !13
  ret void, !dbg !13
}

;; An ifunc's symbol resolves to whatever its resolver returns at load time,
;; which is not the address of the symbol itself.
; ELF-LABEL: name: ifunc_address
; ELF: DBG_INSTR_REF ![[#]], !DIExpression(DW_OP_LLVM_arg, 0)
define void @ifunc_address() !dbg !14 {
entry:
    #dbg_value(ptr @chosen, !15, !DIExpression(), !16)
  tail call void @sink(ptr @chosen), !dbg !16
  ret void, !dbg !16
}

define internal void @implementation() {
  ret void
}

define internal ptr @resolve_chosen() {
  ret ptr @implementation
}

declare void @sink(ptr)

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 7, !"Dwarf Version", i32 5}
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang", isOptimized: true, emissionKind: FullDebug)
!3 = !DIFile(filename: "t.c", directory: "/")
!4 = !DISubroutineType(types: !{null})
!5 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!6 = distinct !DISubprogram(name: "external_declaration", scope: !3, file: !3, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64)
!8 = distinct !DILexicalBlock(scope: !6, file: !3, line: 2, column: 1)
!9 = !DILocalVariable(name: "p", scope: !8, file: !3, line: 2, type: !7)
!10 = !DILocation(line: 2, column: 1, scope: !8)
!11 = distinct !DISubprogram(name: "dllimport_address", scope: !3, file: !3, line: 5, type: !4, scopeLine: 5, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!12 = !DILocalVariable(name: "q", scope: !17, file: !3, line: 6, type: !7)
!13 = !DILocation(line: 6, column: 1, scope: !17)
!14 = distinct !DISubprogram(name: "ifunc_address", scope: !3, file: !3, line: 9, type: !4, scopeLine: 9, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!15 = !DILocalVariable(name: "r", scope: !18, file: !3, line: 10, type: !7)
!16 = !DILocation(line: 10, column: 1, scope: !18)
!17 = distinct !DILexicalBlock(scope: !11, file: !3, line: 6, column: 1)
!18 = distinct !DILexicalBlock(scope: !14, file: !3, line: 10, column: 1)
