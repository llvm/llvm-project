; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s

; A typedef chain that closes on itself. Verifier::visitDIDerivedType checks
; only that a base type is a type, so this passes the IR verifier, and
; stripToScalarType() has to terminate on it rather than walk the cycle
; forever. No source language can write one, since a typedef name is not in
; scope in its own declaration.
;
; The variable's type never resolves to a DIBasicType, so the constant is not
; named and the record is dropped. Reaching the end of the module at all is
; what this test is for.

; CHECK: OpFunctionEnd

target triple = "spirv64-unknown-unknown"

define spir_func void @cyclic_typedef() !dbg !5 {
entry:
    #dbg_value(i32 7, !9, !DIExpression(), !10)
  ret void, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-value-typedef-cycle.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !6)
!6 = !{null}
!5 = distinct !DISubprogram(name: "cyclic_typedef", linkageName: "cyclic_typedef", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!7 = !DIDerivedType(tag: DW_TAG_typedef, name: "a", baseType: !8)
!8 = !DIDerivedType(tag: DW_TAG_typedef, name: "b", baseType: !7)
!9 = !DILocalVariable(name: "cyclic", scope: !5, file: !1, line: 2, type: !7)
!10 = !DILocation(line: 3, column: 3, scope: !5)
