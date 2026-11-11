; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --implicit-check-not=DebugLocalVariable --implicit-check-not=DebugDeclare
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; A declare on live storage, so the OpVariable is there, but the variable's
; type has no DWARF address space and its DebugTypePointer is skipped.

; CHECK: OpVariable {{%[0-9]+}} Function
; CHECK: OpExtInst {{.*}} DebugFunctionDefinition

target triple = "spirv64-unknown-unknown"

define spir_func void @f() !dbg !5 {
entry:
  %p = alloca ptr, align 8
    #dbg_declare(ptr %p, !9, !DIExpression(), !10)
  store ptr null, ptr %p, align 8, !dbg !10
  ret void, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-declare-skip-type.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64)
!5 = distinct !DISubprogram(name: "f", linkageName: "f", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!9 = !DILocalVariable(name: "p", scope: !5, file: !1, line: 2, type: !8)
!10 = !DILocation(line: 3, column: 1, scope: !5)
