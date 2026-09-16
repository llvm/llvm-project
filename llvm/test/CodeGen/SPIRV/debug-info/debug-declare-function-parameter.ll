; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --implicit-check-not=DebugDeclare
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; A declare whose address is an incoming argument rather than an alloca, which
; is the shape clang produces for a byval parameter on this triple. The
; location register is defined by OpFunctionParameter, so the declare is
; dropped as per spec.

; At the time of writing, spirv-val accepts it, since its rule allows OpVariable or
; OpFunctionParameter, but the spec restricts the Variable operand to
; OpVariable.
;
; spirv64-amd-amdhsa never reaches this shape: its ABI passes the struct byref,
; so the declare lands on the callee's own copy, an alloca.

; CHECK: OpExtInst {{.*}} DebugLocalVariable

target triple = "spirv64-unknown-unknown"

define spir_func void @f(ptr %p) !dbg !5 {
entry:
    #dbg_declare(ptr %p, !9, !DIExpression(), !10)
  ret void, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-declare-function-parameter.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !6)
!6 = !{null, !8}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64, dwarfAddressSpace: 4)
!5 = distinct !DISubprogram(name: "f", linkageName: "f", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!9 = !DILocalVariable(name: "p", arg: 1, scope: !5, file: !1, line: 1, type: !8)
!10 = !DILocation(line: 1, column: 20, scope: !5)
