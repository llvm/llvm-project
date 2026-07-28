; RUN: llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; emitDebugTypeArray emits nothing when the element type is not in DebugTypeRegs.
; The element here is a pointer with no DWARF address space, which
; emitDebugTypePointer skips for lack of a storage class. The pointer stays out
; of DebugTypeRegs regardless of later type support, so the array is skipped.

; CHECK-SPIRV: OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-SPIRV-NOT: DebugTypeArray

define spir_func void @test() !dbg !6 {
entry:
  %a = alloca [4 x ptr], align 8
    #dbg_declare(ptr %a, !10, !DIExpression(), !14)
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_HLSL, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "array.hlsl", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 1, type: !7, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !9)
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = !{}
!10 = !DILocalVariable(name: "a", scope: !6, file: !1, line: 2, type: !11)
!11 = !DICompositeType(tag: DW_TAG_array_type, baseType: !12, size: 256, elements: !13)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!13 = !{!DISubrange(count: 4)}
!14 = !DILocation(line: 2, column: 10, scope: !6)
!15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
