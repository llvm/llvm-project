; RUN: llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; DW_TAG_array_type composite not yet supported

; CHECK: OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK: OpExtInst {{.*}} DebugCompilationUnit
; CHECK-NOT: DebugTypePointer

define spir_func void @ptr_to_array() !dbg !10 {
entry:
  %p = alloca ptr addrspace(4), align 8
    #dbg_declare(ptr %p, !11, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !14)
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-type-pointer-composite-pointee.c", directory: "/src", checksumkind: CSK_MD5, checksum: "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"frame-pointer", i32 2}

!6 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !7)
!7 = !{null}
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DICompositeType(tag: DW_TAG_array_type, baseType: !8, size: 256, elements: !12)
!12 = !{!13}
!13 = !DISubrange(count: 8)

!10 = distinct !DISubprogram(name: "ptr_to_array", linkageName: "ptr_to_array", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !15)
!15 = !{}
!11 = !DILocalVariable(name: "pa", scope: !10, file: !1, line: 2, type: !16)
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64, dwarfAddressSpace: 4)
!14 = !DILocation(line: 2, column: 5, scope: !10)
