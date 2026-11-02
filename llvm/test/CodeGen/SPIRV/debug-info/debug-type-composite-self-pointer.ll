; RUN: llc --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --implicit-check-not=DebugTypeMember
; RUN: %if spirv-tools %{ llc --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; struct S { S *p; } with a local of type S*. p points back at S while S is
; still being emitted, so p is left out. The local still gets a pointer to S.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[S_NAME:%[0-9]+]] = OpString "S"
; CHECK-DAG: [[LOCAL_NAME:%[0-9]+]] = OpString "local"
; CHECK:     [[S:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugTypeComposite [[S_NAME]]
; CHECK:     [[PTR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugTypePointer [[S]]
; CHECK:     OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[LOCAL_NAME]] [[PTR]]

target triple = "spirv64-unknown-unknown"

define spir_func void @f() !dbg !10 {
entry:
  %local = alloca ptr addrspace(4), align 8
    #dbg_declare(ptr %local, !12, !DIExpression(), !14)
  ret void, !dbg !14
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !4)
!1 = !DIFile(filename: "self-pointer.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}

!4 = !{!5}
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !1, line: 1, size: 64, elements: !6)
!6 = !{!7}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "p", file: !1, line: 1, baseType: !8, size: 64)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64, dwarfAddressSpace: 4)

!9 = !DISubroutineType(types: !11)
!11 = !{null}
!10 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 3, type: !9, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !13)
!12 = !DILocalVariable(name: "local", scope: !10, file: !1, line: 4, type: !8)
!13 = !{!12}
!14 = !DILocation(line: 4, column: 1, scope: !10)
