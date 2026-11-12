; RUN: llc --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; void f(const int *c, volatile int *v, int *restrict r, const void *cv,
;        const volatile int *cvi);
; plus an _Atomic int that only appears in the compile unit's retainedTypes.
; const volatile int nests a const qualifier around a volatile one.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[F:%[0-9]+]] = OpString "f"
; CHECK-DAG: [[C0:%[0-9]+]] = OpConstant [[I32]] 0{{$}}
; CHECK-DAG: [[C1:%[0-9]+]] = OpConstant [[I32]] 1{{$}}
; CHECK-DAG: [[C2:%[0-9]+]] = OpConstant [[I32]] 2{{$}}
; CHECK-DAG: [[C3:%[0-9]+]] = OpConstant [[I32]] 3{{$}}
; CHECK-DAG: [[NONE:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugInfoNone
; CHECK-DAG: [[INT:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugTypeBasic
; CHECK-DAG: [[CONST:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugTypeQualifier [[INT]] [[C0]]
; CHECK-DAG: [[VOLATILE:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugTypeQualifier [[INT]] [[C1]]
; CHECK-DAG: [[CONSTVOL:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugTypeQualifier [[VOLATILE]] [[C0]]
; CHECK-DAG: [[PTR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugTypePointer [[INT]]
; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugTypeQualifier [[PTR]] [[C2]]
; CHECK-DAG: [[CONSTVOID:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugTypeQualifier [[NONE]] [[C0]]
; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugTypeQualifier [[INT]] [[C3]]
; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugTypePointer [[CONST]]
; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugTypePointer [[VOLATILE]]
; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugTypePointer [[CONSTVOID]]
; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugTypePointer [[CONSTVOL]]
; CHECK-DAG: [[DF:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugFunction [[F]]
; CHECK: OpExtInst [[VOID]] [[EXT]] DebugFunctionDefinition [[DF]]

define spir_func void @f(ptr addrspace(4) %c, ptr addrspace(4) %v, ptr addrspace(4) %r, ptr addrspace(4) %cv, ptr addrspace(4) %cvi) !dbg !10 {
entry:
  ret void, !dbg !20
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1, !2}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !4)
!1 = !{i32 7, !"Dwarf Version", i32 5}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIFile(filename: "qualifier.c", directory: "/src")
!4 = !{!19}
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !5)
!7 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !5)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64, dwarfAddressSpace: 4)
!9 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !8)
!10 = distinct !DISubprogram(name: "f", scope: !3, file: !3, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!11 = !DISubroutineType(types: !12)
!12 = !{null, !13, !14, !9, !16, !22}
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64, dwarfAddressSpace: 4)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64, dwarfAddressSpace: 4)
!15 = !DIDerivedType(tag: DW_TAG_const_type, baseType: null)
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64, dwarfAddressSpace: 4)
!19 = !DIDerivedType(tag: DW_TAG_atomic_type, baseType: !5)
!20 = !DILocation(line: 1, column: 1, scope: !10)
!21 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !7)
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !21, size: 64, dwarfAddressSpace: 4)
