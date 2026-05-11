; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --check-prefix=CHECK
; TODO(#109287): spirv-val coverage remains disabled for DebugTypePointer with
; DebugInfoNone as the base type.
;
; Pointer parameter with null baseType should lower to DebugTypePointer using
; DebugInfoNone as Base Type, and still be consumed by DebugTypeFunction.

; CHECK: [[ext:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[type_void:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[type_int32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[flag_zero:%[0-9]+]] = OpConstant [[type_int32]] 0{{$}}
; CHECK-DAG: [[debug_none:%[0-9]+]] = OpExtInst [[type_void]] [[ext]] DebugInfoNone
; CHECK-DAG: [[storage_class:%[0-9]+]] = OpConstant [[type_int32]] 8{{$}}
; CHECK: [[dbg_ptr:%[0-9]+]] = OpExtInst [[type_void]] [[ext]] DebugTypePointer [[debug_none]] [[storage_class]] [[flag_zero]]
; CHECK: OpExtInst [[type_void]] [[ext]] DebugTypeFunction [[flag_zero]] [[type_void]] [[dbg_ptr]]

target triple = "spirv64-unknown-unknown"

define spir_func void @ptr_null_base(ptr addrspace(4) %p) !dbg !10 {
entry:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version XX.X", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-type-function-pointer-debug-none-base.c", directory: "/AAAAAAAAAA/BBBBBBBB/CCCCCCCCC", checksumkind: CSK_MD5, checksum: "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"frame-pointer", i32 2}

!6 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !7)
!7 = !{null, !8}
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64, dwarfAddressSpace: 4)

!10 = distinct !DISubprogram(name: "ptr_null_base", linkageName: "ptr_null_base", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
