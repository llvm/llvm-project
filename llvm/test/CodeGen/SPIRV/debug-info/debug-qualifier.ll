; RUN: llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info --print-after=spirv-nonsemantic-debug-info -O0 -mtriple=spirv64-unknown-unknown %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-MIR
; RUN: llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --check-prefix=CHECK-OPTION
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-MIR-DAG: [[TYPE_VOID:%[0-9]+:type.*]] = OpTypeVoid
; CHECK-MIR-DAG: [[TYPE_I32:%[0-9]+:type.*]] = OpTypeInt 32, 0
; CHECK-MIR-DAG: [[DBG_SOURCE:%[0-9]+:id\(s32\)]] = OpExtInst [[TYPE_VOID]], 3, 35, {{%[0-9]+\:[a-z0-9\(\)]+}}
; CHECK-MIR-DAG: [[DBG_CU:%[0-9]+:id\(s32\)]] = OpExtInst [[TYPE_VOID]], 3, 1, {{%[0-9]+\:[a-z0-9\(\)]+}}, {{%[0-9]+\:[a-z0-9\(\)]+}}, [[DBG_SOURCE]], {{%[0-9]+\:[a-z0-9\(\)]+}}
; CHECK-MIR-DAG: [[STR_INT:%[0-9]+:id\(s32\)]] = OpString 7630441
; CHECK-MIR-DAG: [[DBG_BASIC_INT:%[0-9]+:id\(s32\)]] = OpExtInst [[TYPE_VOID]], 3, 2, [[STR_INT]]
; CHECK-MIR-DAG: [[STR_VAR:%[0-9]+:id\(s32\)]] = OpString 97
; CHECK-MIR-DAG: [[STR_EMPTY:%[0-9]+:id\(s32\)]] = OpString 0
; CHECK-MIR-DAG: [[DBG_GLOBAL:%[0-9]+:id\(s32\)]] = OpExtInst [[TYPE_VOID]], 3, 18, [[STR_VAR]], [[DBG_BASIC_INT]], [[DBG_SOURCE]], {{%[0-9]+\:[a-z0-9\(\)]+}}, {{%[0-9]+\:[a-z0-9\(\)]+}}, [[DBG_CU]]
; CHECK-MIR-DAG: [[DBG_QUAL:%[0-9]+:id\(s32\)]] = OpExtInst [[TYPE_VOID]], 3, 4, [[DBG_BASIC_INT]], {{%[0-9]+\:[a-z0-9\(\)]+}}

; CHECK-SPIRV: [[int_str:%[0-9]+]] = OpString "int"
; CHECK-SPIRV: [[a_str:%[0-9]+]] = OpString "a"
; CHECK-SPIRV-DAG: [[type_void:%[0-9]+]] = OpTypeVoid
; CHECK-SPIRV-DAG: [[type_i32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-SPIRV: [[dbg_src:%[0-9]+]] = OpExtInst [[type_void]] %[[#]] DebugSource
; CHECK-SPIRV: [[dbg_cu:%[0-9]+]] = OpExtInst [[type_void]] %[[#]] DebugCompilationUnit %[[#]] %[[#]] [[dbg_src]]
; CHECK-SPIRV: [[dbg_int:%[0-9]+]] = OpExtInst [[type_void]] %[[#]] DebugTypeBasic [[int_str]]
; CHECK-SPIRV: [[dbg_none:%[0-9]+]] = OpExtInst [[type_void]] %[[#]] DebugInfoNone
; CHECK-SPIRV: [[dbg_qual:%[0-9]+]] = OpExtInst [[type_void]] %[[#]] DebugTypeQualifier [[dbg_int]]

; CHECK-OPTION-NOT: OpExtInstImport "NonSemantic.Shader.DebugInfo.100"

@a = dso_local constant i32 10, align 4, !dbg !0

define dso_local i32 @main() !dbg !15 {
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  ret i32 0, !dbg !18
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9, !10, !11, !12, !13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 1, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version XX.X.XXXX (FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "debugqualifier.c", directory: "/AAAAAAAAAA/BBBBBBBB/CCCCCCCCC", checksumkind: CSK_MD5, checksum: "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
!4 = !{!0}
!5 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !6, flags: DIFlagPublic)
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed, flags: DIFlagPublic)
!7 = !{i32 7, !"Dwarf Version", i32 5}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{i32 8, !"PIC Level", i32 2}
!11 = !{i32 7, !"PIE Level", i32 2}
!12 = !{i32 7, !"uwtable", i32 2}
!13 = !{i32 7, !"frame-pointer", i32 2}
!15 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 2, type: !16, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !2, flags: DIFlagPublic)
!16 = !DISubroutineType(types: !17, flags: DIFlagPublic)
!17 = !{!6}
!18 = !DILocation(line: 3, column: 3, scope: !15)
