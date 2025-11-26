; RUN: llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info --print-after=spirv-nonsemantic-debug-info -O0 -mtriple=spirv64-unknown-unknown -stop-after=spirv-nonsemantic-debug-info  %s -o - | FileCheck %s --check-prefix=CHECK-MIR
; RUN: llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --check-prefix=CHECK-OPTION
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-MIR: [[VOID:%[0-9]+]]:type(s64) = OpTypeVoid
; CHECK-MIR: [[I32:%[0-9]+]]:type = OpTypeInt 32, 0
; CHECK-MIR: [[STR:%[0-9]+]]:id(s32) = OpString 1094795567, 1094795585
; CHECK-MIR: [[DBG_SOURCE:%[0-9]+]]:id(s32) = OpExtInst [[VOID]](s64), 3, 35, [[STR]](s32)
; CHECK-MIR: [[DBG_CU:%[0-9]+]]:id(s32) = OpExtInst [[VOID]](s64), 3, 1
; CHECK-MIR: [[STR_INT:%[0-9]+]]:id(s32) = OpString 7630441
; CHECK-MIR: [[DBG_INT:%[0-9]+]]:id(s32) = OpExtInst [[VOID]](s64), 3, 2, [[STR_INT]](s32)
; CHECK-MIR: [[DBG_ARRAY:%[0-9]+]]:id(s32) = OpExtInst [[VOID]](s64), 3, 5, [[DBG_INT]](s32), {{%[0-9]+}}
; CHECK-MIR: [[STR_STRUCT:%[0-9]+]]:id(s32) = OpString 1635017060, 0
; CHECK-MIR: [[DBG_STRUCT:%[0-9]+]]:id(s32) = OpExtInst [[VOID]](s64), 3, 11, [[STR_STRUCT]](s32), [[DBG_ARRAY]](s32), [[DBG_SOURCE]](s32), {{%[0-9]+}}
; CHECK-MIR: [[DBG_FUNC:%[0-9]+]]:id(s32) = OpExtInst [[VOID]](s64), 3, 10, {{%[0-9]+}}(s32), {{%[0-9]+}}(s32), [[DBG_SOURCE]](s32)
; CHECK-MIR: [[DBG_INFO:%[0-9]+]]:id(s32) = OpExtInst [[VOID]](s64), 3, 0
; CHECK-MIR: [[DBG_TEMPL:%[0-9]+]]:id(s32) = OpExtInst [[VOID]](s64), 3, 15, {{%[0-9]+}}(s32), {{%[0-9]+}}(s32), [[DBG_INFO]](s32)
; CHECK-MIR: [[STR_X:%[0-9]+]]:id(s32) = OpString 78
; CHECK-MIR: [[DBG_X:%[0-9]+]]:id(s32) = OpExtInst [[VOID]](s64), 3, 15, [[STR_X]](s32), [[DBG_INT]](s32)
; CHECK-MIR: [[DBG_COMPOSITE:%[0-9]+]]:id(s32) = OpExtInst [[VOID]](s64), 3, 14, [[DBG_FUNC]](s32), [[DBG_TEMPL]](s32), [[DBG_X]](s32)

; CHECK-SPIRV: [[data_str:%[0-9]+]] = OpString "data"
; CHECK-SPIRV: [[T_str:%[0-9]+]] = OpString "T"
; CHECK-SPIRV: [[N_str:%[0-9]+]] = OpString "N"
; CHECK-SPIRV: [[fa_global_str:%[0-9]+]] = OpString "fa_global"
; CHECK-SPIRV-DAG: [[type_int:%[0-9]+]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: [[type_void:%[0-9]+]] = OpTypeVoid
; CHECK-SPIRV: [[dbg_src:%[0-9]+]] = OpExtInst [[type_void]] %[[#]] DebugSource %[[#]]
; CHECK-SPIRV: [[dbg_cu:%[0-9]+]] = OpExtInst [[type_void]] %[[#]] DebugCompilationUnit %[[#]] %[[#]] [[dbg_src]] %[[#]]
; CHECK-SPIRV: [[dbg_int:%[0-9]+]] = OpExtInst [[type_void]] %[[#]] DebugTypeBasic %[[#]] %[[#]] %[[#]] %[[#]]
; CHECK-SPIRV: [[dbg_arr:%[0-9]+]] = OpExtInst [[type_void]] %[[#]] DebugTypeArray [[dbg_int]] %[[#]]
; CHECK-SPIRV: [[dbg_member:%[0-9]+]] = OpExtInst [[type_void]] %[[#]] DebugTypeMember [[data_str]] [[dbg_arr]] [[dbg_src]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#]]
; CHECK-SPIRV: [[dbg_comp:%[0-9]+]] = OpExtInst [[type_void]] %[[#]] DebugTypeComposite %[[#]] %[[#]] [[dbg_src]] %[[#]] %[[#]] [[dbg_cu]] %[[#]] %[[#]] %[[#]] [[dbg_member]]
; CHECK-SPIRV: [[dbg_none_1:%[0-9]+]] = OpExtInst [[type_void]] %[[#]] DebugInfoNone
; CHECK-SPIRV: [[dbg_tparam_T:%[0-9]+]] = OpExtInst [[type_void]] %[[#]] DebugTypeTemplateParameter [[T_str]] [[dbg_int]] [[dbg_none_1]] [[dbg_src]] %[[#]] %[[#]]
; CHECK-SPIRV: [[dbg_tparam_N:%[0-9]+]] = OpExtInst [[type_void]] %[[#]] DebugTypeTemplateParameter [[N_str]] [[dbg_int]] %[[#]] [[dbg_src]] %[[#]] %[[#]]
; CHECK-SPIRV: [[dbg_templ_1:%[0-9]+]] = OpExtInst [[type_void]] %[[#]] DebugTypeTemplate [[dbg_comp]] [[dbg_tparam_T]] [[dbg_tparam_N]]

; CHECK-OPTION-NOT: OpExtInstImport "NonSemantic.Shader.DebugInfo.100"

%struct.FixedArray = type { [10 x i32] }
@fa_global = dso_local global %struct.FixedArray zeroinitializer, align 4, !dbg !0

define dso_local noundef i32 @main() !dbg !23 {
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  ret i32 0, !dbg !26
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!15, !16, !17, !18, !19, !20, !21}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "fa_global", scope: !2, file: !3, line: 8, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version XX.X.XXXX (FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "template.cpp", directory: "/AAAAAAAAAA/BBBBBBBB/CCCCCCCCC", checksumkind: CSK_MD5, checksum: "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
!4 = !{!0}
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "FixedArray<int, 10>", file: !3, line: 4, size: 320, flags: DIFlagTypePassByValue, elements: !6, templateParams: !12, identifier: "_ZTS10FixedArrayIiLi10EE")
!6 = !{!7}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "data", scope: !5, file: !3, line: 5, baseType: !8, size: 320, flags: DIFlagPublic)
!8 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 320, elements: !10)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed, flags: DIFlagPublic)
!10 = !{!11}
!11 = !DISubrange(count: 10)
!12 = !{!13, !14}
!13 = !DITemplateTypeParameter(name: "T", type: !9)
!14 = !DITemplateValueParameter(name: "N", type: !9, value: i32 10)
!15 = !{i32 7, !"Dwarf Version", i32 5}
!16 = !{i32 2, !"Debug Info Version", i32 3}
!17 = !{i32 1, !"wchar_size", i32 4}
!18 = !{i32 8, !"PIC Level", i32 2}
!19 = !{i32 7, !"PIE Level", i32 2}
!20 = !{i32 7, !"uwtable", i32 2}
!21 = !{i32 7, !"frame-pointer", i32 2}
!23 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 10, type: !24, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2)
!24 = !DISubroutineType(types: !25, flags: DIFlagPublic)
!25 = !{!9}
!26 = !DILocation(line: 11, column: 3, scope: !23)
