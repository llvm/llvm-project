; RUN: llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info --print-after=spirv-nonsemantic-debug-info -O0 -mtriple=spirv64-unknown-unknown -stop-after=spirv-nonsemantic-debug-info  %s -o - | FileCheck %s --check-prefix=CHECK-MIR
; RUN: llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --check-prefix=CHECK-OPTION
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-MIR: [[TYPE_VOID:%[0-9]+]]:type(s64) = OpTypeVoid
; CHECK-MIR: [[STR:%[0-9]+]]:id(s32) = OpString 1094795567, 1094795585
; CHECK-MIR: [[DBG_SOURCE:%[0-9]+]]:id(s32) = OpExtInst [[TYPE_VOID]](s64), 3, 35, [[STR]](s32)
; CHECK-MIR: [[DBG_CU:%[0-9]+]]:id(s32) = OpExtInst [[TYPE_VOID]](s64), 3, 1
; CHECK-MIR: [[STR_INT:%[0-9]+]]:id(s32) = OpString 7630441
; CHECK-MIR: [[DBG_INT:%[0-9]+]]:id(s32) = OpExtInst [[TYPE_VOID]](s64), 3, 2, [[STR_INT]](s32)
; CHECK-MIR: [[DBG_ARRAY1:%[0-9]+]]:id(s32) = OpExtInst [[TYPE_VOID]](s64), 3, 5, [[DBG_INT]](s32)
; CHECK-MIR: [[DBG_ARRAY2:%[0-9]+]]:id(s32) = OpExtInst [[TYPE_VOID]](s64), 3, 5, [[DBG_INT]](s32)
; CHECK-MIR: [[DBG_GLOBAL:%[0-9]+]]:id(s32) = OpExtInst [[TYPE_VOID]](s64), 3, 18, {{%[0-9]+}}(s32), [[DBG_ARRAY1]](s32), [[DBG_SOURCE]](s32)

; CHECK-SPIRV: [[int_str:%[0-9]+]] = OpString "int"
; CHECK-SPIRV-DAG: [[type_void:%[0-9]+]] = OpTypeVoid
; CHECK-SPIRV: [[dbg_src:%[0-9]+]] = OpExtInst [[type_void]] %[[#]] DebugSource %[[#]]
; CHECK-SPIRV: [[dbg_cu:%[0-9]+]] = OpExtInst [[type_void]] %[[#]] DebugCompilationUnit %[[#]] %[[#]] [[dbg_src]] %[[#]]
; CHECK-SPIRV: [[dbg_int:%[0-9]+]] = OpExtInst [[type_void]] %[[#]] DebugTypeBasic [[int_str]] %[[#]] %[[#]] [[flags:%[0-9]+]]
; CHECK-SPIRV: [[dbg_tarr_1:%[0-9]+]] = OpExtInst [[type_void]] %[[#]] DebugTypeArray [[dbg_int]] %[[#]] %[[#]]
; CHECK-SPIRV: [[dbg_tarr_2:%[0-9]+]] = OpExtInst [[type_void]] %[[#]] DebugTypeArray [[dbg_int]] %[[#]]
; CHECK-SPIRV: [[dbg_none:%[0-9]+]] = OpExtInst [[type_void]] %[[#]] DebugInfoNone
; CHECK-SPIRV: [[dbg_global:%[0-9]+]] = OpExtInst [[type_void]] %[[#]] DebugGlobalVariable %[[#]] [[dbg_tarr_1]] [[dbg_src]] %[[#]] %[[#]] [[dbg_cu]] %[[#]] [[dbg_none]] %[[#]]

; CHECK-OPTION-NOT: OpExtInstImport "NonSemantic.Shader.DebugInfo.100"

@__const.main.local_array = private unnamed_addr constant [2 x i32] [i32 1, i32 2], align 4
@global_array = dso_local global [4 x [3 x i32]] zeroinitializer, align 16, !dbg !0
define dso_local i32 @main() !dbg !18 {

entry:
  %retval = alloca i32, align 4
  %local_array = alloca [2 x i32], align 4
  store i32 0, ptr %retval, align 4
    #dbg_declare(ptr %local_array, !22, !DIExpression(), !26)
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %local_array, ptr align 4 @__const.main.local_array, i64 8, i1 false), !dbg !26
  %arrayidx = getelementptr inbounds [2 x i32], ptr %local_array, i64 0, i64 0, !dbg !27
  %0 = load i32, ptr %arrayidx, align 4, !dbg !27
  ret i32 %0, !dbg !28
}

declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10, !11, !12, !13, !14, !15, !16}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "global_array", scope: !2, file: !3, line: 2, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version XX.X.XXXX (FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "typearray.c", directory: "/AAAAAAAAAA/BBBBBBBB/CCCCCCCCC", checksumkind: CSK_MD5, checksum: "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
!4 = !{!0}
!5 = !DICompositeType(tag: DW_TAG_array_type, baseType: !6, size: 384, elements: !7)
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed, flags: DIFlagPublic)
!7 = !{!8, !9}
!8 = !DISubrange(count: 4)
!9 = !DISubrange(count: 3)
!10 = !{i32 7, !"Dwarf Version", i32 5}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{i32 8, !"PIC Level", i32 2}
!14 = !{i32 7, !"PIE Level", i32 2}
!15 = !{i32 7, !"uwtable", i32 2}
!16 = !{i32 7, !"frame-pointer", i32 2}
!18 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 4, type: !19, scopeLine: 4, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !21, flags: DIFlagPublic)
!19 = !DISubroutineType(types: !20, flags: DIFlagPublic)
!20 = !{!6}
!21 = !{}
!22 = !DILocalVariable(name: "local_array", scope: !18, file: !3, line: 5, type: !23, flags: DIFlagPublic)
!23 = !DICompositeType(tag: DW_TAG_array_type, baseType: !6, size: 64, elements: !24)
!24 = !{!25}
!25 = !DISubrange(count: 2)
!26 = !DILocation(line: 5, column: 9, scope: !18)
!27 = !DILocation(line: 6, column: 12, scope: !18)
!28 = !DILocation(line: 6, column: 5, scope: !18)
