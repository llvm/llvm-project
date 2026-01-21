; RUN: llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info --print-after=spirv-nonsemantic-debug-info -O0 -mtriple=spirv64-unknown-unknown -stop-after=spirv-nonsemantic-debug-info  %s -o - | FileCheck %s --check-prefix=CHECK-MIR
; RUN: llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --check-prefix=CHECK-OPTION
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-MIR: [[TYPE_VOID:%[0-9]+]]:type(s64) = OpTypeVoid
; CHECK-MIR: [[DBG_SOURCE:%[0-9]+]]:id(s32) = OpExtInst [[TYPE_VOID]](s64), 3, 35
; CHECK-MIR: [[DBG_CU:%[0-9]+]]:id(s32) = OpExtInst [[TYPE_VOID]](s64), 3, 1
; CHECK-MIR: [[STR_INT:%[0-9]+]]:id(s32) = OpString 7630441
; CHECK-MIR: [[DBG_INT:%[0-9]+]]:id(s32) = OpExtInst [[TYPE_VOID]](s64), 3, 2, [[STR_INT]](s32)
; CHECK-MIR: [[STR_S:%[0-9]+]]:id(s32) = OpString 97
; CHECK-MIR: [[DBG_VAR_S:%[0-9]+]]:id(s32) = OpExtInst [[TYPE_VOID]](s64), 3, 11, [[STR_S]](s32), [[DBG_INT]](s32), [[DBG_SOURCE]](s32)
; CHECK-MIR: [[DBG_FUNC:%[0-9]+]]:id(s32) = OpExtInst [[TYPE_VOID]](s64), 3, 10, {{%[0-9]+}}(s32), {{%[0-9]+}}(s32), [[DBG_SOURCE]](s32)
; CHECK-MIR: [[dbg_global1:%[0-9]+]]:id(s32) = OpExtInst [[TYPE_VOID]](s64), 3, 13, [[DBG_INT]](s32), [[DBG_FUNC]](s32)

; CHECK-SPIRV-DAG: [[int_str:%[0-9]+]] = OpString "int"
; CHECK-SPIRV-DAG: [[struct_str:%[0-9]+]] = OpString "S"
; CHECK-SPIRV-DAG: [[member_a_str:%[0-9]+]] = OpString "a"
; CHECK-SPIRV-DAG: [[ptr_str:%[0-9]+]] = OpString "ptr"
; CHECK-SPIRV: [[dbg_src:%[0-9]+]] = OpExtInst [[void_ty:%[0-9]+]] %[[#]] DebugSource %[[#]]
; CHECK-SPIRV: [[dbg_cu:%[0-9]+]] = OpExtInst [[void_ty]] %[[#]] DebugCompilationUnit %[[#]] %[[#]] [[dbg_src]] %[[#]]
; CHECK-SPIRV: [[dbg_int:%[0-9]+]] = OpExtInst [[void_ty]] %[[#]] DebugTypeBasic [[int_str]] %[[#]] %[[#]] %[[#]]
; CHECK-SPIRV: [[dbg_member:%[0-9]+]] = OpExtInst [[void_ty]] %[[#]] DebugTypeMember [[member_a_str]] [[dbg_int]] [[dbg_src]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#]]
; CHECK-SPIRV: [[dbg_struct:%[0-9]+]] = OpExtInst [[void_ty]] %[[#]] DebugTypeComposite [[struct_str]] %[[#]] [[dbg_src]] %[[#]] %[[#]] [[dbg_cu]] %[[#]] %[[#]] %[[#]] [[dbg_member]]
; CHECK-SPIRV: [[dbg_ptr:%[0-9]+]] = OpExtInst [[void_ty]] %[[#]] DebugTypePtrToMember [[dbg_int]] [[dbg_struct]]

; CHECK-OPTION-NOT: OpExtInstImport "NonSemantic.Shader.DebugInfo.100"

%struct.S = type { i32 }
@ptr = dso_local global i64 0, align 8, !dbg !0
@__const._Z3usev.s = private unnamed_addr constant %struct.S { i32 42 }, align 4

define dso_local noundef i32 @_Z3usev() !dbg !18 {
entry:
  %s = alloca %struct.S, align 4
    #dbg_declare(ptr %s, !22, !DIExpression(), !23)
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %s, ptr align 4 @__const._Z3usev.s, i64 4, i1 false), !dbg !23
  %0 = load i64, ptr @ptr, align 8, !dbg !24
  %memptr.offset = getelementptr inbounds i8, ptr %s, i64 %0, !dbg !25
  %1 = load i32, ptr %memptr.offset, align 4, !dbg !25
  ret i32 %1, !dbg !26
}

declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10, !11, !12, !13, !14, !15, !16}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "ptr", scope: !2, file: !3, line: 7, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version XX.X.XXXX (FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "ptrtomember.cpp", directory: "/AAAAAAAAAA/BBBBBBBB/CCCCCCCCC", checksumkind: CSK_MD5, checksum: "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
!4 = !{!0}
!5 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !6, size: 64, extraData: !7)
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !3, line: 3, size: 32, flags: DIFlagTypePassByValue, elements: !8, identifier: "_ZTS1S")
!8 = !{!9}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !7, file: !3, line: 4, baseType: !6, size: 32)
!10 = !{i32 7, !"Dwarf Version", i32 5}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{i32 8, !"PIC Level", i32 2}
!14 = !{i32 7, !"PIE Level", i32 2}
!15 = !{i32 7, !"uwtable", i32 2}
!16 = !{i32 7, !"frame-pointer", i32 2}
!18 = distinct !DISubprogram(name: "use", linkageName: "_Z3usev", scope: !3, file: !3, line: 9, type: !19, scopeLine: 9, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !21)
!19 = !DISubroutineType(types: !20, flags: DIFlagPublic)
!20 = !{!6}
!21 = !{}
!22 = !DILocalVariable(name: "s", scope: !18, file: !3, line: 10, type: !7, flags: DIFlagPublic)
!23 = !DILocation(line: 10, column: 5, scope: !18)
!24 = !DILocation(line: 11, column: 13, scope: !18)
!25 = !DILocation(line: 11, column: 11, scope: !18)
!26 = !DILocation(line: 11, column: 3, scope: !18)
