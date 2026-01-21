; RUN: llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info --print-after=spirv-nonsemantic-debug-info -O0 -mtriple=spirv64-unknown-unknown -stop-after=spirv-nonsemantic-debug-info  %s -o - | FileCheck %s --check-prefix=CHECK-MIR
; RUN: llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --check-prefix=CHECK-OPTION 
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-MIR: [[TYPE_VOID:%[0-9]+]]:type(s64) = OpTypeVoid
; CHECK-MIR: [[TYPE_I32:%[0-9]+]]:type = OpTypeInt 32, 0
; CHECK-MIR: [[str:%[0-9]+]]:id(s32) = OpString 1094795567, 1094795585
; CHECK-MIR: [[DEBUG_SOURCE:%[0-9]+]]:id(s32) = OpExtInst [[TYPE_VOID]](s64), 3, 35, [[str]](s32)
; CHECK-MIR: [[DEBUG_COMPILATION_UNIT:%[0-9]+]]:id(s32) = OpExtInst [[TYPE_VOID]](s64), 3, 1
; CHECK-MIR: [[str_76:%[0-9]+]]:id(s32) = OpString 7630441
; CHECK-MIR: [[DEBUG_TYPE_BASIC:%[0-9]+]]:id(s32) = OpExtInst [[TYPE_VOID]](s64), 3, 2, [[str_76]](s32)
; CHECK-MIR: [[STR_POINT:%[0-9]+]]:id(s32) = OpString 1852403536, 116
; CHECK-MIR: [[STR_MEMBER1:%[0-9]+]]:id(s32) = OpString 120
; CHECK-MIR: [[DEBUG_MEMBER1:%[0-9]+]]:id(s32) = OpExtInst [[TYPE_VOID]](s64), 3, 11, [[STR_MEMBER1]](s32), [[DEBUG_TYPE_BASIC]](s32), [[DEBUG_SOURCE]](s32)
; CHECK-MIR: [[STR_MEMBER2:%[0-9]+]]:id(s32) = OpString 121
; CHECK-MIR: [[DEBUG_MEMBER2:%[0-9]+]]:id(s32) = OpExtInst [[TYPE_VOID]](s64), 3, 11, [[STR_MEMBER2]](s32), [[DEBUG_TYPE_BASIC]](s32), [[DEBUG_SOURCE]](s32)
; CHECK-MIR: [[DEBUG_COMPOSITE_POINT:%[0-9]+]]:id(s32) = OpExtInst [[TYPE_VOID]](s64), 3, 10, [[STR_POINT]](s32), {{%[0-9]+}}, [[DEBUG_SOURCE]](s32), {{%[0-9]+}}(s32), {{%[0-9]+}}, {{%[0-9]+}}(s32), {{%[0-9]+}}(s32), {{%[0-9]+}}(s32), {{%[0-9]+}}(s32), [[DEBUG_MEMBER1]](s32), [[DEBUG_MEMBER2]](s32)

; CHECK-SPIRV: [[point_str:%[0-9]+]] = OpString "Point"
; CHECK-SPIRV: [[x_str:%[0-9]+]] = OpString "x"
; CHECK-SPIRV: [[y_str:%[0-9]+]] = OpString "y"
; CHECK-SPIRV: [[type_void:%[0-9]+]] = OpTypeVoid
; CHECK-SPIRV: [[dbg_src:%[0-9]+]] = OpExtInst [[type_void]] %[[#]] DebugSource %[[#]]
; CHECK-SPIRV: [[dbg_cu:%[0-9]+]] = OpExtInst [[type_void]] %[[#]] DebugCompilationUnit %[[#]] %[[#]] [[dbg_src]] %[[#]]
; CHECK-SPIRV: [[dbg_int:%[0-9]+]] = OpExtInst [[type_void]] %[[#]] DebugTypeBasic
; CHECK-SPIRV: [[dbg_x:%[0-9]+]] = OpExtInst [[type_void]] %[[#]] DebugTypeMember [[x_str]] [[dbg_int]] [[dbg_src]]
; CHECK-SPIRV: [[dbg_y:%[0-9]+]] = OpExtInst [[type_void]] %[[#]] DebugTypeMember [[y_str]] [[dbg_int]] [[dbg_src]]
; CHECK-SPIRV: [[dbg_point:%[0-9]+]] = OpExtInst [[type_void]] %[[#]] DebugTypeComposite [[point_str]] %[[#]] [[dbg_src]] %[[#]] %[[#]] [[dbg_cu]] %[[#]] %[[#]] %[[#]] [[dbg_x]] [[dbg_y]]

; CHECK-OPTION-NOT: OpExtInstImport "NonSemantic.Shader.DebugInfo.100"

%struct.Point = type { i32, i32 }
@p = dso_local global %struct.Point zeroinitializer, align 4, !dbg !0

define dso_local noundef i32 @main()!dbg !18 {
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  store i32 10, ptr @p, align 4, !dbg !21
  store i32 20, ptr getelementptr inbounds nuw (%struct.Point, ptr @p, i32 0, i32 1), align 4, !dbg !22
  ret i32 0, !dbg !23
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10, !11, !12, !13, !14, !15, !16}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "p", scope: !2, file: !3, line: 7, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version XX.X.XXXX (FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "composite.cpp", directory: "/AAAAAAAAAA/BBBBBBBB/CCCCCCCCC", checksumkind: CSK_MD5, checksum: "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
!4 = !{!0}
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Point", file: !3, line: 2, size: 64, flags: DIFlagTypePassByValue, elements: !6, identifier: "_ZTS5Point")
!6 = !{!7, !9}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !5, file: !3, line: 3, baseType: !8, size: 32, flags: DIFlagPublic)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed, flags: DIFlagPublic)
!9 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !5, file: !3, line: 4, baseType: !8, size: 32, offset: 32, flags: DIFlagPublic)
!10 = !{i32 7, !"Dwarf Version", i32 5}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{i32 8, !"PIC Level", i32 2}
!14 = !{i32 7, !"PIE Level", i32 2}
!15 = !{i32 7, !"uwtable", i32 2}
!16 = !{i32 7, !"frame-pointer", i32 2}
!18 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 8, type: !19, scopeLine: 8, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2)
!19 = !DISubroutineType(types: !20, flags: DIFlagPublic)
!20 = !{!8}
!21 = !DILocation(line: 9, column: 7, scope: !18)
!22 = !DILocation(line: 10, column: 7, scope: !18)
!23 = !DILocation(line: 11, column: 3, scope: !18)
