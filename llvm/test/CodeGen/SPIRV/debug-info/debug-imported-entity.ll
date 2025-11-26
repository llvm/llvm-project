; RUN: llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info --print-after=spirv-nonsemantic-debug-info -O0 -mtriple=spirv64-unknown-unknown -stop-after=spirv-nonsemantic-debug-info  %s -o - | FileCheck %s --check-prefix=CHECK-MIR
; RUN: llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --check-prefix=CHECK-OPTION
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-MIR: [[i32_5:%[0-9]+]]:iid = OpConstantI [[i32type:%[0-9]+]], 5
; CHECK-MIR: [[void_type:%[0-9]+]]:type(s64) = OpTypeVoid
; CHECK-MIR: [[i32_3:%[0-9]+]]:iid(s32) = OpConstantI [[i32type]], 3
; CHECK-MIR: [[i32_13:%[0-9]+]]:iid(s32) = OpConstantI [[i32type]], 13
; CHECK-MIR: [[i32_1:%[0-9]+]]:iid(s32) = OpConstantI [[i32type]], 1
; CHECK-MIR: [[i32type:%[0-9]+]]:type = OpTypeInt 32, 0
; CHECK-MIR: [[debug_source:%[0-9]+]]:id(s32) = OpExtInst [[void_type]](s64), 3, 35
; CHECK-MIR: [[debug_comp_unit:%[0-9]+]]:id(s32) = OpExtInst [[void_type]](s64), 3, 1, [[i32_3]](s32), [[i32_5]], [[debug_source]](s32), [[i32_13]](s32)
; CHECK-MIR: [[debug_info_none:%[0-9]+]]:id(s32) = OpExtInst [[void_type]](s64), 3, 0
; CHECK-MIR: [[debug_imported_entity:%[0-9]+]]:id(s32) = OpExtInst [[void_type]](s64), 3, 34, {{%[0-9]+}}(s32), {{%[0-9]+}}, [[debug_source]](s32), [[debug_info_none]](s32), [[i32_3]](s32), [[i32_1]](s32)

; CHECK-SPIRV-DAG: [[i32type:%[0-9]+]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: [[void_type:%[0-9]+]] = OpTypeVoid
; CHECK-SPIRV-DAG: [[debug_source:%[0-9]+]] = OpExtInst [[void_type]] {{%[0-9]+}} DebugSource
; CHECK-SPIRV-DAG: [[debug_comp_unit:%[0-9]+]] = OpExtInst [[void_type]] {{%[0-9]+}} DebugCompilationUnit {{%[0-9]+}} {{%[0-9]+}} [[debug_source]] {{%[0-9]+}}
; CHECK-SPIRV-DAG: [[debug_info_none:%[0-9]+]] = OpExtInst [[void_type]] {{%[0-9]+}} DebugInfoNone
; CHECK-SPIRV-DAG: [[debug_imported_entity:%[0-9]+]] = OpExtInst [[void_type]] {{%[0-9]+}} DebugImportedEntity {{%[0-9]+}} {{%[0-9]+}} [[debug_source]] [[debug_info_none]] {{%[0-9]+}} {{%[0-9]+}} [[debug_comp_unit]]

; CHECK-OPTION-NOT: DebugImportedEntity

define dso_local noundef i32 @main() !dbg !13 {
entry:
  %retval = alloca i32, align 4
  %r = alloca i32, align 4
  store i32 0, ptr %retval, align 4
    #dbg_declare(ptr %r, !18, !DIExpression(), !19)
  %call = call noundef i32 @_ZN4myns9double_itEi(i32 noundef 5), !dbg !20
  store i32 %call, ptr %r, align 4, !dbg !19
  %0 = load i32, ptr %r, align 4, !dbg !21
  ret i32 %0, !dbg !22
}

declare noundef i32 @_ZN4myns9double_itEi(i32 noundef)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5, !6, !7, !8, !9, !10, !11}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version XX.X", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, imports: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "main.cpp", directory: "/tmp", checksumkind: CSK_MD5, checksum: "2ca3a2708747979970bbb428cfe8db65")
!2 = !{!3}
!3 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !0, entity: !4, file: !1, line: 3)
!4 = !DINamespace(name: "myns", scope: null)
!5 = !{i32 7, !"Dwarf Version", i32 5}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{i32 8, !"PIC Level", i32 2}
!9 = !{i32 7, !"PIE Level", i32 2}
!10 = !{i32 7, !"uwtable", i32 2}
!11 = !{i32 7, !"frame-pointer", i32 2}
!13 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 5, type: !14, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !17)
!14 = !DISubroutineType(types: !15, flags: DIFlagPublic)
!15 = !{!16}
!16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed, flags: DIFlagPublic)
!17 = !{}
!18 = !DILocalVariable(name: "r", scope: !13, file: !1, line: 6, type: !16, flags: DIFlagPublic)
!19 = !DILocation(line: 6, column: 9, scope: !13)
!20 = !DILocation(line: 6, column: 13, scope: !13)
!21 = !DILocation(line: 7, column: 12, scope: !13)
!22 = !DILocation(line: 7, column: 5, scope: !13)
