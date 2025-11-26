; RUN: llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info --print-after=spirv-nonsemantic-debug-info -O0 -mtriple=spirv64-unknown-unknown -stop-after=spirv-nonsemantic-debug-info  %s -o - | FileCheck %s --check-prefix=CHECK-MIR
; RUN: llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --check-prefix=CHECK-OPTION
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-MIR: [[C_ONE:%[0-9]+]]:iid = OpConstantI [[TYPE_I32:%[0-9]+]], 1
; CHECK-MIR: [[TYPE_VOID:%[0-9]+]]:type(s64) = OpTypeVoid
; CHECK-MIR: [[C_FIVE:%[0-9]+]]:iid(s32) = OpConstantI [[TYPE_I32]], 5
; CHECK-MIR: [[C_THREE:%[0-9]+]]:iid(s32) = OpConstantI [[TYPE_I32]], 3
; CHECK-MIR: [[C_THIRTEEN:%[0-9]+]]:iid(s32) = OpConstantI [[TYPE_I32]], 13
; CHECK-MIR: [[C_THIRTYTWO:%[0-9]+]]:iid(s32) = OpConstantI [[TYPE_I32]], 32
; CHECK-MIR: [[C_FOUR:%[0-9]+]]:iid(s32) = OpConstantI [[TYPE_I32]], 4
; CHECK-MIR: [[TYPE_I32]]:type = OpTypeInt 32, 0
; CHECK-MIR: [[STR:%[0-9]+]]:id(s32) = OpString 1094795567, 1094795585
; CHECK-MIR: [[DBG_SOURCE:%[0-9]+]]:id(s32) = OpExtInst [[TYPE_VOID]](s64), 3, 35, [[STR]](s32)
; CHECK-MIR: [[DBG_CU:%[0-9]+]]:id(s32) = OpExtInst [[TYPE_VOID]](s64), 3, 1, [[C_THREE]](s32), [[C_FIVE]](s32), [[DBG_SOURCE]](s32), [[C_THIRTEEN]](s32)
; CHECK-MIR: [[STR_RED:%[0-9]+]]:id(s32) = OpString 6579538
; CHECK-MIR: [[STR_GREEN:%[0-9]+]]:id(s32) = OpString 1701147207, 110
; CHECK-MIR: [[STR_BLUE:%[0-9]+]]:id(s32) = OpString 1702194242, 0
; CHECK-MIR: [[DBG_ENUM:%[0-9]+]]:id(s32) = OpExtInst [[TYPE_VOID]](s64), 3, 9, {{%[0-9]+}}(s32), {{%[0-9]+}}(s32), [[DBG_SOURCE]](s32), [[C_ONE]], [[C_ONE]], {{%[0-9]+}}(s32), [[C_THIRTYTWO]](s32), [[C_THREE]](s32), [[C_ONE]], [[STR_RED]](s32), {{%[0-9]+}}(s32), [[STR_GREEN]](s32), [[C_FOUR]](s32), [[STR_BLUE]](s32)

; CHECK-SPIRV-DAG: %[[DBG_SRC:[0-9]+]] = OpExtInst %[[VOID:[0-9]+]] {{%[0-9]+}} DebugSource
; CHECK-SPIRV-DAG: %[[DBG_CU:[0-9]+]] = OpExtInst %[[VOID]] {{%[0-9]+}} DebugCompilationUnit {{%[0-9]+}} {{%[0-9]+}} %[[DBG_SRC]] {{%[0-9]+}}
; CHECK-SPIRV-DAG: %[[STR_INT:[0-9]+]] = OpString "int"
; CHECK-SPIRV-DAG: %[[DBG_TY_INT:[0-9]+]] = OpExtInst %[[VOID]] {{%[0-9]+}} DebugTypeBasic %[[STR_INT]] {{%[0-9]+}} {{%[0-9]+}} {{%[0-9]+}}
; CHECK-SPIRV-DAG: %[[STR_COLOR:[0-9]+]] = OpString "Color"
; CHECK-SPIRV-DAG: %[[STR_RED:[0-9]+]]   = OpString "Red"
; CHECK-SPIRV-DAG: %[[STR_GREEN:[0-9]+]] = OpString "Green"
; CHECK-SPIRV-DAG: %[[STR_BLUE:[0-9]+]]  = OpString "Blue"
; CHECK-SPIRV-DAG: %[[DBG_ENUM_COLOR:[0-9]+]] = OpExtInst %[[VOID]] {{%[0-9]+}} DebugTypeEnum %[[STR_COLOR]] %[[DBG_TY_INT]] %[[DBG_SRC]] {{%[0-9]+}} {{%[0-9]+}} %[[DBG_CU]] {{%[0-9]+}} {{%[0-9]+}} {{%[0-9]+}} %[[STR_RED]] {{%[0-9]+}} %[[STR_GREEN]] {{%[0-9]+}} %[[STR_BLUE]]

; CHECK-OPTION-NOT: DebugTypeEnum

@c = dso_local global i32 1, align 4, !dbg !0

define dso_local noundef i32 @main() !dbg !21 {
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  %0 = load i32, ptr @c, align 4, !dbg !23
  ret i32 %0, !dbg !24
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!13, !14, !15, !16, !17, !18, !19}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "c", scope: !2, file: !3, line: 7, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version XX.X.XXXX (FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !11, globals: !12, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "example.cpp", directory: "/AAAAAAAAAA/BBBBBBBB/CCCCCCCCC", checksumkind: CSK_MD5, checksum: "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
!4 = !{!5}
!5 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "Color", file: !3, line: 1, baseType: !6, size: 32, elements: !7, identifier: "_ZTS5Color", flags: DIFlagPublic)
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed, flags: DIFlagPublic)
!7 = !{!8, !9, !10}
!8 = !DIEnumerator(name: "Red", value: 1)
!9 = !DIEnumerator(name: "Green", value: 2)
!10 = !DIEnumerator(name: "Blue", value: -4)
!11 = !{!6}
!12 = !{!0}
!13 = !{i32 7, !"Dwarf Version", i32 5}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{i32 1, !"wchar_size", i32 4}
!16 = !{i32 8, !"PIC Level", i32 2}
!17 = !{i32 7, !"PIE Level", i32 2}
!18 = !{i32 7, !"uwtable", i32 2}
!19 = !{i32 7, !"frame-pointer", i32 2}
!21 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 9, type: !22, scopeLine: 9, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2)
!22 = !DISubroutineType(types: !11, flags: DIFlagPublic)
!23 = !DILocation(line: 10, column: 29, scope: !21)
!24 = !DILocation(line: 10, column: 5, scope: !21)
