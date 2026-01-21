; RUN: llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info --print-after=spirv-nonsemantic-debug-info -O0 -mtriple=spirv64-unknown-unknown -stop-after=spirv-nonsemantic-debug-info  %s -o - | FileCheck %s --check-prefix=CHECK-MIR
; RUN: llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --check-prefix=CHECK-OPTION
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-MIR: [[TYPE_I32:%[0-9]+]]:type = OpTypeInt 32, 0
; CHECK-MIR: [[TYPE_VOID:%[0-9]+]]:type(s64) = OpTypeVoid
; CHECK-MIR: [[DBG_SOURCE:%[0-9]+]]:id(s32) = OpExtInst [[TYPE_VOID]](s64), 3, 35
; CHECK-MIR: [[DBG_CU:%[0-9]+]]:id(s32) = OpExtInst [[TYPE_VOID]](s64), 3, 1
; CHECK-MIR: [[STR_INT:%[0-9]+]]:id(s32) = OpString 7630441
; CHECK-MIR: [[DBG_INT:%[0-9]+]]:id(s32) = OpExtInst [[TYPE_VOID]](s64), 3, 2, [[STR_INT]](s32)
; CHECK-MIR: [[STR_TYPEDEF:%[0-9]+]]:id(s32) = OpString 1852406125, 116
; CHECK-MIR: [[DBG_TYPEDEF:%[0-9]+]]:id(s32) = OpExtInst [[TYPE_VOID]](s64), 3, 7, [[STR_TYPEDEF]](s32), [[DBG_INT]](s32), [[DBG_SOURCE]](s32)

; CHECK-SPIRV-DAG: [[int_str:%[0-9]+]] = OpString "int"
; CHECK-SPIRV-DAG: [[typedef_str:%[0-9]+]] = OpString "myint"
; CHECK-SPIRV: [[dbg_src:%[0-9]+]] = OpExtInst [[void_ty:%[0-9]+]] %[[#]] DebugSource
; CHECK-SPIRV: OpExtInst [[void_ty]] %[[#]] DebugCompilationUnit %[[#]] %[[#]] [[dbg_src]] %[[#]]
; CHECK-SPIRV: [[dbg_int:%[0-9]+]] = OpExtInst [[void_ty]] %[[#]] DebugTypeBasic [[int_str]] %[[#]] %[[#]] %[[#]]
; CHECK-SPIRV: OpExtInst [[void_ty]] %[[#]] DebugTypedef [[typedef_str]] [[dbg_int]] [[dbg_src]] %[[#]] %[[#]] %[[#]]

; CHECK-OPTION-NOT: OpExtInstImport "NonSemantic.Shader.DebugInfo.100"

define dso_local i32 @square(i32 noundef %x)!dbg !10 {
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4
    #dbg_declare(ptr %x.addr, !16, !DIExpression(), !17)
  %0 = load i32, ptr %x.addr, align 4, !dbg !18
  %1 = load i32, ptr %x.addr, align 4, !dbg !19
  %mul = mul nsw i32 %0, %1, !dbg !20
  ret i32 %mul, !dbg !21
}

define dso_local i32 @main() !dbg !22 {
entry:
  %retval = alloca i32, align 4
  %val = alloca i32, align 4
  %result = alloca i32, align 4
  store i32 0, ptr %retval, align 4
    #dbg_declare(ptr %val, !25, !DIExpression(), !26)
  store i32 7, ptr %val, align 4, !dbg !26
    #dbg_declare(ptr %result, !27, !DIExpression(), !28)
  %0 = load i32, ptr %val, align 4, !dbg !29
  %call = call i32 @square(i32 noundef %0), !dbg !30
  store i32 %call, ptr %result, align 4, !dbg !28
  %1 = load i32, ptr %result, align 4, !dbg !31
  ret i32 %1, !dbg !32
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version XX.X.XXXX (FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "typedef.c", directory: "/AAAAAAAAAA/BBBBBBBB/CCCCCCCCC", checksumkind: CSK_MD5, checksum: "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!10 = distinct !DISubprogram(name: "square", scope: !1, file: !1, line: 5, type: !11, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !15)
!11 = !DISubroutineType(types: !12, flags: DIFlagPublic)
!12 = !{!13, !13}
!13 = !DIDerivedType(tag: DW_TAG_typedef, name: "myint", file: !1, line: 3, baseType: !14)
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed, flags: DIFlagPublic)
!15 = !{}
!16 = !DILocalVariable(name: "x", arg: 1, scope: !10, file: !1, line: 5, type: !13, flags: DIFlagPublic)
!17 = !DILocation(line: 5, column: 20, scope: !10)
!18 = !DILocation(line: 6, column: 12, scope: !10)
!19 = !DILocation(line: 6, column: 16, scope: !10)
!20 = !DILocation(line: 6, column: 14, scope: !10)
!21 = !DILocation(line: 6, column: 5, scope: !10)
!22 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 9, type: !23, scopeLine: 9, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !15)
!23 = !DISubroutineType(types: !24, flags: DIFlagPublic)
!24 = !{!14}
!25 = !DILocalVariable(name: "val", scope: !22, file: !1, line: 10, type: !13, flags: DIFlagPublic)
!26 = !DILocation(line: 10, column: 11, scope: !22)
!27 = !DILocalVariable(name: "result", scope: !22, file: !1, line: 11, type: !13, flags: DIFlagPublic)
!28 = !DILocation(line: 11, column: 11, scope: !22)
!29 = !DILocation(line: 11, column: 27, scope: !22)
!30 = !DILocation(line: 11, column: 20, scope: !22)
!31 = !DILocation(line: 12, column: 12, scope: !22)
!32 = !DILocation(line: 12, column: 5, scope: !22)
