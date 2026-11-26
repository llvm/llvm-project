; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --check-prefix=QUALCOUNT
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --check-prefix=UNSUPPORTED
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - -filetype=obj | spirv-val %}

; A constant assignment to a variable of each qualified type the IR verifier
; accepts on a DIDerivedType.
;
; DW_TAG_immutable_type and DW_TAG_LLVM_ptrauth_type have no Type Qualifier
; value in NonSemantic.Shader.DebugInfo.100.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[PLAINNAME:%[0-9]+]] = OpString "plain"
; CHECK-DAG: [[CONSTNAME:%[0-9]+]] = OpString "const_qualified"
; CHECK-DAG: [[VOLATILENAME:%[0-9]+]] = OpString "volatile_qualified"
; CHECK-DAG: [[RESTRICTNAME:%[0-9]+]] = OpString "restrict_qualified"
; CHECK-DAG: [[ATOMICNAME:%[0-9]+]] = OpString "atomic_qualified"
; CHECK-DAG: [[C0:%[0-9]+]] = OpConstant [[I32]] 0{{ *$}}
; CHECK-DAG: [[C1:%[0-9]+]] = OpConstant [[I32]] 1{{ *$}}
; CHECK-DAG: [[C2:%[0-9]+]] = OpConstant [[I32]] 2{{ *$}}
; CHECK-DAG: [[C3:%[0-9]+]] = OpConstant [[I32]] 3{{ *$}}
; CHECK-DAG: [[C42:%[0-9]+]] = OpConstant [[I32]] 42{{ *$}}
; CHECK-DAG: [[C43:%[0-9]+]] = OpConstant [[I32]] 43{{ *$}}
; CHECK-DAG: [[C44:%[0-9]+]] = OpConstant [[I32]] 44{{ *$}}
; CHECK-DAG: [[C45:%[0-9]+]] = OpConstant [[I32]] 45{{ *$}}
; CHECK-DAG: [[C46:%[0-9]+]] = OpConstant [[I32]] 46{{ *$}}
; CHECK-DAG: [[INT:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugTypeBasic
; CHECK-DAG: [[CONSTTY:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugTypeQualifier [[INT]] [[C0]]{{ *$}}
; CHECK-DAG: [[VOLATILETY:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugTypeQualifier [[INT]] [[C1]]{{ *$}}
; CHECK-DAG: [[RESTRICTTY:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugTypeQualifier [[INT]] [[C2]]{{ *$}}
; CHECK-DAG: [[ATOMICTY:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugTypeQualifier [[INT]] [[C3]]{{ *$}}
; CHECK-DAG: [[PLAINVAR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[PLAINNAME]] [[INT]]
; CHECK-DAG: [[CONSTVAR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[CONSTNAME]] [[CONSTTY]]
; CHECK-DAG: [[VOLATILEVAR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[VOLATILENAME]] [[VOLATILETY]]
; CHECK-DAG: [[RESTRICTVAR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[RESTRICTNAME]] [[RESTRICTTY]]
; CHECK-DAG: [[ATOMICVAR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[ATOMICNAME]] [[ATOMICTY]]

; CHECK: OpFunction
; CHECK: OpExtInst [[VOID]] [[EXT]] DebugValue [[PLAINVAR]] [[C42]]
; CHECK: OpExtInst [[VOID]] [[EXT]] DebugValue [[CONSTVAR]] [[C43]]
; CHECK: OpExtInst [[VOID]] [[EXT]] DebugValue [[VOLATILEVAR]] [[C44]]
; CHECK: OpExtInst [[VOID]] [[EXT]] DebugValue [[RESTRICTVAR]] [[C45]]
; CHECK: OpExtInst [[VOID]] [[EXT]] DebugValue [[ATOMICVAR]] [[C46]]
; CHECK-NOT: DebugValue
; CHECK: OpFunctionEnd

; One DebugTypeQualifier per lowered qualifier, and none for the others.
; QUALCOUNT-COUNT-4: DebugTypeQualifier
; QUALCOUNT-NOT: DebugTypeQualifier

; UNSUPPORTED-DAG: [[IMMUTABLENAME:%[0-9]+]] = OpString "immutable_qualified"
; UNSUPPORTED-DAG: [[PTRAUTHNAME:%[0-9]+]] = OpString "ptrauth_qualified"
; UNSUPPORTED-NOT: DebugLocalVariable [[IMMUTABLENAME]]
; UNSUPPORTED-NOT: DebugLocalVariable [[PTRAUTHNAME]]

target triple = "spirv64-unknown-unknown"

define spir_func void @f() !dbg !5 {
entry:
    #dbg_value(i32 42, !8, !DIExpression(), !10)
    #dbg_value(i32 43, !9, !DIExpression(), !10)
    #dbg_value(i32 44, !12, !DIExpression(), !10)
    #dbg_value(i32 45, !14, !DIExpression(), !10)
    #dbg_value(i32 46, !16, !DIExpression(), !10)
    #dbg_value(i32 47, !18, !DIExpression(), !10)
    #dbg_value(i32 48, !20, !DIExpression(), !10)
  ret void, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-value-qualified-constant.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = distinct !DISubprogram(name: "f", linkageName: "f", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!8 = !DILocalVariable(name: "plain", scope: !5, file: !1, line: 2, type: !7)
!9 = !DILocalVariable(name: "const_qualified", scope: !5, file: !1, line: 3, type: !11)
!10 = !DILocation(line: 9, column: 3, scope: !5)
!11 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !7)
!12 = !DILocalVariable(name: "volatile_qualified", scope: !5, file: !1, line: 4, type: !13)
!13 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !7)
!14 = !DILocalVariable(name: "restrict_qualified", scope: !5, file: !1, line: 5, type: !15)
!15 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !7)
!16 = !DILocalVariable(name: "atomic_qualified", scope: !5, file: !1, line: 6, type: !17)
!17 = !DIDerivedType(tag: DW_TAG_atomic_type, baseType: !7)
!18 = !DILocalVariable(name: "immutable_qualified", scope: !5, file: !1, line: 7, type: !19)
!19 = !DIDerivedType(tag: DW_TAG_immutable_type, baseType: !7)
!20 = !DILocalVariable(name: "ptrauth_qualified", scope: !5, file: !1, line: 8, type: !21)
!21 = !DIDerivedType(tag: DW_TAG_LLVM_ptrauth_type, baseType: !7, ptrAuthKey: 1)
