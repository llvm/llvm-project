; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --asm-verbose=0 --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --asm-verbose=0 --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --check-prefix=UNIQUE
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; A narrow or wide constant can be named when the semantic module already
; defines its scalar type. The function signature below requires all four
; optional-capability types, so the debug handler must reuse each type and
; emit only the constants needed by the DebugValue records.
;
; The signature uses no 32-bit integer, so this is the one module in the suite
; where the handler creates OpTypeInt 32 0 rather than finding it. That type
; has a second owner, the line and column constants every DebugLine needs, and
; two declarations of it are a duplicate the validator rejects.

; CHECK-DAG: OpCapability Int16
; CHECK-DAG: OpCapability Int64
; CHECK-DAG: OpCapability Float16
; CHECK-DAG: OpCapability Float64
; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I16:%[0-9]+]] = OpTypeInt 16 0
; CHECK-DAG: [[I64:%[0-9]+]] = OpTypeInt 64 0
; CHECK-DAG: [[F16:%[0-9]+]] = OpTypeFloat 16
; CHECK-DAG: [[F64:%[0-9]+]] = OpTypeFloat 64
; CHECK-DAG: [[I16C:%[0-9]+]] = OpConstant [[I16]] 7{{ *$}}
; CHECK-DAG: [[I64C:%[0-9]+]] = OpConstant [[I64]] 1234605616436508552{{ *$}}
; A half prints as its raw bit pattern and a double prints as a value, so
; pinning both covers the width the printer is told and the two-word
; reconstruction that 1.0 needs.
; CHECK-DAG: [[F16C:%[0-9]+]] = OpConstant [[F16]] 15360{{ *$}}
; CHECK-DAG: [[F64C:%[0-9]+]] = OpConstant [[F64]] 1{{ *$}}
; CHECK-DAG: [[I32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[I32C:%[0-9]+]] = OpConstant [[I32]] 987654321{{ *$}}
; CHECK-DAG: [[I16NAME:%[0-9]+]] = OpString "i16_constant"
; CHECK-DAG: [[I64NAME:%[0-9]+]] = OpString "i64_constant"
; CHECK-DAG: [[F16NAME:%[0-9]+]] = OpString "half_constant"
; CHECK-DAG: [[F64NAME:%[0-9]+]] = OpString "double_constant"
; CHECK-DAG: [[I32NAME:%[0-9]+]] = OpString "i32_constant"
; CHECK-DAG: [[I16VAR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[I16NAME]]
; CHECK-DAG: [[I64VAR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[I64NAME]]
; CHECK-DAG: [[F16VAR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[F16NAME]]
; CHECK-DAG: [[F64VAR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[F64NAME]]
; CHECK-DAG: [[I32VAR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[I32NAME]]

; CHECK: OpFunction
; CHECK: OpExtInst [[VOID]] [[EXT]] DebugValue [[I16VAR]] [[I16C]]
; CHECK: OpExtInst [[VOID]] [[EXT]] DebugValue [[I64VAR]] [[I64C]]
; CHECK: OpExtInst [[VOID]] [[EXT]] DebugValue [[F16VAR]] [[F16C]]
; CHECK: OpExtInst [[VOID]] [[EXT]] DebugValue [[F64VAR]] [[F64C]]
; CHECK: OpExtInst [[VOID]] [[EXT]] DebugValue [[I32VAR]] [[I32C]]
; Returning the i64 literal makes semantic code create it too. DebugValue and
; OpReturnValue naming one id exercises module-constant reuse and the two-word
; comparison of a 64-bit value.
; CHECK: OpReturnValue [[I64C]]

; The handler must reuse each existing type rather than declare a second copy.
; This needs its own FileCheck pass: a bare CHECK-NOT would close the CHECK-DAG
; group above it, and the OpStrings that group matches come earlier in the
; output than the types.
;
; One DAG group then five negatives, so every negative region starts after the
; whole semantic type block and reaches end of file. Pairing each positive with
; its own negative would end each region at the next positive, before the point
; where emitNonSemanticGlobalDebugInfo() would add a duplicate.
; UNIQUE-DAG: OpTypeInt 16 0
; UNIQUE-DAG: OpTypeInt 64 0
; UNIQUE-DAG: OpTypeFloat 16
; UNIQUE-DAG: OpTypeFloat 64
; UNIQUE-DAG: OpTypeInt 32 0
; UNIQUE-NOT: OpTypeInt 16 0
; UNIQUE-NOT: OpTypeInt 64 0
; UNIQUE-NOT: OpTypeFloat 16
; UNIQUE-NOT: OpTypeFloat 64
; UNIQUE-NOT: OpTypeInt 32 0

target triple = "spirv64-unknown-unknown"

define spir_func i64 @existing_types(i16 %s, i64 %l, half %h, double %d) !dbg !5 {
entry:
    #dbg_value(i16 7, !11, !DIExpression(), !15)
    #dbg_value(i64 1234605616436508552, !12, !DIExpression(), !15)
    #dbg_value(half 0xH3C00, !13, !DIExpression(), !15)
    #dbg_value(double 1.000000e+00, !14, !DIExpression(), !15)
    #dbg_value(i32 987654321, !17, !DIExpression(), !15)
  ret i64 1234605616436508552, !dbg !15
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-value-constant-existing-type.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !6)
!5 = distinct !DISubprogram(name: "existing_types", linkageName: "existing_types", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!6 = !{!8, !7, !8, !9, !10}
!7 = !DIBasicType(name: "short", size: 16, encoding: DW_ATE_signed)
!8 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!9 = !DIBasicType(name: "half", size: 16, encoding: DW_ATE_float)
!10 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!11 = !DILocalVariable(name: "i16_constant", scope: !5, file: !1, line: 2, type: !7)
!12 = !DILocalVariable(name: "i64_constant", scope: !5, file: !1, line: 3, type: !8)
!13 = !DILocalVariable(name: "half_constant", scope: !5, file: !1, line: 4, type: !9)
!14 = !DILocalVariable(name: "double_constant", scope: !5, file: !1, line: 5, type: !10)
!15 = !DILocation(line: 6, column: 3, scope: !5)
!16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!17 = !DILocalVariable(name: "i32_constant", scope: !5, file: !1, line: 7, type: !16)
