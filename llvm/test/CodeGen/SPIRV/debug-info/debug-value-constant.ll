; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --asm-verbose=0 --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --implicit-check-not="OpCapability Int16" --implicit-check-not="OpCapability Int64" --implicit-check-not="OpCapability Float16" --implicit-check-not="OpCapability Float64"
; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --asm-verbose=0 --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --check-prefix=UNIQUE
; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --asm-verbose=0 --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --check-prefix=DROPPED
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; A constant assignment names an OpConstant, which satisfies DebugValue's
; requirement that its Value operand be the result id of a non-debug
; instruction. Machine IR stores the constant as an untyped immediate, so the
; type comes from the variable and the id is emitted at module scope, which
; every function body follows.
;
; A boolean is spelled OpConstantTrue or OpConstantFalse of OpTypeBool rather
; than OpConstant, and both share the one OpTypeBool.
; An omitted Boolean size must still preserve the constant's truth value.
;
; A location operand wider than 64 bits is narrowed before entering the
; handler's 64-bit type-and-value key. An unsupported expression is rejected
; before its wider floating-point payload is read.
; A negative integer location narrower than its variable is sign-extended by
; MachineIRBuilder, so IR collection must use that same value for the lookup.
;
; A non-semantic instruction can be removed from a module without changing it,
; so debug info must not make the module require something it otherwise would
; not. OpTypeInt 16, OpTypeInt 64, OpTypeFloat 16 and OpTypeFloat 64 each
; oblige the module to declare a capability, and spirv-val rejects the module
; without it, so a narrow or wide constant the module does not already define
; is dropped instead. The implicit-check-not options on the first RUN line
; assert that none of the four is added. OpTypeBool and the 32-bit scalars
; carry no such requirement and are created freely. OpTypeFloat takes no
; signedness operand, so creating the 32-bit float exercises a shape the
; integer path cannot.

; CHECK-DAG: [[I32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[C42:%[0-9]+]] = OpConstant [[I32]] 42{{ *$}}
; CHECK-DAG: [[CNEG:%[0-9]+]] = OpConstant [[I32]] 4294967295{{ *$}}
; CHECK-DAG: [[F32:%[0-9]+]] = OpTypeFloat 32{{ *$}}
; CHECK-DAG: [[F32C:%[0-9]+]] = OpConstant [[F32]] 1{{ *$}}
; CHECK-DAG: [[BOOL:%[0-9]+]] = OpTypeBool
; Both constants naming one [[BOOL]] is what says the handler reuses the type
; it emitted rather than adding a second OpTypeBool per constant. The UNIQUE
; prefix scans from the top of the output, so the COUNT-1 and NOT pair below
; fails on a duplicate, which a CHECK-DAG group cannot.
; CHECK-DAG: [[TRUE:%[0-9]+]] = OpConstantTrue [[BOOL]]
; CHECK-DAG: [[FALSE:%[0-9]+]] = OpConstantFalse [[BOOL]]

; UNIQUE-COUNT-1: OpTypeBool
; UNIQUE-NOT: OpTypeBool
; CHECK-DAG: [[NEGNAME:%[0-9]+]] = OpString "negative"
; CHECK-DAG: [[NARROWNAME:%[0-9]+]] = OpString "narrow"
; CHECK-DAG: [[WIDENAME:%[0-9]+]] = OpString "wide"
; CHECK-DAG: [[HALFNAME:%[0-9]+]] = OpString "half"
; CHECK-DAG: [[DOUBLENAME:%[0-9]+]] = OpString "double"
; CHECK-DAG: [[UNSIZEDYESNAME:%[0-9]+]] = OpString "unsized_yes"
; CHECK-DAG: [[UNSIZEDNONAME:%[0-9]+]] = OpString "unsized_no"
; CHECK-DAG: [[WIDESOURCENAME:%[0-9]+]] = OpString "wide_source"
; CHECK-DAG: [[WIDEFLOATNAME:%[0-9]+]] = OpString "wide_float_location"
; CHECK-DAG: [[WIDENEDNEGNAME:%[0-9]+]] = OpString "widened_negative"
; CHECK-DAG: [[F32NAME:%[0-9]+]] = OpString "single"
; CHECK-DAG: [[NEGVAR:%[0-9]+]] = OpExtInst {{%[0-9]+}} {{%[0-9]+}} DebugLocalVariable [[NEGNAME]]
; The narrow and wide variables still get a DebugLocalVariable. Only the
; binding to their value is dropped, since naming it would need a type the
; module does not have. The DROPPED prefix below asserts that: it captures the
; four ids from the module section, which precedes every function body, so its
; negative region covers all of them.
; CHECK-DAG: OpExtInst {{%[0-9]+}} {{%[0-9]+}} DebugLocalVariable [[NARROWNAME]]
; CHECK-DAG: OpExtInst {{%[0-9]+}} {{%[0-9]+}} DebugLocalVariable [[WIDENAME]]
; CHECK-DAG: OpExtInst {{%[0-9]+}} {{%[0-9]+}} DebugLocalVariable [[HALFNAME]]
; CHECK-DAG: OpExtInst {{%[0-9]+}} {{%[0-9]+}} DebugLocalVariable [[DOUBLENAME]]

; DROPPED-DAG: [[DNARROWNAME:%[0-9]+]] = OpString "narrow"
; DROPPED-DAG: [[DWIDENAME:%[0-9]+]] = OpString "wide"
; DROPPED-DAG: [[DHALFNAME:%[0-9]+]] = OpString "half"
; DROPPED-DAG: [[DDOUBLENAME:%[0-9]+]] = OpString "double"
; DROPPED-DAG: [[DNARROW:%[0-9]+]] = OpExtInst {{%[0-9]+}} {{%[0-9]+}} DebugLocalVariable [[DNARROWNAME]]
; DROPPED-DAG: [[DWIDE:%[0-9]+]] = OpExtInst {{%[0-9]+}} {{%[0-9]+}} DebugLocalVariable [[DWIDENAME]]
; DROPPED-DAG: [[DHALF:%[0-9]+]] = OpExtInst {{%[0-9]+}} {{%[0-9]+}} DebugLocalVariable [[DHALFNAME]]
; DROPPED-DAG: [[DDOUBLE:%[0-9]+]] = OpExtInst {{%[0-9]+}} {{%[0-9]+}} DebugLocalVariable [[DDOUBLENAME]]
; DROPPED-NOT: DebugValue [[DNARROW]]
; DROPPED-NOT: DebugValue [[DWIDE]]
; DROPPED-NOT: DebugValue [[DHALF]]
; DROPPED-NOT: DebugValue [[DDOUBLE]]
; CHECK-DAG: [[UNSIZEDYESVAR:%[0-9]+]] = OpExtInst {{%[0-9]+}} {{%[0-9]+}} DebugLocalVariable [[UNSIZEDYESNAME]]
; CHECK-DAG: [[UNSIZEDNOVAR:%[0-9]+]] = OpExtInst {{%[0-9]+}} {{%[0-9]+}} DebugLocalVariable [[UNSIZEDNONAME]]
; CHECK-DAG: [[WIDESOURCEVAR:%[0-9]+]] = OpExtInst {{%[0-9]+}} {{%[0-9]+}} DebugLocalVariable [[WIDESOURCENAME]]
; CHECK-DAG: [[WIDEFLOATVAR:%[0-9]+]] = OpExtInst {{%[0-9]+}} {{%[0-9]+}} DebugLocalVariable [[WIDEFLOATNAME]]
; CHECK-DAG: [[WIDENEDNEGVAR:%[0-9]+]] = OpExtInst {{%[0-9]+}} {{%[0-9]+}} DebugLocalVariable [[WIDENEDNEGNAME]]
; CHECK-DAG: [[F32VAR:%[0-9]+]] = OpExtInst {{%[0-9]+}} {{%[0-9]+}} DebugLocalVariable [[F32NAME]]

; CHECK: OpExtInst {{%[0-9]+}} {{%[0-9]+}} DebugValue {{%[0-9]+}} [[C42]]
; CHECK: OpExtInst {{%[0-9]+}} {{%[0-9]+}} DebugValue {{%[0-9]+}} [[TRUE]]
; CHECK: OpExtInst {{%[0-9]+}} {{%[0-9]+}} DebugValue {{%[0-9]+}} [[FALSE]]
; CHECK: OpExtInst {{%[0-9]+}} {{%[0-9]+}} DebugValue [[NEGVAR]] [[CNEG]]
; CHECK: OpExtInst {{%[0-9]+}} {{%[0-9]+}} DebugValue [[UNSIZEDYESVAR]] [[TRUE]]
; CHECK: OpExtInst {{%[0-9]+}} {{%[0-9]+}} DebugValue [[UNSIZEDNOVAR]] [[FALSE]]
; CHECK: OpExtInst {{%[0-9]+}} {{%[0-9]+}} DebugValue [[WIDESOURCEVAR]] [[C42]]
; CHECK: OpExtInst {{%[0-9]+}} {{%[0-9]+}} DebugValue [[WIDENEDNEGVAR]] [[CNEG]]
; CHECK: OpExtInst {{%[0-9]+}} {{%[0-9]+}} DebugValue [[F32VAR]] [[F32C]]
; CHECK-NOT: DebugValue [[WIDEFLOATVAR]]

target triple = "spirv64-unknown-unknown"

define spir_func i32 @constant_value(i32 %x) !dbg !5 {
entry:
    #dbg_value(i32 42, !9, !DIExpression(), !10)
    #dbg_value(i1 true, !11, !DIExpression(), !10)
    #dbg_value(i1 false, !12, !DIExpression(), !10)
    #dbg_value(i32 -1, !13, !DIExpression(), !10)
    #dbg_value(i16 7, !14, !DIExpression(), !10)
    #dbg_value(i64 1234605616436508552, !15, !DIExpression(), !10)
    #dbg_value(half 0xH3C00, !16, !DIExpression(), !10)
    #dbg_value(double 1.000000e+00, !17, !DIExpression(), !10)
    #dbg_value(i1 true, !22, !DIExpression(), !10)
    #dbg_value(i1 false, !23, !DIExpression(), !10)
    #dbg_value(i128 18446744073709551658, !25, !DIExpression(), !10)
    #dbg_value(i16 -1, !27, !DIExpression(), !10)
    #dbg_value(float 1.000000e+00, !30, !DIExpression(), !10)
    #dbg_value(fp128 0xL00000000000000003FFF000000000000, !26, !DIExpression(DW_OP_LLVM_convert, 64, DW_ATE_unsigned), !10)
  ret i32 %x, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-value-constant.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !6)
!6 = !{!7, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = distinct !DISubprogram(name: "constant_value", linkageName: "constant_value", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!8 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!9 = !DILocalVariable(name: "constant", scope: !5, file: !1, line: 2, type: !7)
!10 = !DILocation(line: 3, column: 3, scope: !5)
!11 = !DILocalVariable(name: "yes", scope: !5, file: !1, line: 4, type: !8)
!12 = !DILocalVariable(name: "no", scope: !5, file: !1, line: 5, type: !8)
!13 = !DILocalVariable(name: "negative", scope: !5, file: !1, line: 6, type: !7)
!14 = !DILocalVariable(name: "narrow", scope: !5, file: !1, line: 7, type: !18)
!15 = !DILocalVariable(name: "wide", scope: !5, file: !1, line: 8, type: !19)
!16 = !DILocalVariable(name: "half", scope: !5, file: !1, line: 9, type: !20)
!17 = !DILocalVariable(name: "double", scope: !5, file: !1, line: 10, type: !21)
!18 = !DIBasicType(name: "short", size: 16, encoding: DW_ATE_signed)
!19 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!20 = !DIBasicType(name: "half", size: 16, encoding: DW_ATE_float)
!21 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!22 = !DILocalVariable(name: "unsized_yes", scope: !5, file: !1, line: 11, type: !24)
!23 = !DILocalVariable(name: "unsized_no", scope: !5, file: !1, line: 12, type: !24)
!24 = !DIBasicType(name: "unsized_bool", encoding: DW_ATE_boolean)
!25 = !DILocalVariable(name: "wide_source", scope: !5, file: !1, line: 13, type: !7)
!26 = !DILocalVariable(name: "wide_float_location", scope: !5, file: !1, line: 14, type: !21)
!27 = !DILocalVariable(name: "widened_negative", scope: !5, file: !1, line: 15, type: !28)
!28 = !DIBasicType(name: "another int", size: 32, encoding: DW_ATE_signed)
!29 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!30 = !DILocalVariable(name: "single", scope: !5, file: !1, line: 16, type: !29)
