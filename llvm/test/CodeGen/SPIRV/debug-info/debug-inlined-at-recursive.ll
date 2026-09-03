; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Exercise a two-level DebugInlinedAt chain. 
;
;   static inline __attribute__((always_inline)) int innermost(int x) {
;     return x * 3;                     // line 2
;   }
;   static inline __attribute__((always_inline)) int middle(int y) {
;     return innermost(y) ^ 7;          // line 6, calls innermost at col 10
;   }
;   int top(int z) {
;     int r = middle(z);                // line 10, calls middle at col 11
;     return r - 5;                     // line 11
;   }
;

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[NAME_INNER:%[0-9]+]] = OpString "innermost"
; CHECK-DAG: [[NAME_MID:%[0-9]+]] = OpString "middle"
; CHECK-DAG: [[NAME_TOP:%[0-9]+]] = OpString "top"
; CHECK-DAG: [[DS:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugSource
; CHECK-DAG: [[DF_INNER:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugFunction [[NAME_INNER]]
; CHECK-DAG: [[DF_MID:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugFunction [[NAME_MID]]
; CHECK-DAG: [[DF_TOP:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugFunction [[NAME_TOP]]
; CHECK-DAG: [[V6:%[0-9]+]] = OpConstant [[I32]] 6{{$}}
; CHECK-DAG: [[V10:%[0-9]+]] = OpConstant [[I32]] 10{{$}}

; CHECK: [[OUTER:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugInlinedAt [[V10]] [[DF_TOP]]
; CHECK-NEXT: [[INNER:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugInlinedAt [[V6]] [[DF_MID]] [[OUTER]]

; CHECK: [[TOP:%[0-9]+]] = OpFunction {{.*}} ; -- Begin function top
; CHECK-NEXT: OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugFunctionDefinition [[DF_TOP]] [[TOP]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugScope [[DF_INNER]] [[INNER]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]]
; CHECK-NEXT: OpIMul
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugScope [[DF_MID]] [[OUTER]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]]
; CHECK-NEXT: OpBitwiseXor
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugScope [[DF_TOP]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]]
; CHECK-NEXT: OpIAdd
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]]
; CHECK-NEXT: OpReturnValue
; CHECK-NEXT: OpFunctionEnd

target triple = "spirv64-unknown-unknown"

define spir_func i32 @top(i32 %z) !dbg !11 {
entry:
  %mul.i = mul nsw i32 %z, 3, !dbg !29
  %xor.i = xor i32 %mul.i, 7, !dbg !30
  %sub = add nsw i32 %xor.i, -5, !dbg !31
  ret i32 %sub, !dbg !32
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-inlined-at-recursive.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}

!11 = distinct !DISubprogram(name: "top", scope: !1, file: !1, line: 9, type: !12, scopeLine: 9, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!12 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !13)
!13 = !{!14, !14}
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

!20 = distinct !DISubprogram(name: "middle", scope: !1, file: !1, line: 5, type: !12, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!25 = distinct !DISubprogram(name: "innermost", scope: !1, file: !1, line: 1, type: !12, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)

!23 = distinct !DILocation(line: 10, column: 11, scope: !11)
!28 = distinct !DILocation(line: 6, column: 10, scope: !20, inlinedAt: !23)

!29 = !DILocation(line: 2, column: 12, scope: !25, inlinedAt: !28)
!30 = !DILocation(line: 6, column: 23, scope: !20, inlinedAt: !23)
!31 = !DILocation(line: 11, column: 12, scope: !11)
!32 = !DILocation(line: 11, column: 3, scope: !11)
