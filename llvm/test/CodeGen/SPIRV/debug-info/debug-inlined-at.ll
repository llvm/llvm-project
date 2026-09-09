; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Exercise a single-level DebugInlinedAt.
;
;   static inline __attribute__((always_inline)) int callee(int x) {
;     return x * 3;                     // line 2
;   }
;   int caller(int y) {
;     int r = callee(y);                // line 6, calls callee at col 11
;     return r - 5;                     // line 7
;   }
;

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[NAME_CALLEE:%[0-9]+]] = OpString "callee"
; CHECK-DAG: [[NAME_CALLER:%[0-9]+]] = OpString "caller"
; CHECK-DAG: [[DS:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugSource
; CHECK-DAG: [[DF_CALLEE:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugFunction [[NAME_CALLEE]]
; CHECK-DAG: [[DF_CALLER:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugFunction [[NAME_CALLER]]
; CHECK-DAG: [[V6:%[0-9]+]] = OpConstant [[I32]] 6{{$}}

; CHECK: [[IA:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugInlinedAt [[V6]] [[DF_CALLER]]

; CHECK: [[FN:%[0-9]+]] = OpFunction {{.*}} ; -- Begin function caller
; CHECK-NEXT: OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugFunctionDefinition [[DF_CALLER]] [[FN]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugScope [[DF_CALLEE]] [[IA]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]]
; CHECK-NEXT: OpIMul
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugScope [[DF_CALLER]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]]
; CHECK-NEXT: OpIAdd
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]]
; CHECK-NEXT: OpReturnValue
; CHECK-NEXT: OpFunctionEnd

target triple = "spirv64-unknown-unknown"

define spir_func i32 @caller(i32 %y) !dbg !11 {
entry:
  %mul.i = mul nsw i32 %y, 3, !dbg !24
  %sub = add nsw i32 %mul.i, -5, !dbg !25
  ret i32 %sub, !dbg !26
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-inlined-at.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}

!11 = distinct !DISubprogram(name: "caller", scope: !1, file: !1, line: 5, type: !12, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!12 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !13)
!13 = !{!14, !14}
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

!20 = distinct !DISubprogram(name: "callee", scope: !1, file: !1, line: 1, type: !12, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)

!23 = distinct !DILocation(line: 6, column: 11, scope: !11)
!24 = !DILocation(line: 2, column: 12, scope: !20, inlinedAt: !23)
!25 = !DILocation(line: 7, column: 12, scope: !11)
!26 = !DILocation(line: 7, column: 3, scope: !11)
