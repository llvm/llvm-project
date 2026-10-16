; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info -spirv-debug-lexical-block-discriminator-scope %s -o - | FileCheck %s

; A DILocation scoped to a DILexicalBlockFile. The block-file becomes a
; DebugLexicalBlockDiscriminator whose parent is the DebugLexicalBlock, and
; DebugScope uses that discriminator as its scope.
;
;   !6  DISubprogram "foo"           -> DebugFunction                  [[DF]]
;   `- !13 DILexicalBlock 4:3        -> DebugLexicalBlock              [[LB]]
;      `- !12 DILexicalBlockFile d=1 -> DebugLexicalBlockDiscriminator [[DISC]]
;
; spirv-val rejects this module. DebugScope's Scope operand must be a lexical
; scope, and the spec does not list DebugLexicalBlockDiscriminator as one.
; DebugScope is the only instruction that gives instructions a scope, and
; DebugLine has no discriminator operand, so the spec leaves no valid way to
; attach a discriminator to code. SPIRV-LLVM-Translator uses the discriminator
; as the DebugScope operand, as here, so the value round-trips.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[PATH:%[0-9]+]] = OpString "{{[/\\]}}tmp{{[/\\]}}foo.c"
; CHECK-DAG: [[NAME:%[0-9]+]] = OpString "foo"
; CHECK-DAG: [[C1:%[0-9]+]] = OpConstant [[I32]] 1
; CHECK-DAG: [[C3:%[0-9]+]] = OpConstant [[I32]] 3
; CHECK-DAG: [[C4:%[0-9]+]] = OpConstant [[I32]] 4
; CHECK-DAG: [[DS:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugSource [[PATH]]
; CHECK-DAG: [[DF:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugFunction [[NAME]]
; CHECK-DAG: [[LB:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLexicalBlock [[DS]] [[C4]] [[C3]] [[DF]]
; CHECK-DAG: [[DISC:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLexicalBlockDiscriminator [[DS]] [[C1]] [[LB]]
; CHECK-DAG: [[CU:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugCompilationUnit

; CHECK:      OpFunction
; CHECK:      OpLabel
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugFunctionDefinition [[DF]] {{%[0-9]+}}
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugScope [[DISC]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[C4]] [[C4]] {{%[0-9]+}} {{%[0-9]+}}
; CHECK-NEXT: {{%[0-9]+}} = OpSGreaterThan
; CHECK-NOT:  DebugScope [[CU]]

target triple = "spirv64-unknown-unknown"

define spir_func i32 @foo(i32 %N) !dbg !6 {
entry:
  %cmp = icmp sgt i32 %N, 0, !dbg !11
  br i1 %cmp, label %t, label %f, !dbg !11

t:
  ret i32 1, !dbg !14

f:
  ret i32 0, !dbg !14
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "foo.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 2, type: !7, isDefinition: true, isOptimized: true, unit: !0)
!7 = !DISubroutineType(types: !2)
!11 = !DILocation(line: 4, scope: !12)
!12 = !DILexicalBlockFile(scope: !13, file: !1, discriminator: 1)
!13 = distinct !DILexicalBlock(scope: !6, file: !1, line: 4, column: 3)
!14 = !DILocation(line: 5, scope: !6)
