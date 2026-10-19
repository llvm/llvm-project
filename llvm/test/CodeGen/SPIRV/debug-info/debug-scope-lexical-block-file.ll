; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

;; Verify that a DILocation whose scope is a DILexicalBlockFile resolves to the
;; underlying DISubprogram's DebugFunction, not to the DebugCompilationUnit.
;; DILexicalBlockFile's SPIR-V counterpart (DebugLexicalBlockDiscriminator)
;; cannot be used as a scope, so resolveScope() must unwrap it.
;; Also tests the nested case (DILexicalBlockFile wrapping DILexicalBlockFile).

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[NAME1:%[0-9]+]] = OpString "test_lbf"
; CHECK-DAG: [[NAME2:%[0-9]+]] = OpString "test_nested_lbf"
; CHECK-DAG: [[DF1:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugFunction [[NAME1]]
; CHECK-DAG: [[DF2:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugFunction [[NAME2]]

;; Single DILexicalBlockFile wrapping DISubprogram.
; CHECK:      OpFunction
; CHECK:      OpLabel
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugFunctionDefinition [[DF1]] {{%[0-9]+}}
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugScope [[DF1]]
; CHECK:      OpReturnValue
; CHECK-NEXT: OpFunctionEnd

;; Nested DILexicalBlockFile (LBF wrapping LBF wrapping DISubprogram).
; CHECK:      OpFunction
; CHECK:      OpLabel
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugFunctionDefinition [[DF2]] {{%[0-9]+}}
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugScope [[DF2]]
; CHECK:      OpReturnValue
; CHECK-NEXT: OpFunctionEnd

target triple = "spirv64-unknown-unknown"

define spir_func i32 @test_lbf(i32 %n) !dbg !5 {
entry:
  %r = add i32 %n, 1, !dbg !10
  ret i32 %r, !dbg !10
}

define spir_func i32 @test_nested_lbf(i32 %n) !dbg !11 {
entry:
  %r = mul i32 %n, 2, !dbg !14
  ret i32 %r, !dbg !14
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}

!4 = !DISubroutineType(types: !6)
!5 = distinct !DISubprogram(name: "test_lbf", linkageName: "test_lbf", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!6 = !{!7, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

!8 = !DIFile(filename: "test-other.c", directory: "/src")
!9 = !DILexicalBlockFile(scope: !5, file: !8, discriminator: 0)
!10 = !DILocation(line: 3, column: 10, scope: !9)

!11 = distinct !DISubprogram(name: "test_nested_lbf", linkageName: "test_nested_lbf", scope: !1, file: !1, line: 10, type: !4, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!12 = !DILexicalBlockFile(scope: !11, file: !8, discriminator: 0)
!13 = !DILexicalBlockFile(scope: !12, file: !8, discriminator: 1)
!14 = !DILocation(line: 12, column: 10, scope: !13)
