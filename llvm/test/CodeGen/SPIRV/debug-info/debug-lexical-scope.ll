; RUN: llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info --print-after=spirv-nonsemantic-debug-info -O0 -mtriple=spirv64-unknown-unknown -stop-after=spirv-nonsemantic-debug-info  %s -o - | FileCheck %s --check-prefix=CHECK-MIR
; RUN: llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --check-prefix=CHECK-OPTION 
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-MIR: [[type_void:%[0-9]+]]:type = OpTypeVoid
; CHECK-MIR: [[source:%[0-9]+]]:id(s32) = OpExtInst [[type_void]], 3, 35
; CHECK-MIR: [[string_1:%[0-9]+]]:id(s32) = OpString 1953719668, 49
; CHECK-MIR: [[func1:%[0-9]+]]:id(s32) = OpExtInst [[type_void]], 3, 20, [[string_1]](s32)
; CHECK-MIR: [[string_2:%[0-9]+]]:id(s32) = OpString 1953719668, 50
; CHECK-MIR: [[func2:%[0-9]+]]:id(s32) = OpExtInst [[type_void]], 3, 20, [[string_2]](s32)
; CHECK-MIR: [[lex_block1:%[0-9]+]]:id(s32) = OpExtInst [[type_void]], 3, 21, [[source]](s32), {{%[0-9]+}}, {{%[0-9]+}}(s32), [[func1]](s32)
; CHECK-MIR: [[lex_block_discr:%[0-9]+]]:id(s32) = OpExtInst [[type_void]], 3, 22, [[source]](s32), {{%[0-9]+}}(s32), [[lex_block1]](s32)
; CHECK-MIR: [[lex_block2:%[0-9]+]]:id(s32) = OpExtInst [[type_void]], 3, 21, [[source]](s32), {{%[0-9]+}}, {{%[0-9]+}}(s32), [[func2]](s32)

; CHECK-SPIRV: %[[#ext_inst_non_semantic:]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-SPIRV: %[[#test1:]] = OpString "test1"
; CHECK-SPIRV: %[[#test2:]] = OpString "test2"
; CHECK-SPIRV: %[[#void:]] = OpTypeVoid
; CHECK-SPIRV: %[[#Ty32:]] = OpTypeInt 32 0
; CHECK-SPIRV: %[[#zero:]] = OpConstant %[[#Ty32]] 0
; CHECK-SPIRV: %[[#debug_source:]] = OpExtInst %[[#void]] %[[#ext_inst_non_semantic]] DebugSource
; CHECK-SPIRV: %[[#func1:]] = OpExtInst %[[#void]] %[[#ext_inst_non_semantic]] DebugFunction %[[#test1]]
; CHECK-SPIRV: %[[#func2:]] = OpExtInst %[[#void]] %[[#ext_inst_non_semantic]] DebugFunction %[[#test2]]
; CHECK-SPIRV: %[[#lex_block:]] = OpExtInst %[[#void]] %[[#ext_inst_non_semantic]] DebugLexicalBlock %[[#debug_source]] %[[#]] %[[#]] %[[#func1]]
; CHECK-SPIRV: %[[#]] = OpExtInst %[[#void]] %[[#ext_inst_non_semantic]] DebugLexicalBlockDiscriminator %[[#debug_source]] %[[#]] %[[#lex_block]]
; CHECK-SPIRV: %[[#]] = OpExtInst %[[#void]] %[[#ext_inst_non_semantic]] DebugLexicalBlock %[[#debug_source]] %[[#]] %[[#]] %[[#func2]]
; CHECK-OPTION-NOT: OpExtInstImport "NonSemantic.Shader.DebugInfo.100"

define spir_func void @test1() !dbg !5 {
entry:
  %a = alloca i32, align 4
  store i32 1, ptr %a, align 4, !dbg !15
  ret void, !dbg !16
}

define spir_func void @test2() !dbg !9 {
entry:
  %b = alloca i32, align 4
  store i32 2, ptr %b, align 4, !dbg !18
  ret void, !dbg !19
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_Zig, file: !1, producer: "clang version XX.X", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "example.c", directory: "/tmp")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 7, !"Dwarf Version", i32 5}
!5 = distinct !DISubprogram(name: "test1", scope: !1, file: !1, line: 1, type: !6, spFlags: DISPFlagDefinition, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{!12}
!9 = distinct !DISubprogram(name: "test2", scope: !1, file: !1, line: 2, type: !6, spFlags: DISPFlagDefinition, unit: !0)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = distinct !DILexicalBlock(scope: !5, file: !1, line: 1, column: 5)
!14 = !DILexicalBlockFile(scope: !13, file: !1, discriminator: 1)
!15 = !DILocation(line: 1, column: 3, scope: !13)
!16 = !DILocation(line: 1, column: 7, scope: !14)
!17 = distinct !DILexicalBlock(scope: !9, file: !1, line: 2, column: 5)
!18 = !DILocation(line: 2, column: 3, scope: !17)
!19 = !DILocation(line: 2, column: 7, scope: !17)
