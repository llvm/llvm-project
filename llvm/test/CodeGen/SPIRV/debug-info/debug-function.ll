; RUN: llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV: %[[#ext_inst_non_semantic:]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-SPIRV: %[[#test1:]] = OpString "test1"
; CHECK-SPIRV: %[[#test2:]] = OpString "test2"
; CHECK-SPIRV: %[[#void:]] = OpTypeVoid
; CHECK-SPIRV: %[[#Ty32:]] = OpTypeInt 32 0
; CHECK-SPIRV: %[[#zero:]] = OpConstant %[[#Ty32]] 0
; CHECK-SPIRV: %[[#debug_source:]] = OpExtInst %[[#void]] %[[#ext_inst_non_semantic]] DebugSource
; CHECK-SPIRV: %[[#debug_compilation:]] = OpExtInst %[[#void]] %[[#ext_inst_non_semantic]] DebugCompilationUnit
; CHECK-SPIRV: %[[#typefunc:]] = OpExtInst %[[#void]] %[[#ext_inst_non_semantic]] DebugTypeFunction %[[#zero]]
; CHECK-SPIRV: %[[#func1:]] = OpExtInst %[[#void]] %[[#ext_inst_non_semantic]] DebugFunction %[[#test1]] %[[#typefunc]] %[[#debug_source]] %[[#]] %[[#]] %[[#debug_compilation]]
; CHECK-SPIRV: %[[#func2:]] = OpExtInst %[[#void]] %[[#ext_inst_non_semantic]] DebugFunction %[[#test2]] %[[#typefunc]] %[[#debug_source]] %[[#]] %[[#]] %[[#debug_compilation]]

define spir_func void @test1() !dbg !5 {
entry:
  %a = alloca i32, align 4
  store i32 1, ptr %a, align 4, !dbg !8
  ret void
}

define spir_func void @test2() !dbg !9 {
entry:
  %b = alloca i32, align 4
  store i32 2, ptr %b, align 4, !dbg !11
  ret void
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
!8 = !DILocation(line: 1, column: 3, scope: !5)
!9 = distinct !DISubprogram(name: "test2", scope: !1, file: !1, line: 2, type: !6, spFlags: DISPFlagDefinition, unit: !0)
!11 = !DILocation(line: 2, column: 3, scope: !9)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
