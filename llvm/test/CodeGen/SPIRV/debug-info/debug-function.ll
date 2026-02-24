; RUN: llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; The path separator may be '/' on Unix or '\' on Windows, so we use a flexible pattern
; CHECK: OpString "{{[/\]}}A{{[/\]}}B{{[/\]}}C{{[/\]}}example.cpp"
; CHECK: [[FILE_STRING:%[0-9]+]] = OpString "{{[/\]}}A{{[/\]}}B{{[/\]}}C{{[/\]}}example.cpp"
; CHECK: [[NAME:%[0-9]+]] = OpString "test1"
; CHECK: [[LINKAGE:%[0-9]+]] = OpString "XXXX"
; CHECK: [[VOID_TYPE:%[0-9]+]] = OpTypeVoid
; CHECK: [[ZERO:%[0-9]+]] = OpConstant {{.*}} 0
; CHECK: [[LINE:%[0-9]+]] = OpConstant {{.*}} 1
; CHECK: [[FLAGS:%[0-9]+]] = OpConstant {{.*}} 8
; CHECK: OpExtInst {{.*}} DebugSource {{.*}}
; CHECK: [[PARENT:%[0-9]+]] = OpExtInst {{.*}} DebugCompilationUnit {{.*}}
; CHECK: [[SOURCE:%[0-9]+]] = OpExtInst {{.*}} DebugSource [[FILE_STRING]]
; CHECK: [[TYPE:%[0-9]+]] = OpExtInst {{.*}} DebugTypeFunction [[ZERO]] [[VOID_TYPE]]
; CHECK: [[DEBUG_FUNC:%[0-9]+]] = OpExtInst {{.*}} DebugFunction [[NAME]] [[TYPE]] [[SOURCE]] [[LINE]] [[ZERO]] [[PARENT]] [[LINKAGE]] [[FLAGS]] [[LINE]]
; CHECK: [[FUNC:%[0-9]+]] = OpFunction {{.*}}
; DebugFunctionDefinition must be after OpFunction
; CHECK: OpExtInst {{.*}} DebugFunctionDefinition [[DEBUG_FUNC]] [[FUNC]]

target triple = "spirv64-unknown-unknown"

define spir_func void  @test1() !dbg !9 {
entry:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_Zig, file: !1, producer: "clang version XX.X.XXXX (F F)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "example.cpp", directory: "/A/B/C")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"frame-pointer", i32 2}
!9 = distinct !DISubprogram(name: "test1", linkageName: "XXXX", scope: !1, file: !1, line: 1, type: !10, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!10 = !DISubroutineType(types: !11)
!11 = !{null}
