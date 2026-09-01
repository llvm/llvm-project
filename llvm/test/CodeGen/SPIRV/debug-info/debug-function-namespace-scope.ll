; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o %t.spt
; RUN: FileCheck %s --check-prefix=CHECK --input-file %t.spt
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; DISubprogram definition scoped in a DINamespace. The namespace is emitted as
; a DebugLexicalBlock (with a Name operand) and used as the DebugFunction
; Parent.

; CHECK: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: OpString "ns_fn"
; CHECK-DAG: [[NS:%[0-9]+]] = OpString "ns"
; CHECK-DAG: [[PATH:%[0-9]+]] = OpString "{{[/\\]}}tmp{{[/\\]}}namespace-scope-fn.c"
; CHECK-DAG: [[EMPTY_PATH:%[0-9]+]] = OpString ""
; CHECK-DAG: [[C100:%[0-9]+]] = OpConstant [[I32]] 100
; CHECK-DAG: [[C5:%[0-9]+]] = OpConstant [[I32]] 5
; CHECK-DAG: [[C0:%[0-9]+]] = OpConstant [[I32]] 0
; CHECK-DAG: [[DS:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugSource [[PATH]]
; CHECK-DAG: [[CU:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugCompilationUnit [[C100]] [[C5]] [[DS]] [[C0]]
; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugTypeFunction [[C0]] [[VOID]]
; CHECK-DAG: [[NS_SRC:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugSource [[EMPTY_PATH]]
; CHECK-DAG: [[LB:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLexicalBlock [[NS_SRC]] [[C0]] [[C0]] [[CU]] [[NS]]
; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugFunction {{%[0-9]+}} {{%[0-9]+}} [[DS]] {{%[0-9]+}} [[C0]] [[LB]]

target triple = "spirv64-unknown-unknown"

define spir_func void @ns_fn() !dbg !9 {
entry:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "namespace-scope-fn.c", directory: "/tmp", checksumkind: CSK_MD5, checksum: "00000000000000000000000000000000")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"frame-pointer", i32 2}

!6 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !7)
!7 = !{}
!8 = !DINamespace(name: "ns", scope: !1)

!9 = distinct !DISubprogram(name: "ns_fn", linkageName: "ns_fn", scope: !8, file: !1, line: 2, type: !6, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
