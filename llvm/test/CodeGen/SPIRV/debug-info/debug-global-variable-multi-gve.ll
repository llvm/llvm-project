; RUN: llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Anonymous-union-style debug info: one llvm::GlobalVariable carries multiple
; DIGlobalVariableExpression attachments (one per named union member). 

;static union {
;  int x;
;  float y;
;};
;
;int use(void) {
;  x = 1;
;  return x;
;}

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32T:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[PATH:%[0-9]+]] = OpString "{{[/\\]}}AAAAAAAAAA{{[/\\]}}BBBBBBBB{{[/\\]}}CCCCCCCCC{{[/\\]}}debug-global-variable-multi-gve.c"
; CHECK-DAG: [[NAME_X:%[0-9]+]] = OpString "x"
; CHECK-DAG: [[NAME_Y:%[0-9]+]] = OpString "y"
; CHECK-DAG: [[LINK_NAME:%[0-9]+]] = OpString "_Z1x"
; CHECK-DAG: [[STR_INT:%[0-9]+]] = OpString "int"
; CHECK-DAG: [[STR_FLOAT:%[0-9]+]] = OpString "float"
; CHECK-DAG: [[C100:%[0-9]+]] = OpConstant [[I32T]] 100
; CHECK-DAG: [[C5:%[0-9]+]] = OpConstant [[I32T]] 5
; CHECK-DAG: [[C0:%[0-9]+]] = OpConstant [[I32T]] 0
; CHECK-DAG: [[C42:%[0-9]+]] = OpConstant [[I32T]] 42
; CHECK-DAG: [[C12:%[0-9]+]] = OpConstant [[I32T]] 12
; CHECK-DAG: [[C32:%[0-9]+]] = OpConstant [[I32T]] 32
; CHECK-DAG: [[C4ENC:%[0-9]+]] = OpConstant [[I32T]] 4
; CHECK-DAG: [[C3ENC:%[0-9]+]] = OpConstant [[I32T]] 3
; CHECK-DAG: [[DS:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugSource [[PATH]]
; CHECK-DAG: [[CU:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugCompilationUnit [[C100]] [[C5]] [[DS]] [[C0]]
; CHECK-DAG: [[DTI:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugTypeBasic [[STR_INT]] [[C32]] [[C4ENC]] [[C0]]
; CHECK-DAG: [[DTF:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugTypeBasic [[STR_FLOAT]] [[C32]] [[C3ENC]] [[C0]]
; CHECK-DAG: [[GV:%[0-9]+]] = OpVariable {{.*}} CrossWorkgroup
; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugGlobalVariable [[NAME_X]] [[DTI]] [[DS]] [[C42]] [[C0]] [[CU]] [[LINK_NAME]] [[GV]] [[C12]]
; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugGlobalVariable [[NAME_Y]] [[DTF]] [[DS]] [[C42]] [[C0]] [[CU]] [[LINK_NAME]] [[GV]] [[C12]]

target triple = "spirv64-unknown-unknown"

@_Z1x = dso_local addrspace(1) global i32 0, align 4, !dbg !0, !dbg !5

define spir_func void @use() {
entry:
  store i32 1, ptr addrspace(1) @_Z1x, align 4
  ret void
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10, !11, !12, !13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "x", linkageName: "_Z1x", scope: !2, file: !3, line: 42, type: !9, isLocal: true, isDefinition: true)
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "y", linkageName: "_Z1x", scope: !2, file: !3, line: 42, type: !8, isLocal: true, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version XX.X", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "debug-global-variable-multi-gve.c", directory: "/AAAAAAAAAA/BBBBBBBB/CCCCCCCCC", checksumkind: CSK_MD5, checksum: "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
!4 = !{!0, !5}
!8 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !{i32 7, !"Dwarf Version", i32 5}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{i32 7, !"frame-pointer", i32 2}
