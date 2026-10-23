; asm-verbose=0 keeps AsmPrinter's ;DEBUG_VALUE: comments out of the output so
; the CHECK-NEXT chains track SPIR-V instructions only.
; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --asm-verbose=0 --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --implicit-check-not=DebugDeclare
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; A direct DBG_VALUE whose virtual register is defined by an emitted SPIR-V
; instruction. The fragment stays in DebugExpression; it is not emitted as a
; DebugValue index.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[PATH:%[0-9]+]] = OpString "{{[/\\]}}src{{[/\\]}}debug-value.c"
; CHECK-DAG: [[RESULTNAME:%[0-9]+]] = OpString "result"
; CHECK-DAG: [[DS:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugSource [[PATH]]
; CHECK-DAG: [[RESULTVAR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[RESULTNAME]]
; CHECK-DAG: [[C0:%[0-9]+]] = OpConstant [[I32]] 0{{ *$}}
; CHECK-DAG: [[C9:%[0-9]+]] = OpConstant [[I32]] 9{{ *$}}
; CHECK-DAG: [[C10:%[0-9]+]] = OpConstant [[I32]] 10{{ *$}}
; CHECK-DAG: [[C16:%[0-9]+]] = OpConstant [[I32]] 16{{ *$}}
; CHECK-DAG: [[C20:%[0-9]+]] = OpConstant [[I32]] 20{{ *$}}
; CHECK-DAG: [[C30:%[0-9]+]] = OpConstant [[I32]] 30{{ *$}}
; CHECK-DAG: [[C31:%[0-9]+]] = OpConstant [[I32]] 31{{ *$}}
; CHECK-DAG: [[C32:%[0-9]+]] = OpConstant [[I32]] 32{{ *$}}
; CHECK-DAG: [[FIRSTNAME:%[0-9]+]] = OpString "first"
; CHECK-DAG: [[ALIASNAME:%[0-9]+]] = OpString "alias"
; CHECK-DAG: [[SECONDNAME:%[0-9]+]] = OpString "second"
; CHECK-DAG: [[FIRSTVAR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[FIRSTNAME]]
; CHECK-DAG: [[ALIASVAR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[ALIASNAME]]
; CHECK-DAG: [[SECONDVAR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[SECONDNAME]]
; CHECK-DAG: [[FRAGMENT:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugOperation [[C9]] [[C0]] [[C16]]{{ *$}}
; CHECK-DAG: [[EXPR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugExpression [[FRAGMENT]]{{ *$}}
; CHECK-DAG: [[EMPTY:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugExpression{{ *$}}

; The binding follows the instruction that defines its value. Anchor the
; negative check at OpFunction so its region does not depend on where the
; CHECK-DAG group above happened to match.
; CHECK: OpFunction
; CHECK-NOT: DebugValue
; CHECK: [[SUM:%[0-9]+]] = OpIAdd
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[C20]] [[C20]] [[C9]] [[C10]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugValue [[RESULTVAR]] [[SUM]] [[EXPR]]{{ *$}}
; CHECK-NOT: DebugValue

; Two variables bound to one value, and a second value in the same block. Each
; binding must land on the instruction that defines it, not on whichever
; definition happens to come first or last.
; CHECK: OpFunction
; CHECK: [[P:%[0-9]+]] = OpIMul
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[C30]] [[C30]] [[C9]] [[C10]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugValue [[FIRSTVAR]] [[P]] [[EMPTY]]{{ *$}}
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[C31]] [[C31]] [[C9]] [[C10]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugValue [[ALIASVAR]] [[P]] [[EMPTY]]{{ *$}}
; CHECK: [[D:%[0-9]+]] = OpISub
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[C32]] [[C32]] [[C9]] [[C10]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugValue [[SECONDVAR]] [[D]] [[EMPTY]]{{ *$}}
; CHECK-NOT: DebugValue

target triple = "spirv64-unknown-unknown"

define spir_func i32 @add_one(i32 %x) !dbg !5 {
entry:
  %sum = add i32 %x, %x, !dbg !10
    #dbg_value(i32 %sum, !9, !DIExpression(DW_OP_LLVM_fragment, 0, 16), !11)
  ret i32 %sum, !dbg !12
}

define spir_func i32 @two_values(i32 %x) !dbg !20 {
entry:
  %prod = mul i32 %x, %x, !dbg !27
    #dbg_value(i32 %prod, !21, !DIExpression(), !24)
    #dbg_value(i32 %prod, !22, !DIExpression(), !25)
  %diff = sub i32 %prod, %x, !dbg !27
    #dbg_value(i32 %diff, !23, !DIExpression(), !26)
  ret i32 %diff, !dbg !28
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-value.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !6)
!6 = !{!7, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = distinct !DISubprogram(name: "add_one", linkageName: "add_one", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!9 = !DILocalVariable(name: "result", scope: !5, file: !1, line: 20, type: !7)
!10 = !DILocation(line: 3, column: 7, scope: !5)
!11 = !DILocation(line: 20, column: 9, scope: !5)
!12 = !DILocation(line: 4, column: 3, scope: !5)
!20 = distinct !DISubprogram(name: "two_values", linkageName: "two_values", scope: !1, file: !1, line: 10, type: !4, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!21 = !DILocalVariable(name: "first", scope: !20, file: !1, line: 30, type: !7)
!22 = !DILocalVariable(name: "alias", scope: !20, file: !1, line: 31, type: !7)
!23 = !DILocalVariable(name: "second", scope: !20, file: !1, line: 32, type: !7)
!24 = !DILocation(line: 30, column: 9, scope: !20)
!25 = !DILocation(line: 31, column: 9, scope: !20)
!26 = !DILocation(line: 32, column: 9, scope: !20)
!27 = !DILocation(line: 11, column: 7, scope: !20)
!28 = !DILocation(line: 12, column: 3, scope: !20)
