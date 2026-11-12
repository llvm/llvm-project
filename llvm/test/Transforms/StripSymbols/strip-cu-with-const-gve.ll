; This test checks that strip-dead-debug-info pass deletes debug compile units
; if global constants from those units are absent in the module.

; RUN: opt -passes='strip-dead-debug-info,verify' -strip-global-constants %s -S | FileCheck %s

; CHECK: !llvm.dbg.cu = !{!{{[0-9]+}}, !{{[0-9]+}}}
; CHECK-COUNT-2: !DICompileUnit

;target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
;target triple = "spir64-unknown-unknown"

; Function Attrs: convergent nounwind
define void @dev_func1() #0 !dbg !16 {
  %1 = call i32 @dev_subfunc1(i32 0) #1, !dbg !19
  ret void, !dbg !20
}

; Function Attrs: convergent nounwind
define i32 @dev_subfunc1(i32 %0) #0 !dbg !21 {
  ret i32 %0, !dbg !24
}

; Function Attrs: convergent nounwind
define void @dev_func2() #0 !dbg !25 {
  ret void, !dbg !26
}

attributes #0 = { convergent nounwind }
attributes #1 = { convergent }

!llvm.dbg.cu = !{!0, !8, !10}
!llvm.module.flags = !{!15}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3, imports: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "dev_func1.cpp", directory: "/home/usr/test")
!2 = !{}
!3 = !{!4}
!4 = !DIGlobalVariableExpression(var: !5, expr: !DIExpression(DW_OP_constu, 0, DW_OP_stack_value))
!5 = distinct !DIGlobalVariable(name: "ZERO", scope: !1, file: !1, line: 32, type: !6, isLocal: true, isDefinition: true)
!6 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !7)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !9, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, imports: !2, splitDebugInlining: false, nameTableKind: None)
!9 = !DIFile(filename: "dev_func2.cpp", directory: "/home/usr/test")
!10 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !11, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !12, imports: !2, splitDebugInlining: false, nameTableKind: None)
!11 = !DIFile(filename: "file3.cpp", directory: "/home/usr/test")
!12 = !{!13}
!13 = !DIGlobalVariableExpression(var: !14, expr: !DIExpression(DW_OP_constu, 0, DW_OP_stack_value))
!14 = distinct !DIGlobalVariable(name: "ZERO", scope: !11, file: !11, line: 145, type: !6, isLocal: true, isDefinition: true)
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = distinct !DISubprogram(name: "dev_func1", linkageName: "dev_func1", scope: !1, file: !1, line: 22, type: !17, scopeLine: 22, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!17 = !DISubroutineType(types: !18)
!18 = !{null}
!19 = !DILocation(line: 12, column: 1, scope: !16)
!20 = !DILocation(line: 29, column: 5, scope: !16)
!21 = distinct !DISubprogram(name: "dev_subfunc1", linkageName: "dev_subfunc1", scope: !1, file: !1, line: 42, type: !22, scopeLine: 42, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!22 = !DISubroutineType(types: !23)
!23 = !{!7, !7}
!24 = !DILocation(line: 5, column: 1, scope: !21)
!25 = distinct !DISubprogram(name: "dev_func2", linkageName: "dev_func2", scope: !9, file: !9, line: 22, type: !17, scopeLine: 22, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !8, retainedNodes: !2)
!26 = !DILocation(line: 29, column: 5, scope: !25)

