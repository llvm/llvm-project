; RUN: llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: invalid type ref
; CHECK: warning: ignoring invalid debug info

@bar = global i64 0, align 8, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!8}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "c", scope: null, isLocal: false, isDefinition: true, type: !6)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, retainedTypes: !4, globals: !7)
!3 = !DIFile(filename: "a", directory: "")
!4 = !{!5}
!5 = distinct !DICompositeType(tag: DW_TAG_class_type, scope: !6, file: !3, identifier: "patatino")
!6 = distinct !DISubprogram(name: "b", scope: null, isLocal: false, isDefinition: true, isOptimized: false, unit: !2)
!7 = !{!0}
!8 = !{i32 2, !"Debug Info Version", i32 3}
