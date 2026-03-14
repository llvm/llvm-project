; RUN: opt -S <%s 2>&1| FileCheck %s
; CHECK: ignoring debug info with an invalid version (0)

; CHECK: @Var = internal global i32 0
; CHECK-NOT: !dbg
@Var = internal global i32 0, !dbg !0

; Test that StripDebugInfo strips global variables.

; CHECK-NOT: DIGlobalVariable

!0 = !DIGlobalVariable(name: "Var", line: 2, isLocal: true, isDefinition: true, scope: !1, file: !3, type: !2)
!1 = distinct !DICompileUnit(language: DW_LANG_C89, producer: "adrian", isOptimized: true, emissionKind: FullDebug, file: !3)
!2 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!3 = !DIFile(filename: "var.c", directory: "/")
