; RUN: opt -analyze -module-debuginfo < %s | FileCheck %s

; This module is generated from the following c-code:
;
; > union X;
; >
; > struct Y {
; >     union X *x;
; > };
; >
; > struct Y y;

; CHECK: Type: Y from /tmp/minimal.c:3 DW_TAG_structure_type
; CHECK: Type: x from /tmp/minimal.c:4 DW_TAG_member
; CHECK: Type: DW_TAG_pointer_type
; CHECK: Type: X from /tmp/minimal.c:1 DW_TAG_structure_type

source_filename = "test/DebugInfo/Generic/debuginfofinder-forward-declaration.ll"

%struct.Y = type { %struct.X* }
%struct.X = type opaque

@y = common global %struct.Y zeroinitializer, align 8, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!11, !12}
!llvm.ident = !{!13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "y", scope: !2, file: !3, line: 7, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 3.7.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !5, imports: !4)
!3 = !DIFile(filename: "minimal.c", directory: "/tmp")
!4 = !{}
!5 = !{!0}
!6 = !DICompositeType(tag: DW_TAG_structure_type, name: "Y", file: !3, line: 3, size: 64, align: 64, elements: !7)
!7 = !{!8}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !6, file: !3, line: 4, baseType: !9, size: 64, align: 64)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64, align: 64)
!10 = !DICompositeType(tag: DW_TAG_structure_type, name: "X", file: !3, line: 1, flags: DIFlagFwdDecl)
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{!"clang version 3.7.0"}

