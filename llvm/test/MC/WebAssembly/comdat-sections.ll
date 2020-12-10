; RUN: llc -dwarf-version=4 -generate-type-units \
; RUN:     -filetype=obj -O0 -mtriple=wasm32-unknown-unknown < %s \
; RUN:     | obj2yaml | FileCheck %s


; CHECK:     Comdats:
; CHECK:      - Name:            '4721183873463917179'
; CHECK:        Entries:
; CHECK:          - Kind:            SECTION
; CHECK:            Index:           3

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown-wasm"

%struct.S = type { i32 }

@s = global %struct.S zeroinitializer, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10, !11}
!llvm.ident = !{!12}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "s", scope: !2, file: !3, line: 5, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 5.0.0 (trunk 295942)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "t.cpp", directory: "/home/probinson/projects/scratch")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !3, line: 1, size: 32, elements: !7, identifier: "_ZTS1S")
!7 = !{!8}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "s1", scope: !6, file: !3, line: 2, baseType: !9, size: 32)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{!"clang version 5.0.0 (trunk 295942)"}
