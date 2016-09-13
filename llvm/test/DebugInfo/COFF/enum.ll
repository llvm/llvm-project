; RUN: llc < %s -filetype=obj | llvm-readobj - -codeview | FileCheck %s

; Generated from the following C++ source:
; enum E : int { BLAH };
; E e;

; CHECK:     CodeViewTypes [
; CHECK:       FieldList (0x1000) {
; CHECK-NEXT:    TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK-NEXT:    Enumerator {
; CHECK-NEXT:      TypeLeafKind: LF_ENUMERATE (0x1502)
; CHECK-NEXT:      AccessSpecifier: Public (0x3)
; CHECK-NEXT:      EnumValue: 0
; CHECK-NEXT:      Name: BLAH
; CHECK-NEXT:    }
; CHECK-NEXT:  }
; CHECK-NEXT:  Enum (0x1001) {
; CHECK-NEXT:    TypeLeafKind: LF_ENUM (0x1507)
; CHECK-NEXT:    NumEnumerators: 1
; CHECK-NEXT:    Properties [ (0x0)
; CHECK-NEXT:    ]
; CHECK-NEXT:    UnderlyingType: int (0x74)
; CHECK-NEXT:    FieldListType: <field list> (0x1000)
; CHECK-NEXT:    Name: E
; CHECK-NEXT:  }

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc18.0.0"

@"\01?e@@3W4E@@A" = global i32 0, align 4, !dbg !9

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11}
!llvm.ident = !{!12}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.9.0 (trunk 272790) (llvm/trunk 272813)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !8)
!1 = !DIFile(filename: "-", directory: "/")
!2 = !{!3}
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "E", file: !4, line: 1, baseType: !5, size: 32, align: 32, elements: !6)
!4 = !DIFile(filename: "<stdin>", directory: "/")
!5 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = !{!7}
!7 = !DIEnumerator(name: "BLAH", value: 0)
!8 = !{!9}
!9 = distinct !DIGlobalVariable(name: "e", linkageName: "\01?e@@3W4E@@A", scope: !0, file: !4, line: 2, type: !3, isLocal: false, isDefinition: true)
!10 = !{i32 2, !"CodeView", i32 1}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{!"clang version 3.9.0 (trunk 272790) (llvm/trunk 272813)"}
