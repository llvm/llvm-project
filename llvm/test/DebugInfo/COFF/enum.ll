; RUN: llc < %s -filetype=obj | llvm-readobj - --codeview | FileCheck %s

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
; CHECK-NEXT:  StringId (0x1002) {
; CHECK-NEXT:    TypeLeafKind: LF_STRING_ID (0x1605)
; CHECK-NEXT:    Id: 0x0
; CHECK-NEXT:    StringData: /foo/bar.cpp
; CHECK-NEXT:  }
; CHECK-NEXT:  UdtSourceLine (0x1003) {
; CHECK-NEXT:    TypeLeafKind: LF_UDT_SRC_LINE (0x1606)
; CHECK-NEXT:    UDT: E (0x1001)
; CHECK-NEXT:    SourceFile: /foo/bar.cpp (0x1002)
; CHECK-NEXT:    LineNumber: 1
; CHECK_NEXT  }

source_filename = "test/DebugInfo/COFF/enum.ll"
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc18.0.0"

@"\01?e@@3W4E@@A" = global i32 0, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!11, !12}
!llvm.ident = !{!13}

!0 = distinct !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "e", linkageName: "\01?e@@3W4E@@A", scope: !2, file: !6, line: 2, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 3.9.0 (trunk 272790) (llvm/trunk 272813)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !10)
!3 = !DIFile(filename: "-", directory: "/")
!4 = !{!5}
!5 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "E", file: !6, line: 1, baseType: !7, size: 32, align: 32, elements: !8)
!6 = !DIFile(filename: "bar.cpp", directory: "/foo")
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !{!9}
!9 = !DIEnumerator(name: "BLAH", value: 0)
!10 = !{!0}
!11 = !{i32 2, !"CodeView", i32 1}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{!"clang version 3.9.0 (trunk 272790) (llvm/trunk 272813)"}

