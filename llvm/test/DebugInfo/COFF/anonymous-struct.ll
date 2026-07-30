; RUN: llc < %s -filetype=obj | llvm-readobj - --codeview | FileCheck %s
; RUN: llc < %s | llvm-mc -filetype=obj --triple=i686-windows | llvm-readobj - --codeview | FileCheck %s

; C++ source to regenerate:
; struct S {
;   int x;
;   struct { int a; } ;
; } s;

; CHECK: CodeViewTypes [
; CHECK:  FieldList ([[S_fl:.*]]) {
; CHECK:    TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK:    DataMember {
; CHECK:      Type: int (0x74)
; CHECK:      FieldOffset: 0x0
; CHECK:      Name: x
; CHECK:    }
; CHECK:    DataMember {
; CHECK:      Type: int (0x74)
; CHECK:      FieldOffset: 0x4
; CHECK:      Name: a
; CHECK:    }
; CHECK:  }
; CHECK:  Struct ({{.*}}) {
; CHECK:    TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:    MemberCount: 2
; CHECK:    Properties [ (0x0)
; CHECK:    ]
; CHECK:    FieldList: <field list> ([[S_fl]])
; CHECK:    SizeOf: 8
; CHECK:    Name: S
; CHECK:  }

source_filename = "test/DebugInfo/COFF/anonymous-struct.ll"
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc18.0.0"

%struct.S = type { i32, %struct.anon }
%struct.anon = type { i32 }

@s = common global %struct.S zeroinitializer, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!15, !16}
!llvm.ident = !{!17}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "s", scope: !2, file: !6, line: 5, type: !7, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 3.9.0 (trunk 274261) (llvm/trunk 274262)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "-", directory: "/usr/local/google/home/majnemer/llvm/src")
!4 = !{}
!5 = !{!0}
!6 = !DIFile(filename: "<stdin>", directory: "/usr/local/google/home/majnemer/llvm/src")
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !6, line: 2, size: 64, align: 32, elements: !8)
!8 = !{!9, !11}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !7, file: !6, line: 3, baseType: !10, size: 32, align: 32)
!10 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !DIDerivedType(tag: DW_TAG_member, scope: !7, file: !6, line: 4, baseType: !12, size: 32, align: 32, offset: 32)
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, scope: !7, file: !6, line: 4, size: 32, align: 32, elements: !13)
!13 = !{!14}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !12, file: !6, line: 4, baseType: !10, size: 32, align: 32)
!15 = !{i32 2, !"CodeView", i32 1}
!16 = !{i32 2, !"Debug Info Version", i32 3}
!17 = !{!"clang version 3.9.0 (trunk 274261) (llvm/trunk 274262)"}

