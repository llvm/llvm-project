; RUN: llc < %s -filetype=obj | llvm-readobj - -codeview | FileCheck %s

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

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc18.0.0"

%struct.S = type { i32, %struct.anon }
%struct.anon = type { i32 }

@s = common global %struct.S zeroinitializer, align 4, !dbg !4

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!14, !15}
!llvm.ident = !{!16}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0 (trunk 274261) (llvm/trunk 274262)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3)
!1 = !DIFile(filename: "-", directory: "/usr/local/google/home/majnemer/llvm/src")
!2 = !{}
!3 = !{!4}
!4 = distinct !DIGlobalVariable(name: "s", scope: !0, file: !5, line: 5, type: !6, isLocal: false, isDefinition: true)
!5 = !DIFile(filename: "<stdin>", directory: "/usr/local/google/home/majnemer/llvm/src")
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !5, line: 2, size: 64, align: 32, elements: !7)
!7 = !{!8, !10}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !6, file: !5, line: 3, baseType: !9, size: 32, align: 32)
!9 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DIDerivedType(tag: DW_TAG_member, scope: !6, file: !5, line: 4, baseType: !11, size: 32, align: 32, offset: 32)
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, scope: !6, file: !5, line: 4, size: 32, align: 32, elements: !12)
!12 = !{!13}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !11, file: !5, line: 4, baseType: !9, size: 32, align: 32)
!14 = !{i32 2, !"CodeView", i32 1}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !{!"clang version 3.9.0 (trunk 274261) (llvm/trunk 274262)"}
