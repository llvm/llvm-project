; RUN: llc -filetype=obj < %s | llvm-readobj --codeview - | FileCheck %s

; Objective-C++ source demonstrating the issue:
; void (^b)(void) = []() {};

; C++ source derived and modified from:
; struct S {
;   struct {
;     int a;
;   };
; } s;

; CHECK: CodeViewTypes [
; CHECK:  FieldList ([[S_fl:.*]]) {
; CHECK:    TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK:    DataMember {
; CHECK:      Type: int (0x74)
; CHECK:      FieldOffset: 0x0
; CHECK:      Name: a
; CHECK:    }
; CHECK:  }
; CHECK:  Struct ({{.*}}) {
; CHECK:    TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:    MemberCount: 2
; CHECK:    FieldList: <field list> ([[S_fl]])
; CHECK:    SizeOf: 4
; CHECK:    Name: S
; CHECK:    LinkageName: .?AUS@@
; CHECK:  }

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686--windows-msvc19.11.0"

%struct.S = type { %struct.anon }
%struct.anon = type { i32 }

@"\01?s@@3US@@A" = dso_local global %struct.S zeroinitializer, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!13, !14, !15, !16}
!llvm.ident = !{!17}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "s", linkageName: "\01?s@@3US@@A", scope: !2, file: !3, line: 5, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 7.0.0 (trunk 325940) (llvm/trunk 325939)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "/tmp/ugh.cpp", directory: "/home/smeenai/llvm/build/llvm/Debug", checksumkind: CSK_MD5, checksum: "8256b51d95df0b5e42b848a3afe9cbda")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !3, line: 1, size: 32, flags: DIFlagTypePassByValue, elements: !7, identifier: ".?AUS@@")
!7 = !{!8, !12}
!8 = distinct !DICompositeType(tag: DW_TAG_structure_type, scope: !6, file: !3, line: 2, size: 32, flags: DIFlagTypePassByValue, elements: !9, identifier: ".?AU<unnamed-type-$S1>@S@@")
!9 = !{!10}
!10 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !8, file: !3, line: 3, baseType: !11, size: 32)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DIDerivedType(tag: DW_TAG_member, scope: !6, file: !3, line: 2, baseType: !18, size: 32) ; !8 changed to !18
!13 = !{i32 1, !"NumRegisterParameters", i32 0}
!14 = !{i32 2, !"CodeView", i32 1}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !{i32 1, !"wchar_size", i32 2}
!17 = !{!"clang version 7.0.0 (trunk 325940) (llvm/trunk 325939)"}
!18 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !8) ; added manually
