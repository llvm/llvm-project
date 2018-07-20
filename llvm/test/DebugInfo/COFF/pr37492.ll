; RUN: llc < %s | FileCheck %s

; Original C++ source:
; struct Bits {
;   unsigned char b0 : 1;
;   unsigned char b1 : 1;
; } bits;
; const unsigned char *p_const;

; In PR37492, there was an issue in global type hashing where we forgot to
; consider the prefix portion of a type record when hashing it. This lead to a
; collision between this LF_BITFIELD and LF_MODIFIER record, so we only emitted
; one under the assumption that the other was redundant. Check that we emit both.

; CHECK-LABEL: # BitField ({{.*}}) {
; CHECK-NEXT: #   TypeLeafKind: LF_BITFIELD (0x1205)
; CHECK-NEXT: #   Type: unsigned char (0x20)
; CHECK-NEXT: #   BitSize: 1
; CHECK-NEXT: #   BitOffset: 0
; CHECK-NEXT: # }

; CHECK-LABEL: # Modifier ({{.*}}) {
; CHECK-NEXT: #   TypeLeafKind: LF_MODIFIER (0x1001)
; CHECK-NEXT: #   ModifiedType: unsigned char (0x20)
; CHECK-NEXT: #   Modifiers [ (0x1)
; CHECK-NEXT: #     Const (0x1)
; CHECK-NEXT: #   ]
; CHECK-NEXT: # }

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.13.26131"

%struct.Bits = type { i8 }

@"?bits@@3UBits@@A" = dso_local global %struct.Bits zeroinitializer, align 1, !dbg !0
@"?p_const@@3PEBEEB" = dso_local global i8* null, align 8, !dbg !6

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!15, !16, !17, !18}
!llvm.ident = !{!19}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "bits", linkageName: "?bits@@3UBits@@A", scope: !2, file: !3, line: 4, type: !11, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 7.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "t.cpp", directory: "C:\5Csrc\5Cllvm-project\5Cbuild", checksumkind: CSK_MD5, checksum: "8910833bbe8b669a3787c8f44dff1313")
!4 = !{}
!5 = !{!0, !6}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "p_const", linkageName: "?p_const@@3PEBEEB", scope: !2, file: !3, line: 5, type: !8, isLocal: false, isDefinition: true)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64)
!9 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !10)
!10 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Bits", file: !3, line: 1, size: 8, flags: DIFlagTypePassByValue, elements: !12, identifier: ".?AUBits@@")
!12 = !{!13, !14}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "b0", scope: !11, file: !3, line: 2, baseType: !10, size: 1, flags: DIFlagBitField, extraData: i64 0)
!14 = !DIDerivedType(tag: DW_TAG_member, name: "b1", scope: !11, file: !3, line: 3, baseType: !10, size: 1, offset: 1, flags: DIFlagBitField, extraData: i64 0)
!15 = !{i32 2, !"CodeView", i32 1}
!16 = !{i32 2, !"Debug Info Version", i32 3}
!17 = !{i32 1, !"wchar_size", i32 2}
!18 = !{i32 7, !"PIC Level", i32 2}
!19 = !{!"clang version 7.0.0 "}
