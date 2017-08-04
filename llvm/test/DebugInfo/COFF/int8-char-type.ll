; RUN: llc < %s -filetype=obj | llvm-readobj - -codeview | FileCheck %s

; Use character types for all 8-bit integers. The VS debugger doesn't cope well
; with the T_[U]INT1 types. Non-C language frontends are likely use the normal
; DW_ATE_[un]signed encoding for all integer types if they don't have distinct
; integer types for characters types. This was PR30552.

; CHECK-LABEL: GlobalData {
; CHECK-NEXT:    Kind: S_GDATA32 (0x110D)
; CHECK-NEXT:    DataOffset:
; CHECK-NEXT:    Type: signed char (0x10)
; CHECK-NEXT:    DisplayName: x
; CHECK-NEXT:    LinkageName: x
; CHECK-NEXT:  }

; CHECK-LABEL: GlobalData {
; CHECK-NEXT:    Kind: S_GDATA32 (0x110D)
; CHECK-NEXT:    DataOffset:
; CHECK-NEXT:    Type: unsigned char (0x20)
; CHECK-NEXT:    DisplayName: y
; CHECK-NEXT:    LinkageName: y
; CHECK-NEXT:  }

source_filename = "-"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.24210"

@x = global i8 0, align 1, !dbg !0
@y = global i8 0, align 1, !dbg !6

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!13, !14, !15}
!llvm.ident = !{!16}

!0 = distinct !DIGlobalVariableExpression(var: !1)
!1 = !DIGlobalVariable(name: "x", scope: !2, file: !8, line: 4, type: !11, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 4.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "-", directory: "C:\5Csrc\5Cllvm\5Cbuild")
!4 = !{}
!5 = !{!0, !6}
!6 = distinct !DIGlobalVariableExpression(var: !7)
!7 = !DIGlobalVariable(name: "y", scope: !2, file: !8, line: 5, type: !9, isLocal: false, isDefinition: true)
!8 = !DIFile(filename: "<stdin>", directory: "C:\5Csrc\5Cllvm\5Cbuild")
!9 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint8_t", file: !8, line: 3, baseType: !10)
!10 = !DIBasicType(size: 8, align: 8, encoding: DW_ATE_unsigned)
!11 = !DIDerivedType(tag: DW_TAG_typedef, name: "int8_t", file: !8, line: 2, baseType: !12)
!12 = !DIBasicType(size: 8, align: 8, encoding: DW_ATE_signed)
!13 = !{i32 2, !"CodeView", i32 1}
; Manually modified to use DW_ATE_unsigned
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{i32 1, !"PIC Level", i32 2}
; Manually modified to use DW_ATE_signed
!16 = !{!"clang version 4.0.0 "}

