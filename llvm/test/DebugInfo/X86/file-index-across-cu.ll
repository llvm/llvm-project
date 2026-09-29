; RUN: llc %s -filetype=obj -o - | llvm-dwarfdump -verify -
; RUN: llc %s -filetype=obj -o - | llvm-dwarfdump -debug-info - | FileCheck %s

;; This test exercises the cross-CU production of type information. Due to
;; unfortunate inputs, when the DW_TAG_variable for the global
;; @a_struct_field8typeDataE is produced it will be done in the CU for 2.cpp.
;; However, the type information for the "a_struct" structure type will have
;; already been produced in 3.cpp's CU. Thus, we'll be creating a DIE in one
;; CU from another CU. Check that when we do so, we get the source file
;; identifier correct -- if the FileID came from the wrong CU then we'd end
;; up with a false declaration file.
;;
;; Due to unrelated reasons there are two copies of the "typeData" static member
;; in this LTO-linked IR -- if that ever gets fitxed, it's fine to drop the
;; duplicate DW_TAG_variable, but we want to keep testing the DW_AT_decl_file
;; that's created across a CU boundary.
;;
;; https://github.com/llvm/llvm-project/issues/109227

; CHECK-LABEL: DW_TAG_structure_type
; CHECK-NOT:   DW_TAG
; CHECK:       DW_AT_name      ("a_struct")
; CHECK-NOT:   DW_TAG
; CHECK:       DW_TAG_variable
; CHECK-NOT:   DW_TAG
; CHECK:       DW_AT_decl_file       ("C:\Users\gbmorsej\source\bees{{\\|/}}3.cpp")
; CHECK-NOT:   DW_TAG
; CHECK:       DW_TAG_variable
; CHECK-NOT:   DW_TAG
; CHECK:       DW_AT_decl_file       ("C:\users\gbmorsej\source/bees{{\\|/}}2.cpp")

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-sie-ps5"

%struct.a_struct = type { i8 }

@__trans_tmp_2 = hidden local_unnamed_addr global %struct.a_struct zeroinitializer, align 1, !dbg !0
@a_struct_field8typeDataE = hidden local_unnamed_addr global i64 0, align 8, !dbg !13

!llvm.dbg.cu = !{!2, !15}
!llvm.linker.options = !{}
!llvm.ident = !{!22, !22}
!llvm.module.flags = !{!23, !24, !25, !26, !27, !28, !29, !30, !31, !32}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "__trans_tmp_2", scope: !2, file: !3, line: 9, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_20, file: !3, producer: "clang version 21.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!3 = !DIFile(filename: "3.cpp", directory: "C:\\Users\\gbmorsej\\source\\bees", checksumkind: CSK_MD5, checksum: "05973c817251e916cc8ba01e728764dc")
!4 = !{!0}
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "a_struct", file: !3, line: 4, size: 8, flags: DIFlagTypePassByValue, elements: !6, identifier: "redacted_struct_name")
!6 = !{!7}
!7 = !DIDerivedType(tag: DW_TAG_variable, name: "typeData", scope: !5, file: !3, line: 6, baseType: !8, flags: DIFlagStaticMember)
!8 = !DIDerivedType(tag: DW_TAG_typedef, name: "TypeData", scope: !10, file: !9, line: 6, baseType: !12)
!9 = !DIFile(filename: ".\\h1.h", directory: "C:\\Users\\gbmorsej\\source\\bees", checksumkind: CSK_MD5, checksum: "0c90de8c8e867df533d869035e11cf8c")
!10 = !DINamespace(name: "b", scope: !11)
!11 = !DINamespace(name: "a", scope: null)
!12 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!13 = !DIGlobalVariableExpression(var: !14, expr: !DIExpression())
!14 = distinct !DIGlobalVariable(name: "typeData", linkageName: "a_struct_field8typeDataE", scope: !15, file: !18, line: 10, type: !19, isLocal: false, isDefinition: true, declaration: !21)
!15 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_20, file: !16, producer: "clang version 21.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !17, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!16 = !DIFile(filename: "C:\\users\\gbmorsej\\source/bees\\2.cpp", directory: "C:\\Users\\gbmorsej\\source\\bees", checksumkind: CSK_MD5, checksum: "231f25b4978512f0d961a1d7baa9cb01")
!17 = !{!13}
!18 = !DIFile(filename: "C:\\users\\gbmorsej\\source/bees/2.cpp", directory: "", checksumkind: CSK_MD5, checksum: "231f25b4978512f0d961a1d7baa9cb01")
!19 = !DIDerivedType(tag: DW_TAG_typedef, name: "TypeData", scope: !10, file: !20, line: 6, baseType: !12)
!20 = !DIFile(filename: "h1.h", directory: "C:\\Users\\gbmorsej\\source\\bees", checksumkind: CSK_MD5, checksum: "0c90de8c8e867df533d869035e11cf8c")
!21 = !DIDerivedType(tag: DW_TAG_variable, name: "typeData", scope: !5, file: !18, line: 6, baseType: !19, flags: DIFlagStaticMember)
!22 = !{!"clang version 21.0.0"}
!23 = !{i32 7, !"Dwarf Version", i32 5}
!24 = !{i32 2, !"Debug Info Version", i32 3}
!25 = !{i32 1, !"wchar_size", i32 2}
!26 = !{i32 1, !"SIE:somestuff", i32 2}
!27 = !{i32 8, !"PIC Level", i32 2}
!28 = !{i32 7, !"uwtable", i32 2}
!29 = !{i32 7, !"frame-pointer", i32 1}
!30 = !{i32 1, !"MaxTLSAlign", i32 256}
!31 = !{i32 1, !"EnableSplitLTOUnit", i32 1}
!32 = !{i32 1, !"UnifiedLTO", i32 1}

