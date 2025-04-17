; RUN: llc %s -o - --filetype=obj | llvm-dwarfdump - --name A --show-children | FileCheck %s --check-prefix=TREE

;; -ggnu-pubnames (nameTableKind: GNU).
; RUN: llc %s -o - --filetype=obj \
; RUN: | llvm-dwarfdump - --debug-gnu-pubtypes \
; RUN: | FileCheck %s --check-prefix=GNU-TYPES

;; -gpubnames (remove nameTableKind field from DICompileUnit).
; RUN: sed 's/, nameTableKind: GNU//g' < %s \
; RUN: | llc - -o - --filetype=obj \
; RUN: | llvm-dwarfdump - --debug-pubtypes \
; RUN: | FileCheck %s --check-prefix=PUB-TYPES

;; C++ source from clang/test/CodeGenCXX/template-alias.cpp, compiled with -gsce:
;; template<typename Y, int Z>
;; struct X {
;;   Y m1 = Z;
;; };
;;
;; template<typename B, int C>
;; using A = X<B, C>;
;;
;; A<int, 5> a;

;; Test emission of DIDerivedType with tag: DW_TAG_template_alias.

; TREE: DW_TAG_template_alias
; TREE: DW_AT_type (0x{{[0-9a-f]+}} "X<int, 5>")
; TREE: DW_AT_name ("A")
; TREE:   DW_TAG_template_type_parameter
; TREE:     DW_AT_type        (0x{{[0-9a-f]+}} "int")
; TREE:     DW_AT_name        ("B")
; TREE:   DW_TAG_template_value_parameter
; TREE:     DW_AT_type        (0x{{[0-9a-f]+}} "int")
; TREE:     DW_AT_name        ("C")
; TREE:     DW_AT_const_value (5)
; TREE:   NULL

; GNU-TYPES: STATIC   TYPE     "A"
; PUB-TYPES: "A"

target triple = "x86_64-unknown-unkown"

%struct.X = type { i32 }

@a = global %struct.X { i32 5 }, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!17, !18}
!llvm.ident = !{!19}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !5, line: 14, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 19.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: GNU)
!3 = !DIFile(filename: "<stdin>", directory: "/")
!4 = !{!0}
!5 = !DIFile(filename: "clang/test/CodeGenCXX/template-alias.cpp", directory: "/")
!6 = !DIDerivedType(tag: DW_TAG_template_alias, name: "A", file: !5, line: 12, baseType: !7, extraData: !14)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "X<int, 5>", file: !5, line: 7, size: 32, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !8, templateParams: !11, identifier: "_ZTS1XIiLi5EE")
!8 = !{!9}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "m1", scope: !7, file: !5, line: 8, baseType: !10, size: 32)
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12, !13}
!12 = !DITemplateTypeParameter(name: "Y", type: !10)
!13 = !DITemplateValueParameter(name: "Z", type: !10, value: i32 5)
!14 = !{!15, !16}
!15 = !DITemplateTypeParameter(name: "B", type: !10)
!16 = !DITemplateValueParameter(name: "C", type: !10, value: i32 5)
!17 = !{i32 2, !"Debug Info Version", i32 3}
!18 = !{i32 1, !"wchar_size", i32 4}
!19 = !{!"clang version 19.0.0git"}
