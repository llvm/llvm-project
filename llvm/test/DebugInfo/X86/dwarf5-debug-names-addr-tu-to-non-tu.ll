; RUN: llc -filetype=obj -O0 -generate-type-units -mtriple=x86_64-unknown-linux-gnu < %s \
; RUN:     | llvm-dwarfdump -debug-info -debug-names - \
; RUN:     | FileCheck %s

;; Test that an entry in the debug names table gets created for a top level DIE when the creation of TU fails.

;; clang++ -O0 main.cpp -gdwarf-5 -fdebug-types-section -gpubnames -S -emit-llvm -glldb -o main.ll
;; int foo;
;; namespace {
;; struct t1 {};
;; } // namespace
;; template <int *> struct t2 {
;;   t1 v1;
;; };
;; struct t3 {
;;   t2<&foo> v1;
;; };
;; t3 v1;

; CHECK: [[OFFSET:0x[0-9a-f]*]]:   DW_TAG_structure_type
; CHECK: [[OFFSET1:0x[0-9a-f]*]]:   DW_TAG_structure_type

; CHECK:        Bucket 0 [
; CHECK-NEXT:    Name 1 {
; CHECK-NEXT:      Hash: {{.+}}
; CHECK-NEXT:      String: {{.+}} "t3"
; CHECK-NEXT:      Entry @ {{.+}} {
; CHECK-NEXT:        Abbrev: 0x1
; CHECK-NEXT:        Tag: DW_TAG_structure_type
; CHECK-NEXT:        DW_IDX_die_offset: [[OFFSET]]
; CHECK-NEXT:        DW_IDX_parent: <parent not indexed>

; CHECK:        Name 5 {
; CHECK-NEXT:      Hash: {{.+}}
; CHECK-NEXT:      String: {{.+}} "t2<&foo>"
; CHECK-NEXT:      Entry @ 0xe1 {
; CHECK-NEXT:        Abbrev: 0x1
; CHECK-NEXT:        Tag: DW_TAG_structure_type
; CHECK-NEXT:        DW_IDX_die_offset: [[OFFSET1]]
; CHECK-NEXT:        DW_IDX_parent: <parent not indexed>

; ModuleID = 'main.cpp'
source_filename = "main.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.t3 = type { i8 }

@foo = dso_local global i32 0, align 4, !dbg !0
@v1 = dso_local global %struct.t3 zeroinitializer, align 1, !dbg !5

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!20, !21, !22, !23, !24, !25, !26}
!llvm.ident = !{!27}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "foo", scope: !2, file: !3, line: 1, type: !19, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 20.0.0git (git@github.com:llvm/llvm-project.git ba373096e8ac83a7136fc44bc4e71a7bc53417a6)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, sysroot: "/")
!3 = !DIFile(filename: "main.cpp", directory: "/StructuredType", checksumkind: CSK_MD5, checksum: "f91f8d905197b1c0309da9526bc4776e")
!4 = !{!0, !5}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "v1", scope: !2, file: !3, line: 11, type: !7, isLocal: false, isDefinition: true)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t3", file: !3, line: 8, size: 8, flags: DIFlagTypePassByValue, elements: !8, identifier: "_ZTS2t3")
!8 = !{!9}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "v1", scope: !7, file: !3, line: 9, baseType: !10, size: 8)
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t2<&foo>", file: !3, line: 5, size: 8, flags: DIFlagTypePassByValue, elements: !11, templateParams: !16, identifier: "_ZTS2t2IXadL_Z3fooEEE")
!11 = !{!12}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "v1", scope: !10, file: !3, line: 6, baseType: !13, size: 8)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t1", scope: !14, file: !3, line: 3, size: 8, flags: DIFlagTypePassByValue, elements: !15)
!14 = !DINamespace(scope: null)
!15 = !{}
!16 = !{!17}
!17 = !DITemplateValueParameter(type: !18, value: ptr @foo)
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !19, size: 64)
!19 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!20 = !{i32 7, !"Dwarf Version", i32 5}
!21 = !{i32 2, !"Debug Info Version", i32 3}
!22 = !{i32 1, !"wchar_size", i32 4}
!23 = !{i32 8, !"PIC Level", i32 2}
!24 = !{i32 7, !"PIE Level", i32 2}
!25 = !{i32 7, !"uwtable", i32 2}
!26 = !{i32 7, !"frame-pointer", i32 2}
!27 = !{!"clang version 20.0.0git (git@github.com:llvm/llvm-project.git ba373096e8ac83a7136fc44bc4e71a7bc53417a6)"}
