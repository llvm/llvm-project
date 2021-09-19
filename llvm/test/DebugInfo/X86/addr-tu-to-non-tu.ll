; RUN: llc -filetype=obj -O0 -generate-type-units -mtriple=x86_64-unknown-linux-gnu -split-dwarf-file=x.dwo < %s \
; RUN:     | llvm-dwarfdump -debug-info -debug-types - \
; RUN:     | FileCheck --implicit-check-not=Unit --implicit-check-not=contents --implicit-check-not=declaration %s

; Test that an address-using-with-Split-DWARF type unit that references a
; non-type unit is handled correctly. A NonTypeUnitContext is used to insulate
; the type construction from being discarded when the prior/outer type has to be
; discarded due to finding it used an address & so can't be type united under
; Split DWARF. 

; The intermediate types tu and t2 are here just to test a bit more
; thoroughly/broadly. They also demonstrate one slight limitation/sub-optimality
; since 't2' isn't put in a type unit.


; extern int foo;
; namespace {
; struct t1 {
; };
; }
; template <int *> struct t2 {
;   t1 v1;
; };
; struct t3 {
;   t2<&foo> v1;
; };
; t3 v1;

; CHECK: .debug_info contents:
; CHECK: Compile Unit:

; CHECK: .debug_info.dwo contents:
; CHECK: Compile Unit:

; FIXME: In theory "t3" could be in a type unit - but at the moment, because it
;        references t2, which needs an address, t3 gets non-type-united.
;        But the same doesn't happen if t3 referenced an anonymous namespace type.

; CHECK: DW_TAG_structure_type
; CHECK:   DW_AT_name ("t3")
; CHECK:   DW_TAG_member
; CHECK:     DW_AT_type {{.*}} "t2<&foo>"
; CHECK: DW_TAG_namespace
; CHECK: [[T1:0x[0-9a-f]*]]:  DW_TAG_structure_type
; CHECK:     DW_AT_name    ("t1")
; CHECK: DW_TAG_structure_type
; CHECK:   DW_AT_name ("t2<&foo>")
; CHECK:   DW_TAG_member
; CHECK:     DW_AT_name    ("v1")
; CHECK:     DW_AT_type    ([[T1]] "(anonymous namespace)::t1")

; CHECK: .debug_types contents:

; CHECK-NOT: .debug_types.dwo contents:


%struct.t3 = type { %struct.t2 }
%struct.t2 = type { %"struct.(anonymous namespace)::t1" }
%"struct.(anonymous namespace)::t1" = type { i8 }

@v1 = dso_local global %struct.t3 zeroinitializer, align 1, !dbg !0
@foo = external dso_local global i32, align 4

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!18, !19, !20}
!llvm.ident = !{!21}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "v1", scope: !2, file: !3, line: 16, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 12.0.0 (git@github.com:llvm/llvm-project.git be646ae2865371c7a4966797e88f355de5653e04)", isOptimized: false, runtimeVersion: 0, splitDebugFilename: "test.dwo", emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: GNU)
!3 = !DIFile(filename: "test.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t3", file: !3, line: 12, size: 8, flags: DIFlagTypePassByValue, elements: !7, identifier: "_ZTS2t3")
!7 = !{!8}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "v1", scope: !6, file: !3, line: 13, baseType: !9, size: 8)
!9 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t2<&foo>", file: !3, line: 8, size: 8, flags: DIFlagTypePassByValue, elements: !10, templateParams: !14, identifier: "_ZTS2t2IXadL_Z3fooEEE")
!10 = !{!11}
!11 = !DIDerivedType(tag: DW_TAG_member, name: "v1", scope: !9, file: !3, line: 9, baseType: !12, size: 8)
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t1", scope: !13, file: !3, line: 4, size: 8, flags: DIFlagTypePassByValue, elements: !4)
!13 = !DINamespace(scope: null)
!14 = !{!15}
!15 = !DITemplateValueParameter(type: !16, value: i32* @foo)
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !17, size: 64)
!17 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!18 = !{i32 7, !"Dwarf Version", i32 4}
!19 = !{i32 2, !"Debug Info Version", i32 3}
!20 = !{i32 1, !"wchar_size", i32 4}
!21 = !{!"clang version 12.0.0 (git@github.com:llvm/llvm-project.git be646ae2865371c7a4966797e88f355de5653e04)"}
