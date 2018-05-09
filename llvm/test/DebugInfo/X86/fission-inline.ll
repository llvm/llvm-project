; RUN: llc -split-dwarf-file=foo.dwo -O0 < %s -mtriple=x86_64-unknown-linux-gnu -filetype=obj > %t
; RUN: llvm-dwarfdump -v -debug-info %t | FileCheck %s
; RUN: llvm-objdump -r %t | FileCheck --check-prefix=RELOCS %s

; Test the emission of gmlt-like inlining information into the skeleton unit.
; This allows inline-aware symbolication/backtracing given only the linked
; executable, without needing access to the .dwos.

; A simple example of inlining generated with clang -gsplit-dwarf

; A member function is used to force emission of the declaration of the
; function into the .dwo file, which may be shared with other CUs in the dwo ;
; under fission, but should not be shared with the skeleton's CU. This also
; tests the general case of context emission, which is suppressed in gmlt-like
; data.

; Include a template just to test template parameters are not emitted in
; gmlt-like data.

; And some varargs to make sure DW_TAG_unspecified_parameters is not emitted.

; And a using declaration in a nested lexical_block... because that shouldn't
; be emitted either.

; Minor complication: after generating the LLVM IR, it was manually edited so
; that the 'f1()' call from f3 was reordered to appear between the two inlined
; f1 calls from f2. This causes f2's inlined_subroutine to use DW_AT_ranges,
; thus exercising range list generation/referencing which was buggy.

; struct foo {
;   template<typename T>
;   static void f2();
;   static void f3(...);
; };
;
; void f1();
;
; template<typename T>
; inline __attribute__((always_inline)) void foo::f2() {
;   f1();
;   f1();
; }
;
; void foo::f3(...) {
;   if (true) {
;     f1();
;     f2<int>();
;     using ::foo;
;   }
; }

; Check that we emit the usual gmlt-like data for this file, including brief
; descriptions of subprograms with inlined scopes.

; FIXME: Once tools support indexed addresses in the skeleton CU, we should use
; those (DW_FORM_addr would become DW_FORM_GNU_addr_index below) since those
; addresses will already be in the address pool anyway.

; CHECK:      DW_TAG_subprogram
; CHECK-NEXT:   DW_AT_name {{.*}} "f2<int>"
; CHECK-NOT: DW_
; CHECK:      DW_TAG_subprogram
; CHECK-NEXT:   DW_AT_low_pc [DW_FORM_addr]
; CHECK-NEXT:   DW_AT_high_pc
; CHECK-NEXT:   DW_AT_name {{.*}} "f3"
; CHECK-NOT: {{DW_|NULL}}
; CHECK:        DW_TAG_inlined_subroutine
; CHECK-NEXT:     DW_AT_abstract_origin {{.*}} "f2<int>"
; CHECK-NEXT:     DW_AT_ranges
; CHECK-NOT: {{DW_AT|DW_TAG|NULL}}
; CHECK:     DW_AT_call_file
; CHECK-NEXT:     DW_AT_call_line {{.*}} (18)
; CHECK-NOT: DW_
; CHECK: .debug_info.dwo contents:

; RELOCS-NOT: RELOCATION RECORDS FOR [.rela.debug_ranges]

; Function Attrs: uwtable
define void @_ZN3foo2f3Ez(...) #0 align 2 !dbg !10 {
entry:
  call void @_Z2f1v(), !dbg !26
  call void @_Z2f1v(), !dbg !25
  call void @_Z2f1v(), !dbg !28
  ret void, !dbg !29
}

declare void @_Z2f1v() #1

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!22, !23}
!llvm.ident = !{!24}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.6.0 ", isOptimized: false, splitDebugFilename: "fission-inline.dwo", emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !3, globals: !2, imports: !18)
!1 = !DIFile(filename: "fission-inline.cpp", directory: "/tmp/dbginfo")
!2 = !{}
!3 = !{!4}
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "foo", line: 1, size: 8, align: 8, file: !1, elements: !5, identifier: "_ZTS3foo")
!5 = !{!6}
!6 = !DISubprogram(name: "f3", linkageName: "_ZN3foo2f3Ez", line: 4, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 4, file: !1, scope: !4, type: !7)
!7 = !DISubroutineType(types: !8)
!8 = !{null, null}
!10 = distinct !DISubprogram(name: "f3", linkageName: "_ZN3foo2f3Ez", line: 15, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 15, file: !1, scope: !4, type: !7, declaration: !6, retainedNodes: !2)
!11 = distinct !DISubprogram(name: "f2<int>", linkageName: "_ZN3foo2f2IiEEvv", line: 10, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 10, file: !1, scope: !4, type: !12, templateParams: !14, declaration: !17, retainedNodes: !2)
!12 = !DISubroutineType(types: !13)
!13 = !{null}
!14 = !{!15}
!15 = !DITemplateTypeParameter(name: "T", type: !16)
!16 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!17 = !DISubprogram(name: "f2<int>", linkageName: "_ZN3foo2f2IiEEvv", line: 10, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 10, file: !1, scope: !4, type: !12, templateParams: !14)
!18 = !{!19}
!19 = !DIImportedEntity(tag: DW_TAG_imported_declaration, file: !1, line: 19, scope: !20, entity: !4)
!20 = distinct !DILexicalBlock(line: 16, column: 13, file: !1, scope: !21)
!21 = distinct !DILexicalBlock(line: 16, column: 7, file: !1, scope: !10)
!22 = !{i32 2, !"Dwarf Version", i32 4}
!23 = !{i32 2, !"Debug Info Version", i32 3}
!24 = !{!"clang version 3.6.0 "}
!25 = !DILocation(line: 17, column: 5, scope: !20)
!26 = !DILocation(line: 11, column: 3, scope: !11, inlinedAt: !27)
!27 = !DILocation(line: 18, column: 5, scope: !20)
!28 = !DILocation(line: 12, column: 3, scope: !11, inlinedAt: !27)
!29 = !DILocation(line: 21, column: 1, scope: !10)
