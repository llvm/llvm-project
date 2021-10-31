; RUN: %llc_dwarf -O0 -filetype=obj -dwarf-linkage-names=Abstract < %s | llvm-dwarfdump -debug-info - > %t
; RUN: FileCheck %s -check-prefix=ONENAME < %t
; RUN: FileCheck %s -check-prefix=REF < %t 
; Verify tuning for SCE gets us Abstract only.
; RUN: %llc_dwarf -O0 -filetype=obj -debugger-tune=sce < %s | llvm-dwarfdump -debug-info - > %t
; RUN: FileCheck %s -check-prefix=ONENAME < %t
; RUN: FileCheck %s -check-prefix=REF < %t 

; Verify that the only linkage-name present is the abstract origin of the
; inlined subprogram.

; IR generated from clang -O0 with:
; void f1();
; __attribute__((always_inline)) void f2() { 
;   f1();
; }
; void f3() {
;   f2();
; }
;
; struct F4 {
;   __attribute__((always_inline)) void f5();
; };
; void F4::f5() {
;   f1();
; }
; void f6() {
;   F4::f5();
; }

; Show that the only linkage names are for the inlined functions,
; because those are the ones with an abstract origin.
; ONENAME-NOT: {{DW_AT(_MIPS)?_linkage_name}}
; ONENAME:     {{DW_AT(_MIPS)?_linkage_name}} ("_Z2f2v")
; ONENAME-NOT: {{DW_AT(_MIPS)?_linkage_name}}
; ONENAME:     {{DW_AT(_MIPS)?_linkage_name}} ("_ZN2F42f5Ev")
; ONENAME-NOT: {{DW_AT(_MIPS)?_linkage_name}}

; For f2() we see the definition pointing to an abstract origin DIE,
; which in turn is where the linkage_name is; and then there's
; an inlined_subroutine pointing back to the abstract origin.
; The order of these DIEs is not important of course, just the links.
; REF:      DW_TAG_subprogram
; REF-NOT:  {{DW_TAG|NULL}}
; REF:      DW_AT_abstract_origin ([[F2:0x.*]] "_Z2f2v")
; REF:      [[F2]]: DW_TAG_subprogram
; REF-NEXT: linkage_name ("_Z2f2v")
; REF:      DW_TAG_inlined_subroutine
; REF-NOT:  {{DW_TAG|NULL}}
; REF:      DW_AT_abstract_origin ([[F2]]

; For F4::f5(), first we see the in-class declaration,
; then the definition, abstract origin, and the inlined_subroutine.
; REF:      DW_TAG_structure_type
; REF-NEXT: DW_AT_name ("F4")
; REF-NOT:  {{DW_TAG|NULL}}
; REF:      [[F5_DECL:0x.*]]: DW_TAG_subprogram
; REF-NEXT: DW_AT_name ("f5")
; REF:      DW_TAG_subprogram
; REF-NOT:  {{DW_TAG|NULL}}
; REF:      DW_AT_abstract_origin ([[F5_ABS:0x.*]] "_ZN2F42f5Ev")
; REF:      [[F5_ABS]]: DW_TAG_subprogram
; REF-NOT:  {{DW_TAG|NULL}}
; REF:      linkage_name ("_ZN2F42f5Ev")
; REF-NEXT: DW_AT_specification ([[F5_DECL]]
; REF:      DW_TAG_inlined_subroutine
; REF-NOT:  {{DW_TAG|NULL}}
; REF:      DW_AT_abstract_origin ([[F5_ABS]]


; Function Attrs: alwaysinline uwtable
define void @_Z2f2v() #0 !dbg !6 {
entry:
  call void @_Z2f1v(), !dbg !9
  ret void, !dbg !10
}

declare void @_Z2f1v()

; Function Attrs: uwtable
define void @_Z2f3v() !dbg !11 {
entry:
  call void @_Z2f1v(), !dbg !12
  ret void, !dbg !14
}

; Function Attrs: alwaysinline uwtable
define void @_ZN2F42f5Ev() #0 align 2 !dbg !15 {
entry:
  call void @_Z2f1v(), !dbg !19
  ret void, !dbg !20
}

; Function Attrs: uwtable
define void @_Z2f6v() !dbg !21 {
entry:
  call void @_Z2f1v(), !dbg !22
  ret void, !dbg !24
}

attributes #0 = { alwaysinline }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 4.0.0 (trunk 288231)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "linkage-name-abstract-static.cpp", directory: "/home/probinson/projects/scratch")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 4.0.0 (trunk 288231)"}
!6 = distinct !DISubprogram(name: "f2", linkageName: "_Z2f2v", scope: !1, file: !1, line: 2, type: !7, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = !DILocation(line: 3, column: 3, scope: !6)
!10 = !DILocation(line: 4, column: 1, scope: !6)
!11 = distinct !DISubprogram(name: "f3", linkageName: "_Z2f3v", scope: !1, file: !1, line: 5, type: !7, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!12 = !DILocation(line: 3, column: 3, scope: !6, inlinedAt: !13)
!13 = distinct !DILocation(line: 6, column: 3, scope: !11)
!14 = !DILocation(line: 7, column: 1, scope: !11)
!15 = distinct !DISubprogram(name: "f5", linkageName: "_ZN2F42f5Ev", scope: !16, file: !1, line: 12, type: !7, isLocal: false, isDefinition: true, scopeLine: 12, flags: DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !18, retainedNodes: !2)
!16 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "F4", file: !1, line: 9, size: 8, elements: !17, identifier: "_ZTS2F4")
!17 = !{!18}
!18 = !DISubprogram(name: "f5", linkageName: "_ZN2F42f5Ev", scope: !16, file: !1, line: 10, type: !7, isLocal: false, isDefinition: false, scopeLine: 10, flags: DIFlagPrototyped, isOptimized: false)
!19 = !DILocation(line: 13, column: 3, scope: !15)
!20 = !DILocation(line: 14, column: 1, scope: !15)
!21 = distinct !DISubprogram(name: "f6", linkageName: "_Z2f6v", scope: !1, file: !1, line: 15, type: !7, isLocal: false, isDefinition: true, scopeLine: 15, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!22 = !DILocation(line: 13, column: 3, scope: !15, inlinedAt: !23)
!23 = distinct !DILocation(line: 16, column: 3, scope: !21)
!24 = !DILocation(line: 17, column: 1, scope: !21)
