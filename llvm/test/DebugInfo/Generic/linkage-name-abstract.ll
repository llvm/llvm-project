; RUN: %llc_dwarf -O0 -filetype=obj -dwarf-linkage-names=Abstract < %s | llvm-dwarfdump -debug-dump=info - > %t
; RUN: FileCheck %s -check-prefix=ONENAME < %t
; RUN: FileCheck %s -check-prefix=REF < %t 
; Verify tuning for SCE gets us Abstract only.
; RUN: %llc_dwarf -O0 -filetype=obj -debugger-tune=sce < %s | llvm-dwarfdump -debug-dump=info - > %t
; RUN: FileCheck %s -check-prefix=ONENAME < %t
; RUN: FileCheck %s -check-prefix=REF < %t 
; REQUIRES: object-emission

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

; Show that there's only one linkage_name.
; ONENAME:     {{DW_AT(_MIPS)?_linkage_name}}
; ONENAME-NOT: {{DW_AT(_MIPS)?_linkage_name}}

; Locate the subprogram DIE with the linkage name.
; Show that the inlined_subroutine refers to it.
; REF:     DW_TAG_subprogram
; REF:     [[FOO:0x.*]]: DW_TAG_subprogram
; REF-NOT: {{DW_TAG|NULL}}
; REF:     {{DW_AT(_MIPS)?_linkage_name}}
; REF:     DW_TAG_inlined_subroutine
; REF-NOT: {{DW_TAG|NULL}}
; REF:     DW_AT_abstract_origin {{.*}} {[[FOO]]}

; Function Attrs: alwaysinline uwtable
define void @_Z2f2v() #0 !dbg !4 {
entry:
  call void @_Z2f1v(), !dbg !11
  ret void, !dbg !12
}

declare void @_Z2f1v()

; Function Attrs: uwtable
define void @_Z2f3v() #2 !dbg !7 {
entry:
  call void @_Z2f1v(), !dbg !13
  ret void, !dbg !15
}

attributes #0 = { alwaysinline uwtable }
attributes #2 = { uwtable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.9.0 (trunk 265282)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "linkage-name-abstract.cpp", directory: "/home/probinson/projects/scratch")
!2 = !{}
!4 = distinct !DISubprogram(name: "f2", linkageName: "_Z2f2v", scope: !1, file: !1, line: 2, type: !5, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = distinct !DISubprogram(name: "f3", linkageName: "_Z2f3v", scope: !1, file: !1, line: 5, type: !5, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{!"clang version 3.9.0 (trunk 265282)"}
!11 = !DILocation(line: 3, column: 3, scope: !4)
!12 = !DILocation(line: 4, column: 1, scope: !4)
!13 = !DILocation(line: 3, column: 3, scope: !4, inlinedAt: !14)
!14 = distinct !DILocation(line: 6, column: 3, scope: !7)
!15 = !DILocation(line: 7, column: 1, scope: !7)
