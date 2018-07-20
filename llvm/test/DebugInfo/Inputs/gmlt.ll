; REQUIRES: object-emission

; Generated from the following source compiled with clang++ -gmlt:
; void f1() {}
; void __attribute__((section("__TEXT,__bar"))) f2() {}
; void __attribute__((always_inline)) f3() { f1(); }
; void f4() { f3(); }

; Check that
;  * -gmlt includes no DW_TAG_subprograms for subprograms without inlined
;    subroutines.
;  * yet still produces DW_AT_ranges and a range list in debug_ranges that
;    describes those subprograms

; CHECK: DW_TAG_compile_unit
; CHECK:   DW_AT_ranges [DW_FORM_sec_offset] (0x00000000
; CHECK-NOT: {{DW_TAG|NULL}}

; Omitting the subprograms without inlined subroutines is not possible
; currently on Darwin as dsymutil will drop the whole CU if it has no subprograms
; (which happens with this optimization if there are no inlined subroutines).

; DARWIN:  DW_TAG_subprogram
; DARWIN-NOT: DW_TAG
; DARWIN:    DW_AT_name {{.*}} "f1"
; DARWIN-NOT: {{DW_TAG|NULL}}
; DARWIN:  DW_TAG_subprogram
; DARWIN-NOT: DW_TAG
; DARWIN:    DW_AT_name {{.*}} "f2"
; DARWIN-NOT: {{DW_TAG|NULL}}
; DARWIN:  DW_TAG_subprogram
; DARWIN-NOT: DW_TAG
; Can't check the abstract_origin value across the DARWIN/CHECK checking and
; ordering, so don't bother - just trust me, it refers to f3 down there.
; DARWIN:    DW_AT_abstract_origin
; DARWIN-NOT: {{DW_TAG|NULL}}


; FIXME: Emitting separate abstract definitions is inefficient when we could
; just attach the DW_AT_name to the inlined_subroutine directly. Except that
; would produce many string relocations. Implement string indexing in the
; skeleton CU to address the relocation problem, then remove abstract
; definitions from -gmlt here.

; CHECK: DW_TAG_subprogram
; CHECK-NEXT:     DW_AT_name {{.*}} "f3"

; FIXME: We don't really need DW_AT_inline, consumers can ignore this due to
; the absence of high_pc/low_pc/ranges and know that they just need it for
; retrieving the name of a concrete inlined instance

; CHECK-NOT: {{DW_TAG|DW_AT|NULL}}

; Check that we only provide the minimal attributes on a subprogram to save space.
; CHECK:   DW_TAG_subprogram
; CHECK-NEXT:     DW_AT_low_pc
; CHECK-NEXT:     DW_AT_high_pc
; CHECK-NEXT:     DW_AT_name
; CHECK-NOT: {{DW_TAG|DW_AT}}
; CHECK:     DW_TAG_inlined_subroutine

; As mentioned above - replace DW_AT_abstract_origin with DW_AT_name to save
; space once we have support for string indexing in non-dwo sections

; CHECK-NEXT:       DW_AT_abstract_origin {{.*}} "f3"
; CHECK-NEXT:       DW_AT_low_pc
; CHECK-NEXT:       DW_AT_high_pc
; CHECK-NEXT:       DW_AT_call_file
; CHECK-NEXT:       DW_AT_call_line

; Make sure we don't have any other subprograms here (subprograms with no
; inlined subroutines are omitted by design to save space)

; CHECK-NOT: {{DW_TAG|DW_AT}}
; CHECK: NULL
; CHECK-NOT: {{DW_TAG|DW_AT}}
; CHECK: NULL

; CHECK: .debug_ranges contents:

; ... some addresses (depends on platform (such as platforms with function
; reordering in the linker), and looks wonky on platforms with zero values
; written in relocation places (dumper needs to	be fixed to read the
; relocations rather than interpret that as the end of a range list))

; CHECK: 00000000 <End of list>


; Check that we don't emit any pubnames or pubtypes under -gmlt
; CHECK-NOT: .debug_pubnames contents:
; CHECK-NOT: .debug_pubtypes contents:

; Function Attrs: nounwind uwtable
define void @_Z2f1v() #0 !dbg !4 {
entry:
  ret void, !dbg !13
}

; Function Attrs: nounwind uwtable
define void @_Z2f2v() #0 section "__TEXT,__bar" !dbg !7 {
entry:
  ret void, !dbg !14
}

; Function Attrs: alwaysinline nounwind uwtable
define void @_Z2f3v() #1 !dbg !8 {
entry:
  call void @_Z2f1v(), !dbg !15
  ret void, !dbg !16
}

; Function Attrs: nounwind uwtable
define void @_Z2f4v() #0 !dbg !9 {
entry:
  call void @_Z2f1v() #2, !dbg !17
  ret void, !dbg !19
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { alwaysinline nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11}
!llvm.ident = !{!12}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.6.0 ", isOptimized: false, emissionKind: LineTablesOnly, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "gmlt.cpp", directory: "/tmp/dbginfo")
!2 = !{}
!4 = distinct !DISubprogram(name: "f1", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 1, file: !1, scope: !5, type: !6, retainedNodes: !2)
!5 = !DIFile(filename: "gmlt.cpp", directory: "/tmp/dbginfo")
!6 = !DISubroutineType(types: !2)
!7 = distinct !DISubprogram(name: "f2", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 2, file: !1, scope: !5, type: !6, retainedNodes: !2)
!8 = distinct !DISubprogram(name: "f3", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 3, file: !1, scope: !5, type: !6, retainedNodes: !2)
!9 = distinct !DISubprogram(name: "f4", line: 4, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 4, file: !1, scope: !5, type: !6, retainedNodes: !2)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{!"clang version 3.6.0 "}
!13 = !DILocation(line: 1, column: 12, scope: !4)
!14 = !DILocation(line: 2, column: 53, scope: !7)
!15 = !DILocation(line: 3, column: 44, scope: !8)
!16 = !DILocation(line: 3, column: 50, scope: !8)
!17 = !DILocation(line: 3, column: 44, scope: !8, inlinedAt: !18)
!18 = !DILocation(line: 4, column: 13, scope: !9)
!19 = !DILocation(line: 4, column: 19, scope: !9)
