; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump -v -debug-info - | FileCheck --implicit-check-not "{{DW_TAG|NULL}}" %s

; Generated from the following source:
; namespace ns {
; void f();
; }
; inline __attribute__((always_inline)) void f1() {
;   using ns::f;
;   f();
; }
; void f2() { f1(); }

; Ensure that top level imported declarations don't produce an extra degenerate
; concrete subprogram definition.

; FIXME: imported entities should only be emitted to the abstract origin if one is present

; CHECK: DW_TAG_compile_unit
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_name {{.*}} "f1"
; CHECK:     DW_TAG_imported_declaration
; CHECK:     NULL
; CHECK:   DW_TAG_namespace
; CHECK:     DW_TAG_subprogram
; CHECK:     NULL
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_name {{.*}} "f2"
; CHECK:     DW_TAG_inlined_subroutine
; CHECK:       DW_TAG_imported_declaration
; CHECK:       NULL
; CHECK:     NULL
; CHECK:   NULL

; Function Attrs: noinline optnone uwtable
define void @_Z2f2v() #0 !dbg !14 {
entry:
  call void @_ZN2ns1fEv(), !dbg !15
  ret void, !dbg !17
}

declare void @_ZN2ns1fEv() #1

attributes #0 = { noinline optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11, !12}
!llvm.ident = !{!13}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 6.0.0 (trunk 309061) (llvm/trunk 309076)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, imports: !3)
!1 = !DIFile(filename: "imported-name-inlined.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch")
!2 = !{}
!3 = !{!4}
!4 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !8, file: !1, line: 5)
!5 = distinct !DISubprogram(name: "f1", linkageName: "_Z2f1v", scope: !1, file: !1, line: 4, type: !6, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !DISubprogram(name: "f", linkageName: "_ZN2ns1fEv", scope: !9, file: !1, line: 2, type: !6, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!9 = !DINamespace(name: "ns", scope: null)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{!"clang version 6.0.0 (trunk 309061) (llvm/trunk 309076)"}
!14 = distinct !DISubprogram(name: "f2", linkageName: "_Z2f2v", scope: !1, file: !1, line: 8, type: !6, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!15 = !DILocation(line: 6, column: 3, scope: !5, inlinedAt: !16)
!16 = distinct !DILocation(line: 8, column: 13, scope: !14)
!17 = !DILocation(line: 8, column: 19, scope: !14)
