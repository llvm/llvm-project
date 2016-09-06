; REQUIRES: object-emission

; RUN: %llc_dwarf < %s -filetype=obj | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; Given the following source, ensure that the decl_line/file is correctly
; emitted and omitted on definitions if it mismatches/matches the declaration

; struct foo {
;   static void f1() {
;   }
;   static void f2();
;   static void f3();
; };
; void foo::f2() {
;   f1(); // just to ensure f1 is emitted
; }
; #line 1 "bar.cpp"
; void foo::f3() {
; }

; Skip the declarations
; CHECK: DW_TAG_subprogram
; CHECK: DW_TAG_subprogram
; CHECK: DW_TAG_subprogram

; CHECK: DW_TAG_subprogram
; CHECK-NOT: {{DW_TAG|NULL|DW_AT_decl_file}}
; CHECK:   DW_AT_decl_line {{.*}}7
; CHECK-NOT: {{DW_TAG|NULL|DW_AT_decl_file}}
; CHECK:   DW_AT_specification {{.*}}f2
; CHECK-NOT: {{DW_TAG|NULL|DW_AT_decl_file}}

; CHECK: DW_TAG_subprogram
; CHECK-NOT: {{DW_TAG|NULL|DW_AT_decl_line|DW_AT_decl_file}}
; CHECK:   DW_AT_specification {{.*}}f1

; CHECK: DW_TAG_subprogram
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   DW_AT_decl_file {{.*}}bar.cpp
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   DW_AT_decl_line {{.*}}1
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   DW_AT_specification {{.*}}f3

; Function Attrs: uwtable
define void @_ZN3foo2f2Ev() #0 align 2 !dbg !12 {
entry:
  call void @_ZN3foo2f1Ev(), !dbg !19
  ret void, !dbg !20
}

; Function Attrs: nounwind uwtable
define linkonce_odr void @_ZN3foo2f1Ev() #1 align 2 !dbg !15 {
entry:
  ret void, !dbg !21
}

; Function Attrs: nounwind uwtable
define void @_ZN3foo2f3Ev() #1 align 2 !dbg !13 {
entry:
  ret void, !dbg !22
}

attributes #0 = { uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!16, !17}
!llvm.ident = !{!18}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.8.0 (trunk 249440) (llvm/trunk 249465)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3)
!1 = !DIFile(filename: "def-line.cpp", directory: "/tmp/dbginfo")
!2 = !{}
!3 = !{!4}
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "foo", file: !1, line: 1, size: 8, align: 8, elements: !5, identifier: "_ZTS3foo")
!5 = !{!6, !9, !10}
!6 = !DISubprogram(name: "f1", linkageName: "_ZN3foo2f1Ev", scope: !4, file: !1, line: 2, type: !7, isLocal: false, isDefinition: false, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false)
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = !DISubprogram(name: "f2", linkageName: "_ZN3foo2f2Ev", scope: !4, file: !1, line: 4, type: !7, isLocal: false, isDefinition: false, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: false)
!10 = !DISubprogram(name: "f3", linkageName: "_ZN3foo2f3Ev", scope: !4, file: !1, line: 5, type: !7, isLocal: false, isDefinition: false, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: false)
!12 = distinct !DISubprogram(name: "f2", linkageName: "_ZN3foo2f2Ev", scope: !4, file: !1, line: 7, type: !7, isLocal: false, isDefinition: true, scopeLine: 7, flags: DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !9, variables: !2)
!13 = distinct !DISubprogram(name: "f3", linkageName: "_ZN3foo2f3Ev", scope: !4, file: !14, line: 1, type: !7, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !10, variables: !2)
!14 = !DIFile(filename: "bar.cpp", directory: "/tmp/dbginfo")
!15 = distinct !DISubprogram(name: "f1", linkageName: "_ZN3foo2f1Ev", scope: !4, file: !1, line: 2, type: !7, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !6, variables: !2)
!16 = !{i32 2, !"Dwarf Version", i32 4}
!17 = !{i32 2, !"Debug Info Version", i32 3}
!18 = !{!"clang version 3.8.0 (trunk 249440) (llvm/trunk 249465)"}
!19 = !DILocation(line: 8, column: 3, scope: !12)
!20 = !DILocation(line: 9, column: 1, scope: !12)
!21 = !DILocation(line: 3, column: 3, scope: !15)
!22 = !DILocation(line: 2, column: 1, scope: !13)
