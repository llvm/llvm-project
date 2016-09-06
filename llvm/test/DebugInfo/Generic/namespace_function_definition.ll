; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj -dwarf-linkage-names=All < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; Generated from clang with the following source:
; namespace ns {
; void func() {
; }
; }

; CHECK: DW_TAG_namespace
; CHECK-NEXT: DW_AT_name {{.*}} "ns"
; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_low_pc
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_linkage_name {{.*}} "_ZN2ns4funcEv"
; CHECK: NULL
; CHECK: NULL

; Function Attrs: nounwind uwtable
define void @_ZN2ns4funcEv() #0 !dbg !4 {
entry:
  ret void, !dbg !11
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "namespace_function_definition.cpp", directory: "/tmp/dbginfo")
!2 = !{}
!4 = distinct !DISubprogram(name: "func", linkageName: "_ZN2ns4funcEv", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 2, file: !1, scope: !5, type: !6, variables: !2)
!5 = !DINamespace(name: "ns", line: 1, file: !1, scope: null)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 1, !"Debug Info Version", i32 3}
!10 = !{!"clang version 3.5.0 "}
!11 = !DILocation(line: 3, scope: !4)
