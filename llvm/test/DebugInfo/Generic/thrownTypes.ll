; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump -debug-info - | FileCheck %s

; CHECK: DW_TAG_subprogram
; CHECK:   DW_AT_name {{.*}} "f"
; CHECK-NOT: DW_TAG
; CHECK:   DW_TAG_thrown_type
; CHECK-NEXT:   DW_AT_type {{.*}} {[[ERROR:.*]]}
; CHECK-NOT: DW_TAG
; CHECK:   DW_TAG_thrown_type
; CHECK-NEXT:   DW_AT_type {{.*}} {[[ERROR2:.*]]}
; CHECK: [[ERROR]]: DW_TAG_structure_type
; CHECK-NEXT:   DW_AT_name {{.*}} "Error"
; CHECK: [[ERROR2]]: DW_TAG_structure_type
; CHECK-NEXT:   DW_AT_name {{.*}} "DifferentError"

; Function Attrs: nounwind uwtable
define void @f() #0 !dbg !5 {
entry:
  ret void, !dbg !11
}

attributes #0 = { nounwind uwtable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}

!0 = distinct !DICompileUnit(language: DW_LANG_Swift, producer: "swiftc", isOptimized: false, emissionKind: FullDebug, file: !1)
!1 = !DIFile(filename: "f.swift", directory: "/")
!3 = !DICompositeType(tag: DW_TAG_structure_type, name: "Error")
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "DifferentError")
!5 = distinct !DISubprogram(name: "f", line: 2, isLocal: false, isDefinition: true, unit: !0, scopeLine: 2, file: !1, scope: !1, type: !6, thrownTypes: !{!3, !4})
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 1, !"Debug Info Version", i32 3}
!11 = !DILocation(line: 3, scope: !5)
