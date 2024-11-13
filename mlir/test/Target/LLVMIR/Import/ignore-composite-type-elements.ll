; RUN: mlir-translate -import-llvm -mlir-print-debuginfo -split-input-file -drop-di-composite-type-elements -split-input-file  %s | FileCheck %s

; Verifies that the according flag avoids the conversion of the elements of the
; DICompositeType.

; CHECK-NOT: di_derive_type
; CHECK: #{{.+}} = #llvm.di_composite_type<tag = DW_TAG_class_type, name = "class">
; CHECK-NOT: di_derive_type

define void @composite_type() !dbg !3 {
  ret void
}

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C, file: !2)
!2 = !DIFile(filename: "debug-info.ll", directory: "/")
!3 = distinct !DISubprogram(name: "composite_type", scope: !2, file: !2, spFlags: DISPFlagDefinition, unit: !1, type: !4)
!4 = !DISubroutineType(types: !5)
!5 = !{!6}
!6 = !DICompositeType(tag: DW_TAG_class_type, name: "class", elements: !7)
!7 = !{!9}
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, flags: DIFlagArtificial | DIFlagObjectPointer)
!9 = !DIDerivedType(tag: DW_TAG_member, name: "call_field", file: !2, baseType: !8)

; // -----

; CHECK: #{{.+}} = #llvm.di_composite_type<tag = DW_TAG_array_type, name = "vector",
; CHECK-SAME: baseType = #{{.*}}, flags = Vector, elements = #llvm.di_subrange<count = 4 : i64>

define void @composite_type() !dbg !3 {
  ret void
}

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C, file: !2)
!2 = !DIFile(filename: "debug-info.ll", directory: "/")
!3 = distinct !DISubprogram(name: "composite_type", scope: !2, file: !2, spFlags: DISPFlagDefinition, unit: !1, type: !4)
!4 = !DISubroutineType(types: !5)
!5 = !{!7}
!6 = !DIBasicType(name: "int")
!7 = !DICompositeType(tag: DW_TAG_array_type, name: "vector", flags: DIFlagVector, elements: !8, baseType: !6)
!8 = !{!9}
!9 = !DISubrange(count: 4)
