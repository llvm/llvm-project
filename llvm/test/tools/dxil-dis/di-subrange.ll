; RUN: llc --filetype=obj %s -o - | dxil-dis -o - | FileCheck %s
target triple = "dxil-unknown-shadermodel6.7-library"

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 7, !"Dwarf Version", i32 2}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "xyz", isOptimized: true, runtimeVersion: 0, emissionKind: 1, retainedTypes: !4)
!3 = !DIFile(filename: "input.hlsl", directory: "/some/path")
!4 = !{!5}
!5 = !DICompositeType(tag: DW_TAG_array_type, baseType: !6, size: 128, flags: DIFlagVector, elements: !7)
!6 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!7 = !{!8}
!8 = !DISubrange(count: 3)

; CHECK: !llvm.module.flags = !{!0, !1}
; CHECK: !llvm.dbg.cu = !{!2}

; CHECK: !0 = !{i32 7, !"Dwarf Version", i32 2}
; CHECK: !1 = !{i32 2, !"Debug Info Version", i32 3}
; CHECK: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "xyz", isOptimized: true, runtimeVersion: 0, emissionKind: 1, retainedTypes: !4)
; CHECK: !3 = !DIFile(filename: "input.hlsl", directory: "/some/path")
; CHECK: !4 = !{!5}
; CHECK: !5 = !DICompositeType(tag: DW_TAG_array_type, baseType: !6, size: 128, flags: DIFlagVector, elements: !7)
; CHECK: !6 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
; CHECK: !7 = !{!8}
; CHECK: !8 = !DISubrange(count: 3)
