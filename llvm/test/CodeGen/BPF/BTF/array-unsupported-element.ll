; RUN: not llc -mtriple=bpfel -filetype=obj %s -o %t 2>&1 | FileCheck %s --implicit-check-not=error:
;
; Source code:
;   struct entry { _Atomic(typeof(nullptr)) items[2]; } nested;
; Compilation flag:
;   clang -target bpf -std=c23 -O2 -g -S -emit-llvm nested.c
;
; Clang names the C23 nullptr type "decltype(nullptr)" in debug info.
; CHECK: error: unsupported BTF type 'decltype(nullptr)'

@nested = global { [2 x ptr] } zeroinitializer, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!9}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "nested", scope: !2, file: !3, line: 1, type: !4, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, emissionKind: FullDebug, globals: !{!0})
!3 = !DIFile(filename: "nested.c", directory: "/")
!4 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "entry", file: !3, line: 1, size: 128, elements: !{!5})
!5 = !DIDerivedType(tag: DW_TAG_member, name: "items", scope: !4, file: !3, line: 1, baseType: !6, size: 128)
!6 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 128, elements: !{!8})
!7 = !DIDerivedType(tag: DW_TAG_atomic_type, baseType: !10)
!8 = !DISubrange(count: 2)
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !DIBasicType(tag: DW_TAG_unspecified_type, name: "decltype(nullptr)")
