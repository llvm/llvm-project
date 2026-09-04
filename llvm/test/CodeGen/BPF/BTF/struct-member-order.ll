; RUN: llc -mtriple=bpfel -filetype=obj -o %t1 %s
; RUN: llvm-objcopy --dump-section='.BTF'=%t2 %t1
; RUN: %python %p/print_btf.py %t2 | FileCheck %s
; RUN: llc -mtriple=bpfeb -filetype=obj -o %t1 %s
; RUN: llvm-objcopy --dump-section='.BTF'=%t2 %t1
; RUN: %python %p/print_btf.py %t2 | FileCheck %s

; Some languages preserve source order in DI even when the compiler lays out
; fields in a different order. BTF requires struct member offsets to be
; nondecreasing. The declaration tag must follow the reordered member, and a
; DW_TAG_variant_part element must participate in the same ordering.

; CHECK:      [1] STRUCT 's' size=12 vlen=3
; CHECK-NEXT:         'first' type_id=2 bits_offset=0
; CHECK-NEXT:         'second' type_id=2 bits_offset=32
; CHECK-NEXT:         '(anon)' type_id=4 bits_offset=64
; CHECK-NEXT: [2] INT 'int' size=4 bits_offset=0 nr_bits=32 encoding=SIGNED
; CHECK-NEXT: [3] DECL_TAG 'second_tag' type_id=1 component_idx=1
; CHECK-NEXT: [4] UNION '(anon)' size=4 vlen=1
; CHECK-NEXT:         'variant' type_id=2 bits_offset=0

%struct.s = type { i32, i32, i32 }

@value = global %struct.s zeroinitializer, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!13, !14}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "value", scope: !2, file: !3, line: 1, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !3, producer: "rustc", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4)
!3 = !DIFile(filename: "test.rs", directory: "/")
!4 = !{!0}
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s", file: !3, line: 1, size: 96, elements: !6)
!6 = !{!20, !9, !7}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "first", scope: !5, file: !3, line: 1, baseType: !8, size: 32)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_member, name: "second", scope: !5, file: !3, line: 1, baseType: !8, size: 32, offset: 32, annotations: !10)
!10 = !{!11}
!11 = !{!"btf_decl_tag", !"second_tag"}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 2, !"Dwarf Version", i32 4}
!20 = !DICompositeType(tag: DW_TAG_variant_part, scope: !5, file: !3, size: 32, align: 32, offset: 64, elements: !21)
!21 = !{!22}
!22 = !DIDerivedType(tag: DW_TAG_member, name: "variant", scope: !20, file: !3, line: 1, baseType: !8, size: 32, align: 32)
