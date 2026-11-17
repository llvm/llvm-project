; RUN: llc -mtriple=bpfel -filetype=obj -o %t1 %s
; RUN: llvm-objcopy --dump-section='.BTF'=%t2 %t1
; RUN: %python %p/print_btf.py %t2 | FileCheck %s
; RUN: llc -mtriple=bpfeb -filetype=obj -o %t1 %s
; RUN: llvm-objcopy --dump-section='.BTF'=%t2 %t1
; RUN: %python %p/print_btf.py %t2 | FileCheck %s

; Pointee definitions are pruned to forward declarations. Emit the forward
; declarations in the order their pointers are first visited, independently of
; metadata allocation order. The names deliberately are not in lexical order.

@g = global { ptr, ptr, ptr } zeroinitializer, align 8, !dbg !0

; CHECK:      [2] PTR '(anon)' type_id=7
; CHECK-NEXT: [3] PTR '(anon)' type_id=8
; CHECK-NEXT: [4] PTR '(anon)' type_id=9
; CHECK:      [7] FWD 'zulu' fwd_kind=struct
; CHECK-NEXT: [8] FWD 'alpha' fwd_kind=union
; CHECK-NEXT: [9] FWD 'middle' fwd_kind=struct

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!4}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !3, line: 1, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !20)
!3 = !DIFile(filename: "fwd-order.c", directory: "/")
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "container", size: 192, elements: !6)
!6 = !{!7, !8, !9}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "first", scope: !5, baseType: !10, size: 64)
!8 = !DIDerivedType(tag: DW_TAG_member, name: "second", scope: !5, baseType: !11, size: 64, offset: 64)
!9 = !DIDerivedType(tag: DW_TAG_member, name: "third", scope: !5, baseType: !12, size: 64, offset: 128)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "middle", size: 32, elements: !16)
!14 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "alpha", size: 32, elements: !16)
!15 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "zulu", size: 32, elements: !16)
!16 = !{}
!20 = !{!0}
