; RUN: llvm-as < %s | llvm-dis | FileCheck %s --check-prefix=IR
; RUN: llc -O0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -filetype=obj < %s \
; RUN:   | llvm-dwarfdump --debug-info - | FileCheck %s

; IR: !DIStringType(name: "character", stringLengthExpression: !DIExpression(DW_OP_push_object_address, DW_OP_constu, 8, DW_OP_LLVM_user, 4)
; IR: !DISubrange(count: !DIExpression(DW_OP_push_object_address, DW_OP_constu, 32, DW_OP_LLVM_user, 4, DW_OP_deref), lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_constu, 24, DW_OP_LLVM_user, 4, DW_OP_deref), stride: !DIExpression(DW_OP_push_object_address, DW_OP_constu, 40, DW_OP_LLVM_user, 4, DW_OP_deref))

; CHECK:      DW_TAG_subrange_type
; CHECK:        DW_AT_lower_bound (DW_OP_push_object_address, DW_OP_lit24, DW_OP_LLVM_user DW_OP_LLVM_offset, DW_OP_deref)
; CHECK-NEXT:   DW_AT_count (DW_OP_push_object_address, DW_OP_constu 0x20, DW_OP_LLVM_user DW_OP_LLVM_offset, DW_OP_deref)
; CHECK-NEXT:   DW_AT_byte_stride (DW_OP_push_object_address, DW_OP_constu 0x28, DW_OP_LLVM_user DW_OP_LLVM_offset, DW_OP_deref)
; CHECK:      DW_TAG_string_type
; CHECK:        DW_AT_string_length (DW_OP_push_object_address, DW_OP_lit8, DW_OP_LLVM_user DW_OP_LLVM_offset)

@a = addrspace(1) global ptr null, align 8, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!13, !14}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 1, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran95, file: !3, producer: "flang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4)
!3 = !DIFile(filename: "test.f90", directory: "/")
!4 = !{!0}
!5 = !DICompositeType(tag: DW_TAG_array_type, baseType: !6, elements: !7, dataLocation: !DIExpression(DW_OP_push_object_address, DW_OP_deref))
!6 = !DIStringType(name: "character", stringLengthExpression: !DIExpression(DW_OP_push_object_address, DW_OP_constu, 8, DW_OP_LLVM_user, 4), encoding: DW_ATE_ASCII)
!7 = !{!8}
!8 = !DISubrange(count: !DIExpression(DW_OP_push_object_address, DW_OP_constu, 32, DW_OP_LLVM_user, 4, DW_OP_deref), lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_constu, 24, DW_OP_LLVM_user, 4, DW_OP_deref), stride: !DIExpression(DW_OP_push_object_address, DW_OP_constu, 40, DW_OP_LLVM_user, 4, DW_OP_deref))
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 7, !"Dwarf Version", i32 5}
