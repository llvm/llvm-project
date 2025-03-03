; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda | FileCheck %s

; Test that translateToNVVMDWARFAddrSpace() function translates NVVM IR address space
; value `Shared` (3) to the corresponding DWARF DW_AT_address_class attribute for PTX.

; CHECK: .section .debug_info
; CHECK:      .b8 103                                 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 55                                 // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 1                                   // DW_AT_decl_line
; CHECK-NEXT: .b8 8                                   // DW_AT_address_class

@g = internal addrspace(3) global i32 0, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!6}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", linkageName: "g", scope: !2, file: !3, line: 1, type: !5, isLocal: true, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4)
!3 = !DIFile(filename: "test.cu", directory: "test")
!4 = !{!0}
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !{i32 1, !"Debug Info Version", i32 3}
