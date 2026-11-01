; RUN: llc -mtriple=nvptx64-nvidia-cuda < %s | FileCheck %s

; The DWARF address space may be pushed with DW_OP_lit<n> as well as
; DW_OP_constu, and dereferenced with DW_OP_xderef_size as well as
; DW_OP_xderef. In every case the sequence is stripped from the location and
; emitted as DW_AT_address_class instead, leaving a bare 9-byte
; DW_OP_addr <address> location.
;
; Each variable uses a different address space, none of which is the shared
; space (8) that would be derived from the globals' IR address space, so each
; DW_AT_address_class can only come from its own expression.

; CHECK:      .b8 65                                  // DW_AT_name
; CHECK:      .b8 5                                   // DW_AT_address_class
; CHECK-NEXT: .b8 9                                   // DW_AT_location
; CHECK-NEXT: .b8 3
; CHECK-NEXT: .b64 A
; CHECK:      .b8 66                                  // DW_AT_name
; CHECK:      .b8 4                                   // DW_AT_address_class
; CHECK-NEXT: .b8 9                                   // DW_AT_location
; CHECK-NEXT: .b8 3
; CHECK-NEXT: .b64 B
; CHECK:      .b8 67                                  // DW_AT_name
; CHECK:      .b8 6                                   // DW_AT_address_class
; CHECK-NEXT: .b8 9                                   // DW_AT_location
; CHECK-NEXT: .b8 3
; CHECK-NEXT: .b64 C

@A = addrspace(3) externally_initialized global i32 poison, align 4, !dbg !0
@B = addrspace(3) externally_initialized global i32 poison, align 4, !dbg !5
@C = addrspace(3) externally_initialized global i32 poison, align 4, !dbg !7

define ptx_kernel void @test() !dbg !14 {
  store i32 0, ptr addrspacecast (ptr addrspace(3) @A to ptr), align 4, !dbg !17
  store i32 0, ptr addrspacecast (ptr addrspace(3) @B to ptr), align 4, !dbg !17
  store i32 0, ptr addrspacecast (ptr addrspace(3) @C to ptr), align 4, !dbg !17
  ret void, !dbg !17
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!11, !12}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression(DW_OP_lit5, DW_OP_swap, DW_OP_xderef))
!1 = distinct !DIGlobalVariable(name: "A", scope: !2, file: !3, line: 1, type: !10, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, emissionKind: FullDebug, globals: !4, nameTableKind: None)
!3 = !DIFile(filename: "test.cu", directory: "/tmp")
!4 = !{!0, !5, !7}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression(DW_OP_constu, 4, DW_OP_swap, DW_OP_xderef_size, 4))
!6 = distinct !DIGlobalVariable(name: "B", scope: !2, file: !3, line: 2, type: !10, isLocal: false, isDefinition: true)
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression(DW_OP_lit6, DW_OP_swap, DW_OP_xderef_size, 4))
!8 = distinct !DIGlobalVariable(name: "C", scope: !2, file: !3, line: 3, type: !10, isLocal: false, isDefinition: true)
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{i32 2, !"Dwarf Version", i32 2}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!14 = distinct !DISubprogram(name: "test", scope: !3, file: !3, line: 5, type: !15, scopeLine: 5, spFlags: DISPFlagDefinition, unit: !2)
!15 = !DISubroutineType(types: !16)
!16 = !{null}
!17 = !DILocation(line: 6, column: 1, scope: !14)
