; RUN: llc -split-dwarf-file=foo.dwo -mtriple=powerpc64-unknown-linux-gnu -O0 -filetype=asm < %s | FileCheck %s

; FIXME: add relocation and DWARF expression support to llvm-dwarfdump & use
; that here instead of raw assembly printing

; CHECK: debug_info.dwo
; 3 bytes of data in this DW_FORM_block1 representation of the location of 'tls'
; CHECK: .byte 3{{ *}}# DW_AT_location
; DW_OP_const_index (0xfx == 252) to refer to the debug_addr table
; CHECK-NEXT: .byte 252
; an index of zero into the debug_addr table
; CHECK-NEXT: .byte 0
; DW_OP_GNU_push_tls_address
; CHECK-NEXT: .byte 224
; check that the expected TLS address description is the first thing in the debug_addr section
; CHECK: .section .debug_addr,"",@progbits
; CHECK-NEXT: .quad tls@DTPREL+32768

source_filename = "test/DebugInfo/PowerPC/tls-fission.ll"

@tls = thread_local global i32 0, align 4, !dbg !0

!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!7, !8}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "tls", scope: null, file: !2, line: 1, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "tls.cpp", directory: "/tmp")
!3 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!4 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "clang version 3.4 ", isOptimized: false, runtimeVersion: 0, splitDebugFilename: "tls.dwo", emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !5)
!5 = !{}
!6 = !{!0}
!7 = !{i32 2, !"Dwarf Version", i32 3}
!8 = !{i32 1, !"Debug Info Version", i32 3}

