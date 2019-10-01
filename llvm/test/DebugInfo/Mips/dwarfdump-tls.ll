; RUN: llc -O0 -march=mips -mcpu=mips32r2 -filetype=obj \
; RUN:     -split-dwarf-file=foo.dwo -o=%t-32.o < %s
; RUN: llvm-dwarfdump %t-32.o 2>&1 | FileCheck %s
; RUN: llc -O0 -march=mips64 -mcpu=mips64r2 -filetype=obj \
; RUN:     -split-dwarf-file=foo.dwo -o=%t-64.o < %s
; RUN: llvm-dwarfdump %t-64.o 2>&1 | FileCheck %s

; RUN: llc -O0 -march=mips -mcpu=mips32r2 -filetype=asm \
; RUN:     -split-dwarf-file=foo.dwo < %s | FileCheck -check-prefix=ASM32 %s
; RUN: llc -O0 -march=mips64 -mcpu=mips64r2 -filetype=asm \
; RUN:     -split-dwarf-file=foo.dwo < %s | FileCheck -check-prefix=ASM64 %s

@x = thread_local global i32 5, align 4, !dbg !0

; CHECK-NOT: error: failed to compute relocation: R_MIPS_TLS_DTPREL

; CHECK:      DW_AT_name      ("x")
; CHECK-NEXT: DW_AT_type      (0x00000025 "int")
; CHECK-NEXT: DW_AT_external  (true)
; CHECK-NEXT: DW_AT_decl_file (0x01)
; CHECK-NEXT: DW_AT_decl_line (1)
; CHECK-NEXT: DW_AT_location  (DW_OP_GNU_const_index 0x0, {{DW_OP_GNU_push_tls_address|DW_OP_form_tls_address}})

; ASM32:              .section        .debug_addr
; ASM32-NEXT: $addr_table_base0:
; ASM32-NEXT:         .4byte  x+32768

; ASM64:              .section        .debug_addr
; ASM64-NEXT: .Laddr_table_base0:
; ASM64-NEXT:         .8byte  x+32768

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "x", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 4.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "tls.c", directory: "/tmp")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
