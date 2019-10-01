; RUN: llc -dwarf-version=5 -filetype=obj -O0 < %s | llvm-dwarfdump - | FileCheck %s --match-full-lines --check-prefix=DW5-CHECK
; RUN: llc -dwarf-version=4 -filetype=obj -O0 < %s | llvm-dwarfdump - | FileCheck %s --match-full-lines --check-prefix=DW4-CHECK

; DW5-CHECK: .debug_info contents:
; DW5-CHECK-NEXT: 0x00000000: Compile Unit: length = 0x0000003e version = 0x0005 unit_type = DW_UT_compile abbr_offset = 0x0000 addr_size = 0x08 (next unit at 0x00000042)
; DW5-CHECK-EMPTY:
; DW5-CHECK-NEXT: 0x0000000c: DW_TAG_compile_unit
; DW5-CHECK-NEXT:               DW_AT_producer	("clang version 9.0.0 (trunk 353791) (llvm/trunk 353801)")
; DW5-CHECK-NEXT:               DW_AT_language	(DW_LANG_C99)
; DW5-CHECK-NEXT:               DW_AT_name	("dbg.c")
; DW5-CHECK-NEXT:               DW_AT_str_offsets_base	(0x00000008)
; DW5-CHECK-NEXT:               DW_AT_stmt_list	(0x00000000)
; DW5-CHECK-NEXT:               DW_AT_comp_dir {{.*}}
; DW5-CHECK-NEXT:               DW_AT_addr_base	(0x00000008)
; DW5-CHECK-EMPTY:
; DW5-CHECK-NEXT: 0x0000001e:   DW_TAG_base_type
; DW5-CHECK-NEXT:                 DW_AT_name	("DW_ATE_signed_8")
; DW5-CHECK-NEXT:                 DW_AT_encoding	(DW_ATE_signed)
; DW5-CHECK-NEXT:                 DW_AT_byte_size	(0x01)
; DW5-CHECK-EMPTY:
; DW5-CHECK-NEXT: 0x00000022:   DW_TAG_base_type
; DW5-CHECK-NEXT:                 DW_AT_name	("DW_ATE_signed_32")
; DW5-CHECK-NEXT:                 DW_AT_encoding	(DW_ATE_signed)
; DW5-CHECK-NEXT:                 DW_AT_byte_size	(0x04)
; DW5-CHECK-EMPTY:
; DW5-CHECK-NEXT: 0x00000026:   DW_TAG_variable
; DW5-CHECK-NEXT:                 DW_AT_name	("global")
; DW5-CHECK-NEXT:                 DW_AT_type	(0x0000003d "int")
; DW5-CHECK-NEXT:                 DW_AT_external	(true)
; DW5-CHECK-NEXT:                 DW_AT_decl_file {{.*}}
; DW5-CHECK-NEXT:                 DW_AT_decl_line	(1)
; DW5-CHECK-NEXT:                 DW_AT_location	(DW_OP_addrx 0x0, DW_OP_deref, DW_OP_convert (0x0000001e) "DW_ATE_signed_8", DW_OP_convert (0x00000022) "DW_ATE_signed_32", DW_OP_stack_value)
; DW5-CHECK-EMPTY:
; DW5-CHECK-NEXT: 0x0000003d:   DW_TAG_base_type
; DW5-CHECK-NEXT:                 DW_AT_name	("int")
; DW5-CHECK-NEXT:                 DW_AT_encoding	(DW_ATE_signed)
; DW5-CHECK-NEXT:                 DW_AT_byte_size	(0x04)
; DW5-CHECK-EMPTY:
; DW5-CHECK-NEXT: 0x00000041:   NULL


; DW4-CHECK: .debug_info contents:
; DW4-CHECK-NEXT: 0x00000000: Compile Unit: length = 0x00000044 version = 0x0004 abbr_offset = 0x0000 addr_size = 0x08 (next unit at 0x00000048)
; DW4-CHECK-EMPTY:
; DW4-CHECK-NEXT: 0x0000000b: DW_TAG_compile_unit
; DW4-CHECK-NEXT:               DW_AT_producer	("clang version 9.0.0 (trunk 353791) (llvm/trunk 353801)")
; DW4-CHECK-NEXT:               DW_AT_language	(DW_LANG_C99)
; DW4-CHECK-NEXT:               DW_AT_name	("dbg.c")
; DW4-CHECK-NEXT:               DW_AT_stmt_list	(0x00000000)
; DW4-CHECK-NEXT:               DW_AT_comp_dir {{.*}}
; DW4-CHECK-EMPTY:
; DW4-CHECK-NEXT: 0x0000001e:   DW_TAG_variable
; DW4-CHECK-NEXT:                 DW_AT_name	("global")
; DW4-CHECK-NEXT:                 DW_AT_type	(0x00000040 "int")
; DW4-CHECK-NEXT:                 DW_AT_external	(true)
; DW4-CHECK-NEXT:                 DW_AT_decl_file {{.*}}
; DW4-CHECK-NEXT:                 DW_AT_decl_line	(1)
; DW4-CHECK-NEXT:                 DW_AT_location	(DW_OP_addr 0x0, DW_OP_deref, DW_OP_dup, DW_OP_constu 0x7, DW_OP_shr, DW_OP_lit0, DW_OP_not, DW_OP_mul, DW_OP_constu 0x8, DW_OP_shl, DW_OP_or, DW_OP_stack_value)
; DW4-CHECK-EMPTY:
; DW4-CHECK-NEXT: 0x00000040:   DW_TAG_base_type
; DW4-CHECK-NEXT:                 DW_AT_name	("int")
; DW4-CHECK-NEXT:                 DW_AT_encoding	(DW_ATE_signed)
; DW4-CHECK-NEXT:                 DW_AT_byte_size	(0x04)
; DW4-CHECK-EMPTY:
; DW4-CHECK-NEXT: 0x00000047:   NULL

source_filename = "dbg.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@global = dso_local global i32 255, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!10}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression(DW_OP_deref, DW_OP_LLVM_convert, 8, DW_ATE_signed, DW_OP_LLVM_convert, 32, DW_ATE_signed, DW_OP_stack_value))
!1 = distinct !DIGlobalVariable(name: "global", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 9.0.0 (trunk 353791) (llvm/trunk 353801)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "dbg.c", directory: "/tmp", checksumkind: CSK_MD5, checksum: "7c731722dd7304ccb8e366ae80269b82")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 2, !"Dwarf Version", i32 5}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{!"clang version 9.0.0 (trunk 353791) (llvm/trunk 353801)"}
