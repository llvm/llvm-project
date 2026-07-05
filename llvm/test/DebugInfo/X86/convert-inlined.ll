; RUN: llc -mtriple=x86_64 -dwarf-version=5 -filetype=obj -O0 < %s | llvm-dwarfdump - \
; RUN:   | FileCheck %s --check-prefix=DW5 "--implicit-check-not={{DW_TAG|NULL}}"
; RUN: llc -mtriple=x86_64 -dwarf-version=4 -filetype=obj -O0 < %s | llvm-dwarfdump - \
; RUN:   | FileCheck %s --check-prefix=DW4 "--implicit-check-not={{DW_TAG|NULL}}"

; DW5: .debug_info contents:
; DW5: DW_TAG_compile_unit
; DW5:[[SIG8:.*]]:   DW_TAG_base_type
; DW5-NEXT:DW_AT_name ("DW_ATE_signed_8")
; DW5-NEXT:DW_AT_encoding (DW_ATE_signed)
; DW5-NEXT:DW_AT_byte_size (0x01)
; DW5-NOT: DW_AT
; DW5:[[SIG32:.*]]:   DW_TAG_base_type
; DW5-NEXT:DW_AT_name ("DW_ATE_signed_32")
; DW5-NEXT:DW_AT_encoding (DW_ATE_signed)
; DW5-NEXT:DW_AT_byte_size (0x04)
; DW5-NOT: DW_AT
; DW5:   DW_TAG_variable
; DW5:     DW_AT_name ("global")
; DW5:     DW_AT_location (DW_OP_addrx 0x0, DW_OP_deref, DW_OP_convert ([[SIG8]]) "DW_ATE_signed_8", DW_OP_convert ([[SIG32]]) "DW_ATE_signed_32", DW_OP_stack_value)
; DW5:   DW_TAG_base_type
; DW5:     DW_AT_name ("int")
; DW5:   NULL

; DW4: .debug_info contents:
; DW4: DW_TAG_compile_unit
; DW4:   DW_TAG_variable
; DW4:     DW_AT_name ("global")
; DW4:     DW_AT_location (DW_OP_addr 0x0, DW_OP_deref, DW_OP_dup, DW_OP_constu 0x7, DW_OP_shr, DW_OP_lit0, DW_OP_not, DW_OP_mul, DW_OP_constu 0x8, DW_OP_shl, DW_OP_or, DW_OP_stack_value)
; DW4:   DW_TAG_base_type
; DW4:     DW_AT_name ("int")
; DW4:   NULL

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
