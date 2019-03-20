; RUN: llc -dwarf-version=5 -filetype=obj -O0 < %s | llvm-dwarfdump - | FileCheck %s --match-full-lines --check-prefix=DW5-CHECK
; RUN: llc -dwarf-version=4 -filetype=obj -O0 < %s | llvm-dwarfdump - | FileCheck %s --match-full-lines --check-prefix=DW4-CHECK

; DW5-CHECK: .debug_info contents:
; DW5-CHECK-NEXT: 0x00000000: Compile Unit: length = 0x0000005c version = 0x0005 unit_type = DW_UT_compile abbr_offset = 0x0000 addr_size = 0x08 (next unit at 0x00000060)
; DW5-CHECK-EMPTY:
; DW5-CHECK-NEXT: 0x0000000c: DW_TAG_compile_unit
; DW5-CHECK-NEXT:               DW_AT_producer	("clang version 9.0.0 (trunk 353791) (llvm/trunk 353801)")
; DW5-CHECK-NEXT:               DW_AT_language	(DW_LANG_C99)
; DW5-CHECK-NEXT:               DW_AT_name	("dbg.c")
; DW5-CHECK-NEXT:               DW_AT_str_offsets_base	(0x00000008)
; DW5-CHECK-NEXT:               DW_AT_stmt_list	(0x00000000)
; DW5-CHECK-NEXT:               DW_AT_comp_dir {{.*}}
; DW5-CHECK-NEXT:               DW_AT_addr_base	(0x00000008)
; DW5-CHECK-NEXT:               DW_AT_low_pc	(0x0000000000000000)
; DW5-CHECK-NEXT:               DW_AT_high_pc	(0x0000000000000007)
; DW5-CHECK-NEXT:               DW_AT_loclists_base	(0x0000000c)
; DW5-CHECK-EMPTY:
; DW5-CHECK-NEXT: 0x00000027:   DW_TAG_base_type
; DW5-CHECK-NEXT:                 DW_AT_name	("DW_ATE_signed_8")
; DW5-CHECK-NEXT:                 DW_AT_encoding	(DW_ATE_signed)
; DW5-CHECK-NEXT:                 DW_AT_byte_size	(0x01)
; DW5-CHECK-EMPTY:
; DW5-CHECK-NEXT: 0x0000002b:   DW_TAG_base_type
; DW5-CHECK-NEXT:                 DW_AT_name	("DW_ATE_signed_32")
; DW5-CHECK-NEXT:                 DW_AT_encoding	(DW_ATE_signed)
; DW5-CHECK-NEXT:                 DW_AT_byte_size	(0x04)
; DW5-CHECK-EMPTY:
; DW5-CHECK-NEXT: 0x0000002f:   DW_TAG_subprogram
; DW5-CHECK-NEXT:                 DW_AT_low_pc	(0x0000000000000000)
; DW5-CHECK-NEXT:                 DW_AT_high_pc	(0x0000000000000007)
; DW5-CHECK-NEXT:                 DW_AT_frame_base	(DW_OP_reg7 RSP)
; DW5-CHECK-NEXT:                 DW_AT_name	("foo")
; DW5-CHECK-NEXT:                 DW_AT_decl_file {{.*}}
; DW5-CHECK-NEXT:                 DW_AT_decl_line	(1)
; DW5-CHECK-NEXT:                 DW_AT_prototyped	(true)
; DW5-CHECK-NEXT:                 DW_AT_type	(0x00000057 "signed char")
; DW5-CHECK-NEXT:                 DW_AT_external	(true)
; DW5-CHECK-EMPTY:
; DW5-CHECK-NEXT: 0x0000003e:     DW_TAG_formal_parameter
; DW5-CHECK-NEXT:                   DW_AT_location	(0x0000000c
; DW5-CHECK-NEXT:                      [0x0000000000000003, 0x0000000000000006): DW_OP_reg0 RAX)
; DW5-CHECK-NEXT:                   DW_AT_name	("x")
; DW5-CHECK-NEXT:                   DW_AT_decl_file {{.*}}
; DW5-CHECK-NEXT:                   DW_AT_decl_line	(1)
; DW5-CHECK-NEXT:                   DW_AT_type	(0x00000057 "signed char")
; DW5-CHECK-EMPTY:
; DW5-CHECK-NEXT: 0x0000004a:     DW_TAG_variable
; DW5-CHECK-NEXT:                   DW_AT_location	(0x00000012
; DW5-CHECK-NEXT:                      [0x0000000000000003, 0x0000000000000006): DW_OP_breg0 RAX+0, DW_OP_constu 0xff, DW_OP_and, DW_OP_convert (0x00000027) "DW_ATE_signed_8", DW_OP_convert (0x0000002b) "DW_ATE_signed_32", DW_OP_stack_value)
; DW5-CHECK-NEXT:                   DW_AT_name	("y")
; DW5-CHECK-NEXT:                   DW_AT_decl_file {{.*}}
; DW5-CHECK-NEXT:                   DW_AT_decl_line	(3)
; DW5-CHECK-NEXT:                   DW_AT_type	(0x0000005b "int")
; DW5-CHECK-EMPTY:
; DW5-CHECK-NEXT: 0x00000056:     NULL
; DW5-CHECK-EMPTY:
; DW5-CHECK-NEXT: 0x00000057:   DW_TAG_base_type
; DW5-CHECK-NEXT:                 DW_AT_name	("signed char")
; DW5-CHECK-NEXT:                 DW_AT_encoding	(DW_ATE_signed_char)
; DW5-CHECK-NEXT:                 DW_AT_byte_size	(0x01)
; DW5-CHECK-EMPTY:
; DW5-CHECK-NEXT: 0x0000005b:   DW_TAG_base_type
; DW5-CHECK-NEXT:                 DW_AT_name	("int")
; DW5-CHECK-NEXT:                 DW_AT_encoding	(DW_ATE_signed)
; DW5-CHECK-NEXT:                 DW_AT_byte_size	(0x04)
; DW5-CHECK-EMPTY:
; DW5-CHECK-NEXT: 0x0000005f:   NULL


; DW4-CHECK: .debug_info contents:
; DW4-CHECK-NEXT: 0x00000000: Compile Unit: length = 0x0000006d version = 0x0004 abbr_offset = 0x0000 addr_size = 0x08 (next unit at 0x00000071)
; DW4-CHECK-EMPTY:
; DW4-CHECK-NEXT: 0x0000000b: DW_TAG_compile_unit
; DW4-CHECK-NEXT:               DW_AT_producer	("clang version 9.0.0 (trunk 353791) (llvm/trunk 353801)")
; DW4-CHECK-NEXT:               DW_AT_language	(DW_LANG_C99)
; DW4-CHECK-NEXT:               DW_AT_name	("dbg.c")
; DW4-CHECK-NEXT:               DW_AT_stmt_list	(0x00000000)
; DW4-CHECK-NEXT:               DW_AT_comp_dir {{.*}}
; DW4-CHECK-NEXT:               DW_AT_low_pc	(0x0000000000000000)
; DW4-CHECK-NEXT:               DW_AT_high_pc	(0x0000000000000007)
; DW4-CHECK-EMPTY:
; DW4-CHECK-NEXT: 0x0000002a:   DW_TAG_subprogram
; DW4-CHECK-NEXT:                 DW_AT_low_pc	(0x0000000000000000)
; DW4-CHECK-NEXT:                 DW_AT_high_pc	(0x0000000000000007)
; DW4-CHECK-NEXT:                 DW_AT_frame_base	(DW_OP_reg7 RSP)
; DW4-CHECK-NEXT:                 DW_AT_name	("foo")
; DW4-CHECK-NEXT:                 DW_AT_decl_file {{.*}}
; DW4-CHECK-NEXT:                 DW_AT_decl_line	(1)
; DW4-CHECK-NEXT:                 DW_AT_prototyped	(true)
; DW4-CHECK-NEXT:                 DW_AT_type	(0x00000062 "signed char")
; DW4-CHECK-NEXT:                 DW_AT_external	(true)
; DW4-CHECK-EMPTY:
; DW4-CHECK-NEXT: 0x00000043:     DW_TAG_formal_parameter
; DW4-CHECK-NEXT:                   DW_AT_location	(0x00000000
; DW4-CHECK-NEXT:                      [0x0000000000000003,  0x0000000000000006): DW_OP_reg0 RAX)
; DW4-CHECK-NEXT:                   DW_AT_name	("x")
; DW4-CHECK-NEXT:                   DW_AT_decl_file {{.*}}
; DW4-CHECK-NEXT:                   DW_AT_decl_line	(1)
; DW4-CHECK-NEXT:                   DW_AT_type	(0x00000062 "signed char")
; DW4-CHECK-EMPTY:
; DW4-CHECK-NEXT: 0x00000052:     DW_TAG_variable
; DW4-CHECK-NEXT:                   DW_AT_location	(0x00000023
; DW4-CHECK-NEXT:                      [0x0000000000000003,  0x0000000000000006): DW_OP_breg0 RAX+0, DW_OP_constu 0xff, DW_OP_and, DW_OP_dup, DW_OP_constu 0x7, DW_OP_shr, DW_OP_lit0, DW_OP_not, DW_OP_mul, DW_OP_constu 0x8, DW_OP_shl, DW_OP_or, DW_OP_stack_value)
; DW4-CHECK-NEXT:                   DW_AT_name	("y")
; DW4-CHECK-NEXT:                   DW_AT_decl_file {{.*}}
; DW4-CHECK-NEXT:                   DW_AT_decl_line	(3)
; DW4-CHECK-NEXT:                   DW_AT_type	(0x00000069 "int")
; DW4-CHECK-EMPTY:
; DW4-CHECK-NEXT: 0x00000061:     NULL
; DW4-CHECK-EMPTY:
; DW4-CHECK-NEXT: 0x00000062:   DW_TAG_base_type
; DW4-CHECK-NEXT:                 DW_AT_name	("signed char")
; DW4-CHECK-NEXT:                 DW_AT_encoding	(DW_ATE_signed_char)
; DW4-CHECK-NEXT:                 DW_AT_byte_size	(0x01)
; DW4-CHECK-EMPTY:
; DW4-CHECK-NEXT: 0x00000069:   DW_TAG_base_type
; DW4-CHECK-NEXT:                 DW_AT_name	("int")
; DW4-CHECK-NEXT:                 DW_AT_encoding	(DW_ATE_signed)
; DW4-CHECK-NEXT:                 DW_AT_byte_size	(0x04)
; DW4-CHECK-EMPTY:
; DW4-CHECK-NEXT: 0x00000070:   NULL


; ModuleID = 'dbg.ll'
source_filename = "dbg.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define dso_local signext i8 @foo(i8 signext %x) !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata i8 %x, metadata !11, metadata !DIExpression()), !dbg !12
  call void @llvm.dbg.value(metadata i8 %x, metadata !13, metadata !DIExpression(DW_OP_LLVM_convert, 8, DW_ATE_signed, DW_OP_LLVM_convert, 32, DW_ATE_signed, DW_OP_stack_value)), !dbg !15
  ret i8 %x, !dbg !16
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata)

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 9.0.0 (trunk 353791) (llvm/trunk 353801)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "dbg.c", directory: "/tmp", checksumkind: CSK_MD5, checksum: "2a034da6937f5b9cf6dd2d89127f57fd")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 9.0.0 (trunk 353791) (llvm/trunk 353801)"}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !8, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10}
!10 = !DIBasicType(name: "signed char", size: 8, encoding: DW_ATE_signed_char)
!11 = !DILocalVariable(name: "x", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!12 = !DILocation(line: 1, column: 29, scope: !7)
!13 = !DILocalVariable(name: "y", scope: !7, file: !1, line: 3, type: !14)
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !DILocation(line: 3, column: 14, scope: !7)
!16 = !DILocation(line: 4, column: 3, scope: !7)
