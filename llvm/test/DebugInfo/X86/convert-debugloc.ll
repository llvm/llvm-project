; RUN: %llc_dwarf -dwarf-version=5 -filetype=obj -O0 < %s | llvm-dwarfdump - \
; RUN:   | FileCheck %s --check-prefix=DW5 "--implicit-check-not={{DW_TAG|NULL}}"
; RUN: %llc_dwarf -dwarf-version=4 -filetype=obj -O0 < %s | llvm-dwarfdump - \
; RUN:   | FileCheck %s --check-prefix=DW4 "--implicit-check-not={{DW_TAG|NULL}}"

; DW5: narf
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
; DW5:   DW_TAG_subprogram
; DW5:     DW_TAG_formal_parameter
; DW5:     DW_TAG_variable
; DW5:       DW_AT_location ({{.*}}, DW_OP_convert ([[SIG8]]) "DW_ATE_signed_8", DW_OP_convert ([[SIG32]]) "DW_ATE_signed_32", DW_OP_stack_value)
; DW5:       DW_AT_name ("y")
; DW5:     NULL
; DW5:   DW_TAG_base_type
; DW5:     DW_AT_name ("signed char")
; DW5:   DW_TAG_base_type
; DW5:     DW_AT_name ("int")
; DW5:   NULL

; DW4: .debug_info contents:
; DW4: DW_TAG_compile_unit
; DW4:   DW_TAG_subprogram
; DW4:     DW_TAG_formal_parameter
; DW4:     DW_TAG_variable
; DW4:       DW_AT_location ({{.*}}, DW_OP_dup, DW_OP_constu 0x7, DW_OP_shr, DW_OP_lit0, DW_OP_not, DW_OP_mul, DW_OP_constu 0x8, DW_OP_shl, DW_OP_or, DW_OP_stack_value)
; DW4:       DW_AT_name ("y")
; DW4:     NULL
; DW4:   DW_TAG_base_type
; DW4:     DW_AT_name ("signed char")
; DW4:   DW_TAG_base_type
; DW4:     DW_AT_name ("int")
; DW4:   NULL


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
