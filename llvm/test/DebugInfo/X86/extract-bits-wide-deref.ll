; RUN: llc -mtriple=x86_64-unknown-linux-gnu -filetype=obj %s -o %t
; RUN: llvm-dwarfdump --debug-info --debug-loclists %t | FileCheck %s --check-prefixes=COMMON,TYPED
; RUN: llvm-dwarfdump --verify %t
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -dwarf-version=4 -filetype=obj %s -o %t.dw4
; RUN: llvm-dwarfdump --debug-info --debug-loc %t.dw4 | FileCheck %s --check-prefixes=COMMON,DROP
; RUN: llvm-dwarfdump --verify %t.dw4
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -debugger-tune=lldb -filetype=obj %s -o %t.lldb
; RUN: llvm-dwarfdump --debug-info --debug-loclists %t.lldb | FileCheck %s --check-prefixes=COMMON,DROP
; RUN: llvm-dwarfdump --verify %t.lldb
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -split-dwarf-file=wide.dwo -dwarf-op-convert=Enable -filetype=obj %s -o %t.split
; RUN: llvm-dwarfdump --debug-info --debug-loclists %t.split | FileCheck %s --check-prefixes=COMMON,TYPED
; RUN: llvm-dwarfdump --verify %t.split
; RUN: llc -mtriple=i386-unknown-linux-gnu -filetype=obj %s -o %t.32
; RUN: llvm-dwarfdump --debug-info %t.32 | FileCheck %s --check-prefix=X32
; RUN: llvm-dwarfdump --verify %t.32

; Check that typed dereferences reference base types with matching byte sizes.
; TYPED: DW_TAG_base_type
; TYPED-NEXT: DW_AT_name ("DW_ATE_unsigned_80")
; TYPED-NEXT: DW_AT_encoding (DW_ATE_unsigned)
; TYPED-NEXT: DW_AT_byte_size (0x0a)
; TYPED: DW_TAG_base_type
; TYPED-NEXT: DW_AT_name ("DW_ATE_signed_72")
; TYPED-NEXT: DW_AT_encoding (DW_ATE_signed)
; TYPED-NEXT: DW_AT_byte_size (0x09)
; TYPED: DW_TAG_base_type
; TYPED-NEXT: DW_AT_name ("DW_ATE_unsigned_72")
; TYPED-NEXT: DW_AT_encoding (DW_ATE_unsigned)
; TYPED-NEXT: DW_AT_byte_size (0x09)

; COMMON-LABEL: DW_TAG_subprogram
; COMMON: DW_AT_name ("memory")
; COMMON: DW_TAG_variable
; COMMON: DW_AT_location (DW_OP_fbreg {{-?[0-9]+}}, DW_OP_deref, DW_OP_stack_value)
; COMMON: DW_AT_name ("generic")
; COMMON: DW_TAG_variable
; TYPED: DW_AT_location (DW_OP_fbreg {{-?[0-9]+}}, DW_OP_deref_type 0xa ({{.*}}) "DW_ATE_unsigned_80", DW_OP_stack_value)
; DROP-NOT: DW_AT_location
; COMMON: DW_AT_name ("wide")
; COMMON: DW_TAG_variable
; TYPED: DW_AT_location (DW_OP_fbreg {{-?[0-9]+}}, DW_OP_deref_type 0x9 ({{.*}}) "DW_ATE_signed_72", DW_OP_constu 0x4, DW_OP_convert ({{.*}}) "DW_ATE_signed_72", DW_OP_shl, DW_OP_constu 0x7, DW_OP_convert ({{.*}}) "DW_ATE_signed_72", DW_OP_shra, DW_OP_stack_value)
; DROP-NOT: DW_AT_location
; COMMON: DW_AT_name ("signed_offset")
; COMMON: DW_TAG_variable
; TYPED: DW_AT_location (DW_OP_fbreg {{-?[0-9]+}}, DW_OP_deref_type 0x9 ({{.*}}) "DW_ATE_unsigned_72", DW_OP_constu 0x7, DW_OP_convert ({{.*}}) "DW_ATE_unsigned_72", DW_OP_shl, DW_OP_constu 0x7, DW_OP_convert ({{.*}}) "DW_ATE_unsigned_72", DW_OP_shr, DW_OP_stack_value)
; DROP-NOT: DW_AT_location
; COMMON: DW_AT_name ("unsigned_partial")
; COMMON: DW_TAG_variable
; COMMON-NOT: DW_AT_location
; COMMON: DW_AT_name ("too_wide")
; COMMON: DW_TAG_variable
; COMMON-NOT: DW_AT_location
; COMMON: DW_AT_name ("offset_too_wide")
; COMMON: DW_TAG_variable
; COMMON-NOT: DW_AT_location
; COMMON: DW_AT_name ("offset_overflow")
; COMMON: DW_TAG_variable
; COMMON-NOT: DW_AT_location
; COMMON: DW_AT_name ("arithmetic")
; COMMON: DW_TAG_variable
; TYPED: DW_AT_location (DW_OP_fbreg {{-?[0-9]+}}, DW_OP_deref_type 0xa ({{.*}}) "DW_ATE_unsigned_80", DW_OP_stack_value, DW_OP_piece 0xa)
; TYPED: DW_AT_LLVM_tag_offset (0x01)
; DROP-NOT: DW_AT_location
; COMMON: DW_AT_name ("fragment")

; An eight-byte read is already wider than the generic type on i386.
; X32: DW_AT_location (DW_OP_fbreg {{[+-]?[0-9]+}}, DW_OP_deref_type 0x8 ({{.*}}) "DW_ATE_unsigned_64", DW_OP_stack_value)
; X32: DW_AT_name ("generic")

@global_too_wide = global [256 x i8] zeroinitializer, !dbg !40

define void @memory() !dbg !10 {
  %storage = alloca [256 x i8], align 8
  call void @llvm.dbg.declare(metadata ptr %storage, metadata !11, metadata !DIExpression(DW_OP_LLVM_extract_bits_zext, 0, 64)), !dbg !18
  call void @llvm.dbg.declare(metadata ptr %storage, metadata !12, metadata !DIExpression(DW_OP_LLVM_extract_bits_zext, 0, 80)), !dbg !18
  call void @llvm.dbg.declare(metadata ptr %storage, metadata !13, metadata !DIExpression(DW_OP_LLVM_extract_bits_sext, 3, 65)), !dbg !18
  call void @llvm.dbg.declare(metadata ptr %storage, metadata !14, metadata !DIExpression(DW_OP_LLVM_extract_bits_zext, 0, 65)), !dbg !18
  call void @llvm.dbg.declare(metadata ptr %storage, metadata !15, metadata !DIExpression(DW_OP_LLVM_extract_bits_zext, 0, 2048)), !dbg !18
  call void @llvm.dbg.declare(metadata ptr %storage, metadata !16, metadata !DIExpression(DW_OP_LLVM_extract_bits_zext, 2040, 1)), !dbg !18
  call void @llvm.dbg.declare(metadata ptr %storage, metadata !24, metadata !DIExpression(DW_OP_LLVM_extract_bits_zext, 18446744073709551615, 1)), !dbg !18
  call void @llvm.dbg.declare(metadata ptr %storage, metadata !17, metadata !DIExpression(DW_OP_LLVM_extract_bits_zext, 0, 80, DW_OP_plus_uconst, 1)), !dbg !18
  call void @llvm.dbg.declare(metadata ptr %storage, metadata !19, metadata !DIExpression(DW_OP_LLVM_extract_bits_zext, 0, 80, DW_OP_LLVM_tag_offset, 1, DW_OP_stack_value, DW_OP_LLVM_fragment, 0, 80)), !dbg !18
  ret void, !dbg !18
}

; Location lists must resolve the type reference as well, including in split
; DWARF. Unsupported entries must be discarded, not kept as partial addresses.
; COMMON-LABEL: DW_TAG_subprogram
; COMMON: DW_AT_name ("locations")
; COMMON: DW_TAG_variable
; TYPED: DW_AT_location
; TYPED: DW_OP_deref_type 0xa ({{.*}}) "DW_ATE_unsigned_80", DW_OP_stack_value
; DROP-NOT: DW_AT_location
; COMMON: DW_AT_name ("list_wide")
; COMMON: DW_TAG_variable
; COMMON-NOT: DW_AT_location
; COMMON: DW_AT_name ("list_too_wide")
; COMMON: DW_TAG_variable
; Rejecting the wide fragment must also discard the earlier generic fragment.
; TYPED: DW_AT_location
; TYPED: DW_OP_deref_size 0x4, DW_OP_stack_value, DW_OP_piece 0x4, {{.*}}DW_OP_deref_type 0xa ({{.*}}) "DW_ATE_unsigned_80", DW_OP_stack_value, DW_OP_piece 0xa
; DROP-NOT: DW_AT_location
; COMMON: DW_AT_name ("list_fragments")
; COMMON: DW_TAG_variable
; COMMON: DW_AT_name ("global_too_wide")
; COMMON-NOT: DW_AT_location
; COMMON: DW_TAG_base_type
; TYPED: .debug_loclists
; TYPED: DW_OP_deref_type 0xa
; DROP-NOT: DW_OP_deref_size 0xa

define void @locations(ptr %p, ptr %q) !dbg !30 {
  call void @clobber(), !dbg !33
  call void @llvm.dbg.declare(metadata ptr %p, metadata !31, metadata !DIExpression(DW_OP_LLVM_extract_bits_zext, 0, 80)), !dbg !33
  call void @llvm.dbg.declare(metadata ptr %p, metadata !32, metadata !DIExpression(DW_OP_LLVM_extract_bits_zext, 0, 2048)), !dbg !33
  call void @llvm.dbg.declare(metadata ptr %p, metadata !34, metadata !DIExpression(DW_OP_LLVM_extract_bits_zext, 0, 32, DW_OP_LLVM_fragment, 0, 32)), !dbg !33
  call void @llvm.dbg.declare(metadata ptr %p, metadata !34, metadata !DIExpression(DW_OP_LLVM_extract_bits_zext, 0, 80, DW_OP_LLVM_fragment, 32, 80)), !dbg !33
  call void @clobber(), !dbg !33
  call void @llvm.dbg.declare(metadata ptr %q, metadata !31, metadata !DIExpression(DW_OP_LLVM_extract_bits_zext, 0, 80)), !dbg !33
  call void @llvm.dbg.declare(metadata ptr %q, metadata !32, metadata !DIExpression(DW_OP_LLVM_extract_bits_zext, 0, 2048)), !dbg !33
  call void @llvm.dbg.declare(metadata ptr %q, metadata !34, metadata !DIExpression(DW_OP_LLVM_extract_bits_zext, 0, 32, DW_OP_LLVM_fragment, 0, 32)), !dbg !33
  call void @llvm.dbg.declare(metadata ptr %q, metadata !34, metadata !DIExpression(DW_OP_LLVM_extract_bits_zext, 0, 80, DW_OP_LLVM_fragment, 32, 80)), !dbg !33
  call void @clobber(), !dbg !33
  ret void, !dbg !33
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare void @clobber()

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !42, nameTableKind: None)
!1 = !DIFile(filename: "wide.cpp", directory: "/")
!2 = !DISubroutineType(types: !{})
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !DIBasicType(name: "uint64_t", size: 64, encoding: DW_ATE_unsigned)
!6 = !DIBasicType(name: "uint80_t", size: 80, encoding: DW_ATE_unsigned)
!7 = !DIBasicType(name: "int128_t", size: 128, encoding: DW_ATE_signed)
!8 = !DIBasicType(name: "uint2048_t", size: 2048, encoding: DW_ATE_unsigned)
!9 = !DIBasicType(name: "uint128_t", size: 128, encoding: DW_ATE_unsigned)
!10 = distinct !DISubprogram(name: "memory", scope: !1, file: !1, line: 1, type: !2, spFlags: DISPFlagDefinition, unit: !0)
!11 = !DILocalVariable(name: "generic", scope: !10, file: !1, line: 1, type: !5)
!12 = !DILocalVariable(name: "wide", scope: !10, file: !1, line: 1, type: !6)
!13 = !DILocalVariable(name: "signed_offset", scope: !10, file: !1, line: 1, type: !7)
!14 = !DILocalVariable(name: "unsigned_partial", scope: !10, file: !1, line: 1, type: !9)
!15 = !DILocalVariable(name: "too_wide", scope: !10, file: !1, line: 1, type: !8)
!16 = !DILocalVariable(name: "offset_too_wide", scope: !10, file: !1, line: 1, type: !5)
!17 = !DILocalVariable(name: "arithmetic", scope: !10, file: !1, line: 1, type: !6)
!18 = !DILocation(line: 1, column: 1, scope: !10)
!19 = !DILocalVariable(name: "fragment", scope: !10, file: !1, line: 1, type: !9)
!24 = !DILocalVariable(name: "offset_overflow", scope: !10, file: !1, line: 1, type: !5)
!30 = distinct !DISubprogram(name: "locations", scope: !1, file: !1, line: 3, type: !2, spFlags: DISPFlagDefinition, unit: !0)
!31 = !DILocalVariable(name: "list_wide", scope: !30, file: !1, line: 3, type: !6)
!32 = !DILocalVariable(name: "list_too_wide", scope: !30, file: !1, line: 3, type: !8)
!33 = !DILocation(line: 3, column: 1, scope: !30)
!34 = !DILocalVariable(name: "list_fragments", scope: !30, file: !1, line: 3, type: !9)
!40 = !DIGlobalVariableExpression(var: !41, expr: !DIExpression(DW_OP_LLVM_extract_bits_zext, 0, 2048))
!41 = distinct !DIGlobalVariable(name: "global_too_wide", scope: !0, file: !1, line: 4, type: !8, isLocal: false, isDefinition: true)
!42 = !{!40}
