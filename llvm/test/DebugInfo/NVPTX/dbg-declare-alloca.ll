; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64-nvidia-cuda | %ptxas-verify %}

; CHECK: .target sm_20, debug

; CHECK: .visible .func use_dbg_declare()
; CHECK: .local .align 8 .b8 __local_depot0[8];
; CHECK: mov.u64 %SPL, __local_depot0;
; CHECK: add.u64 %rd1, %SP, 0;
; CHECK: .loc 1 5 3                   // t.c:5:3
; CHECK: { // callseq 0, 0
; CHECK: .reg .b32 temp_param_reg;
; CHECK: .param .b64 param0;
; CHECK: st.param.b64 [param0+0], %rd1;
; CHECK: call.uni
; CHECK: escape_foo,
; CHECK: (
; CHECK: param0
; CHECK: );
; CHECK: } // callseq 0
; CHECK: .loc 1 6 1                   // t.c:6:1
; CHECK: ret;
; CHECK: }

; CHECK: .file 1 "test{{(/|\\\\)}}t.c"

; CHECK: .section .debug_abbrev
; CHECK-NEXT: {
; CHECK-NEXT: .b8 1                                // Abbreviation Code
; CHECK-NEXT: .b8 17                               // DW_TAG_compile_unit
; CHECK-NEXT: .b8 1                                // DW_CHILDREN_yes
; CHECK-NEXT: .b8 37                               // DW_AT_producer
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 19                               // DW_AT_language
; CHECK-NEXT: .b8 5                                // DW_FORM_data2
; CHECK-NEXT: .b8 3                                // DW_AT_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 16                               // DW_AT_stmt_list
; CHECK-NEXT: .b8 6                                // DW_FORM_data4
; CHECK-NEXT: .b8 27                               // DW_AT_comp_dir
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 17                               // DW_AT_low_pc
; CHECK-NEXT: .b8 1                                // DW_FORM_addr
; CHECK-NEXT: .b8 18                               // DW_AT_high_pc
; CHECK-NEXT: .b8 1                                // DW_FORM_addr
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 2                                // Abbreviation Code
; CHECK-NEXT: .b8 46                               // DW_TAG_subprogram
; CHECK-NEXT: .b8 1                                // DW_CHILDREN_yes
; CHECK-NEXT: .b8 17                               // DW_AT_low_pc
; CHECK-NEXT: .b8 1                                // DW_FORM_addr
; CHECK-NEXT: .b8 18                               // DW_AT_high_pc
; CHECK-NEXT: .b8 1                                // DW_FORM_addr
; CHECK-NEXT: .b8 64                               // DW_AT_frame_base
; CHECK-NEXT: .b8 10                               // DW_FORM_block1
; CHECK-NEXT: .b8 3                                // DW_AT_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 58                               // DW_AT_decl_file
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 59                               // DW_AT_decl_line
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 39                               // DW_AT_prototyped
; CHECK-NEXT: .b8 12                               // DW_FORM_flag
; CHECK-NEXT: .b8 63                               // DW_AT_external
; CHECK-NEXT: .b8 12                               // DW_FORM_flag
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 3                                // Abbreviation Code
; CHECK-NEXT: .b8 52                               // DW_TAG_variable
; CHECK-NEXT: .b8 0                                // DW_CHILDREN_no
; CHECK-NEXT: .b8 51                               // DW_AT_address_class
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 2                                // DW_AT_location
; CHECK-NEXT: .b8 10                               // DW_FORM_block1
; CHECK-NEXT: .b8 3                                // DW_AT_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 58                               // DW_AT_decl_file
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 59                               // DW_AT_decl_line
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 73                               // DW_AT_type
; CHECK-NEXT: .b8 19                               // DW_FORM_ref4
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 4                                // Abbreviation Code
; CHECK-NEXT: .b8 19                               // DW_TAG_structure_type
; CHECK-NEXT: .b8 1                                // DW_CHILDREN_yes
; CHECK-NEXT: .b8 3                                // DW_AT_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 11                               // DW_AT_byte_size
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 58                               // DW_AT_decl_file
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 59                               // DW_AT_decl_line
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 5                                // Abbreviation Code
; CHECK-NEXT: .b8 13                               // DW_TAG_member
; CHECK-NEXT: .b8 0                                // DW_CHILDREN_no
; CHECK-NEXT: .b8 3                                // DW_AT_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 73                               // DW_AT_type
; CHECK-NEXT: .b8 19                               // DW_FORM_ref4
; CHECK-NEXT: .b8 58                               // DW_AT_decl_file
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 59                               // DW_AT_decl_line
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 56                               // DW_AT_data_member_location
; CHECK-NEXT: .b8 10                               // DW_FORM_block1
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 6                                // Abbreviation Code
; CHECK-NEXT: .b8 36                               // DW_TAG_base_type
; CHECK-NEXT: .b8 0                                // DW_CHILDREN_no
; CHECK-NEXT: .b8 3                                // DW_AT_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 62                               // DW_AT_encoding
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 11                               // DW_AT_byte_size
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 0                                // EOM(3)
; CHECK-NEXT: }
; CHECK-NEXT: .section .debug_info
; CHECK-NEXT: {
; CHECK-NEXT: .b32 134                             // Length of Unit
; CHECK-NEXT: .b8 2                                // DWARF version number
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 .debug_abbrev                   // Offset Into Abbrev. Section
; CHECK-NEXT: .b8 8                                // Address Size (in bytes)
; CHECK-NEXT: .b8 1                                // Abbrev [1] 0xb:0x7f DW_TAG_compile_unit
; CHECK-NEXT: .b8 99                               // DW_AT_producer
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 103
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 12                               // DW_AT_language
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 116                              // DW_AT_name
; CHECK-NEXT: .b8 46
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 .debug_line                     // DW_AT_stmt_list
; CHECK-NEXT: .b8 116                              // DW_AT_comp_dir
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 115
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b64 $L__func_begin0                 // DW_AT_low_pc
; CHECK-NEXT: .b64 $L__func_end0                   // DW_AT_high_pc
; CHECK-NEXT: .b8 2                                // Abbrev [2] 0x31:0x3c DW_TAG_subprogram
; CHECK-NEXT: .b64 $L__func_begin0                 // DW_AT_low_pc
; CHECK-NEXT: .b64 $L__func_end0                   // DW_AT_high_pc
; CHECK-NEXT: .b8 1                                // DW_AT_frame_base
; CHECK-NEXT: .b8 156
; CHECK-NEXT: .b8 117                              // DW_AT_name
; CHECK-NEXT: .b8 115
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 98
; CHECK-NEXT: .b8 103
; CHECK-NEXT: .b8 95
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 3                                // DW_AT_decl_line
; CHECK-NEXT: .b8 1                                // DW_AT_prototyped
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x58:0x14 DW_TAG_variable
; CHECK-NEXT: .b8 6                                // DW_AT_address_class
; CHECK-NEXT: .b8 9                                // DW_AT_location
; CHECK-NEXT: .b8 3
; CHECK-NEXT: .b64 __local_depot0
; CHECK-NEXT: .b8 111                              // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 4                                // DW_AT_decl_line
; CHECK-NEXT: .b32 109                             // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x6d:0x15 DW_TAG_structure_type
; CHECK-NEXT: .b8 70                               // DW_AT_name
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_byte_size
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 1                                // DW_AT_decl_line
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0x75:0xc DW_TAG_member
; CHECK-NEXT: .b8 120                              // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 130                             // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 1                                // DW_AT_decl_line
; CHECK-NEXT: .b8 2                                // DW_AT_data_member_location
; CHECK-NEXT: .b8 35
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x82:0x7 DW_TAG_base_type
; CHECK-NEXT: .b8 105                              // DW_AT_name
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 5                                // DW_AT_encoding
; CHECK-NEXT: .b8 4                                // DW_AT_byte_size
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: }

%struct.Foo = type { i32 }

; Function Attrs: noinline nounwind uwtable
define void @use_dbg_declare() #0 !dbg !7 {
entry:
  %o = alloca %struct.Foo, align 4
  call void @llvm.dbg.declare(metadata %struct.Foo* %o, metadata !10, metadata !15), !dbg !16
  call void @escape_foo(%struct.Foo* %o), !dbg !17
  ret void, !dbg !18
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @escape_foo(%struct.Foo*)

attributes #0 = { noinline nounwind uwtable }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "t.c", directory: "test")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 2}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang"}
!7 = distinct !DISubprogram(name: "use_dbg_declare", scope: !1, file: !1, line: 3, type: !8, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocalVariable(name: "o", scope: !7, file: !1, line: 4, type: !11)
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo", file: !1, line: 1, size: 32, elements: !12)
!12 = !{!13}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !11, file: !1, line: 1, baseType: !14, size: 32)
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !DIExpression()
!16 = !DILocation(line: 4, column: 14, scope: !7)
!17 = !DILocation(line: 5, column: 3, scope: !7)
!18 = !DILocation(line: 6, column: 1, scope: !7)
