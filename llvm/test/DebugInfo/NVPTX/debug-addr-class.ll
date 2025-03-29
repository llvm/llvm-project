; RUN: llc -mtriple=nvptx64-nvidia-cuda < %s | FileCheck %s
; RUN: %if ptxas %{ llc -mtriple=nvptx64-nvidia-cuda < %s | %ptxas-verify %}

@GLOBAL = addrspace(1) externally_initialized global i32 0, align 4, !dbg !0
@SHARED = addrspace(3) externally_initialized global i32 undef, align 4, !dbg !6

define ptx_kernel void @test(float, ptr, ptr, i32) !dbg !17 {
  %5 = alloca float, align 4
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca i32, align 4
  store float %0, ptr %5, align 4
  call void @llvm.dbg.declare(metadata ptr %5, metadata !22, metadata !DIExpression()), !dbg !23
  store ptr %1, ptr %6, align 8
  call void @llvm.dbg.declare(metadata ptr %6, metadata !24, metadata !DIExpression()), !dbg !25
  store ptr %2, ptr %7, align 8
  call void @llvm.dbg.declare(metadata ptr %7, metadata !26, metadata !DIExpression()), !dbg !27
  store i32 %3, ptr %8, align 4
  call void @llvm.dbg.declare(metadata ptr %8, metadata !28, metadata !DIExpression()), !dbg !29
  %9 = load float, ptr %5, align 4, !dbg !30
  %10 = load ptr, ptr %6, align 8, !dbg !31
  %11 = load i32, ptr %8, align 4, !dbg !32
  %12 = sext i32 %11 to i64, !dbg !31
  %13 = getelementptr inbounds float, ptr %10, i64 %12, !dbg !31
  %14 = load float, ptr %13, align 4, !dbg !31
  %15 = fmul contract float %9, %14, !dbg !33
  %16 = load ptr, ptr %7, align 8, !dbg !34
  %17 = load i32, ptr %8, align 4, !dbg !35
  %18 = sext i32 %17 to i64, !dbg !34
  %19 = getelementptr inbounds float, ptr %16, i64 %18, !dbg !34
  store float %15, ptr %19, align 4, !dbg !36
  store i32 0, ptr addrspacecast (ptr addrspace(1) @GLOBAL to ptr), align 4, !dbg !37
  store i32 0, ptr addrspacecast (ptr addrspace(3) @SHARED to ptr), align 4, !dbg !38
  ret void, !dbg !39
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!11, !12, !13, !14, !15}
!llvm.ident = !{!16}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "GLOBAL", scope: !2, file: !8, line: 3, type: !9, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 9.0.0 (trunk 351969) (llvm/trunk 351973)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "new.cc", directory: "/tmp")
!4 = !{}
!5 = !{!0, !6}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression(DW_OP_constu, 8, DW_OP_swap, DW_OP_xderef))
!7 = distinct !DIGlobalVariable(name: "SHARED", scope: !2, file: !8, line: 4, type: !9, isLocal: false, isDefinition: true)
!8 = !DIFile(filename: "test.cu", directory: "/tmp")
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{i32 2, !"Dwarf Version", i32 2}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 1, !"wchar_size", i32 4}
!14 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!15 = !{i32 7, !"PIC Level", i32 2}
!16 = !{!"clang version 9.0.0 (trunk 351969) (llvm/trunk 351973)"}
!17 = distinct !DISubprogram(name: "test", linkageName: "test", scope: !8, file: !8, line: 6, type: !18, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !40)
!18 = !DISubroutineType(types: !19)
!19 = !{null, !20, !21, !21, !9}
!20 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !20, size: 64)
!22 = !DILocalVariable(name: "a", arg: 1, scope: !17, file: !8, line: 6, type: !20)
!23 = !DILocation(line: 6, column: 41, scope: !17)
!24 = !DILocalVariable(name: "x", arg: 2, scope: !17, file: !8, line: 6, type: !21)
!25 = !DILocation(line: 6, column: 51, scope: !17)
!26 = !DILocalVariable(name: "y", arg: 3, scope: !17, file: !8, line: 6, type: !21)
!27 = !DILocation(line: 6, column: 61, scope: !17)
!28 = !DILocalVariable(name: "i", arg: 4, scope: !17, file: !8, line: 6, type: !9)
!29 = !DILocation(line: 6, column: 68, scope: !17)
!30 = !DILocation(line: 7, column: 10, scope: !17)
!31 = !DILocation(line: 7, column: 14, scope: !17)
!32 = !DILocation(line: 7, column: 16, scope: !17)
!33 = !DILocation(line: 7, column: 12, scope: !17)
!34 = !DILocation(line: 7, column: 3, scope: !17)
!35 = !DILocation(line: 7, column: 5, scope: !17)
!36 = !DILocation(line: 7, column: 8, scope: !17)
!37 = !DILocation(line: 8, column: 10, scope: !17)
!38 = !DILocation(line: 9, column: 10, scope: !17)
!39 = !DILocation(line: 10, column: 1, scope: !17)
!40 = !{!22, !24, !26, !28}

; CHECK: 	.section	.debug_abbrev
; CHECK-NEXT: 	{
; CHECK-NEXT: .b8 1                                   // Abbreviation Code
; CHECK-NEXT: .b8 17                                  // DW_TAG_compile_unit
; CHECK-NEXT: .b8 1                                   // DW_CHILDREN_yes
; CHECK-NEXT: .b8 37                                  // DW_AT_producer
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 19                                  // DW_AT_language
; CHECK-NEXT: .b8 5                                   // DW_FORM_data2
; CHECK-NEXT: .b8 3                                   // DW_AT_name
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 16                                  // DW_AT_stmt_list
; CHECK-NEXT: .b8 6                                   // DW_FORM_data4
; CHECK-NEXT: .b8 27                                  // DW_AT_comp_dir
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 2                                   // Abbreviation Code
; CHECK-NEXT: .b8 52                                  // DW_TAG_variable
; CHECK-NEXT: .b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT: .b8 3                                   // DW_AT_name
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 73                                  // DW_AT_type
; CHECK-NEXT: .b8 19                                  // DW_FORM_ref4
; CHECK-NEXT: .b8 63                                  // DW_AT_external
; CHECK-NEXT: .b8 12                                  // DW_FORM_flag
; CHECK-NEXT: .b8 58                                  // DW_AT_decl_file
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 59                                  // DW_AT_decl_line
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 51                                  // DW_AT_address_class
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 2                                   // DW_AT_location
; CHECK-NEXT: .b8 10                                  // DW_FORM_block1
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 3                                   // Abbreviation Code
; CHECK-NEXT: .b8 36                                  // DW_TAG_base_type
; CHECK-NEXT: .b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT: .b8 3                                   // DW_AT_name
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 62                                  // DW_AT_encoding
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 11                                  // DW_AT_byte_size
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 4                                   // Abbreviation Code
; CHECK-NEXT: .b8 46                                  // DW_TAG_subprogram
; CHECK-NEXT: .b8 1                                   // DW_CHILDREN_yes
; CHECK-NEXT: .b8 17                                  // DW_AT_low_pc
; CHECK-NEXT: .b8 1                                   // DW_FORM_addr
; CHECK-NEXT: .b8 18                                  // DW_AT_high_pc
; CHECK-NEXT: .b8 1                                   // DW_FORM_addr
; CHECK-NEXT: .b8 64                                  // DW_AT_frame_base
; CHECK-NEXT: .b8 10                                  // DW_FORM_block1
; CHECK-NEXT: .b8 135                                 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 64
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 3                                   // DW_AT_name
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 58                                  // DW_AT_decl_file
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 59                                  // DW_AT_decl_line
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 63                                  // DW_AT_external
; CHECK-NEXT: .b8 12                                  // DW_FORM_flag
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 5                                   // Abbreviation Code
; CHECK-NEXT: .b8 5                                   // DW_TAG_formal_parameter
; CHECK-NEXT: .b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT: .b8 51                                  // DW_AT_address_class
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 2                                   // DW_AT_location
; CHECK-NEXT: .b8 10                                  // DW_FORM_block1
; CHECK-NEXT: .b8 3                                   // DW_AT_name
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 58                                  // DW_AT_decl_file
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 59                                  // DW_AT_decl_line
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 73                                  // DW_AT_type
; CHECK-NEXT: .b8 19                                  // DW_FORM_ref4
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 6                                   // Abbreviation Code
; CHECK-NEXT: .b8 5                                   // DW_TAG_formal_parameter
; CHECK-NEXT: .b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT: .b8 3                                   // DW_AT_name
; CHECK-NEXT: .b8 8                                   // DW_FORM_string
; CHECK-NEXT: .b8 58                                  // DW_AT_decl_file
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 59                                  // DW_AT_decl_line
; CHECK-NEXT: .b8 11                                  // DW_FORM_data1
; CHECK-NEXT: .b8 73                                  // DW_AT_type
; CHECK-NEXT: .b8 19                                  // DW_FORM_ref4
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 7                                   // Abbreviation Code
; CHECK-NEXT: .b8 15                                  // DW_TAG_pointer_type
; CHECK-NEXT: .b8 0                                   // DW_CHILDREN_no
; CHECK-NEXT: .b8 73                                  // DW_AT_type
; CHECK-NEXT: .b8 19                                  // DW_FORM_ref4
; CHECK-NEXT: .b8 0                                   // EOM(1)
; CHECK-NEXT: .b8 0                                   // EOM(2)
; CHECK-NEXT: .b8 0                                   // EOM(3)
; CHECK-NEXT: 	}
; CHECK-NEXT: 	.section	.debug_info
; CHECK-NEXT: 	{
; CHECK-NEXT: .b32 238                                // Length of Unit
; CHECK-NEXT: .b8 2                                   // DWARF version number
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 .debug_abbrev                      // Offset Into Abbrev. Section
; CHECK-NEXT: .b8 8                                   // Address Size (in bytes)
; CHECK-NEXT: .b8 1                                   // Abbrev [1] 0xb:0xe7 DW_TAG_compile_unit
; CHECK-NEXT: .b8 99                                  // DW_AT_producer
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 103
; CHECK-NEXT: .b8 32
; CHECK-NEXT: .b8 118
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 115
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 32
; CHECK-NEXT: .b8 57
; CHECK-NEXT: .b8 46
; CHECK-NEXT: .b8 48
; CHECK-NEXT: .b8 46
; CHECK-NEXT: .b8 48
; CHECK-NEXT: .b8 32
; CHECK-NEXT: .b8 40
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 107
; CHECK-NEXT: .b8 32
; CHECK-NEXT: .b8 51
; CHECK-NEXT: .b8 53
; CHECK-NEXT: .b8 49
; CHECK-NEXT: .b8 57
; CHECK-NEXT: .b8 54
; CHECK-NEXT: .b8 57
; CHECK-NEXT: .b8 41
; CHECK-NEXT: .b8 32
; CHECK-NEXT: .b8 40
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 118
; CHECK-NEXT: .b8 109
; CHECK-NEXT: .b8 47
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 107
; CHECK-NEXT: .b8 32
; CHECK-NEXT: .b8 51
; CHECK-NEXT: .b8 53
; CHECK-NEXT: .b8 49
; CHECK-NEXT: .b8 57
; CHECK-NEXT: .b8 55
; CHECK-NEXT: .b8 51
; CHECK-NEXT: .b8 41
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                   // DW_AT_language
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 110                                 // DW_AT_name
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 119
; CHECK-NEXT: .b8 46
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 .debug_line                        // DW_AT_stmt_list
; CHECK-NEXT: .b8 47                                  // DW_AT_comp_dir
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 109
; CHECK-NEXT: .b8 112
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                   // Abbrev [2] 0x55:0x1a DW_TAG_variable
; CHECK-NEXT: .b8 71                                  // DW_AT_name
; CHECK-NEXT: .b8 76
; CHECK-NEXT: .b8 79
; CHECK-NEXT: .b8 66
; CHECK-NEXT: .b8 65
; CHECK-NEXT: .b8 76
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 111                                // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_external
; CHECK-NEXT: .b8 1                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 3                                   // DW_AT_decl_line
; CHECK-NEXT: .b8 5                                   // DW_AT_address_class
; CHECK-NEXT: .b8 9                                   // DW_AT_location
; CHECK-NEXT: .b8 3
; CHECK-NEXT: .b64 GLOBAL
; CHECK-NEXT: .b8 3                                   // Abbrev [3] 0x6f:0x7 DW_TAG_base_type
; CHECK-NEXT: .b8 105                                 // DW_AT_name
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 5                                   // DW_AT_encoding
; CHECK-NEXT: .b8 4                                   // DW_AT_byte_size
; CHECK-NEXT: .b8 2                                   // Abbrev [2] 0x76:0x1a DW_TAG_variable
; CHECK-NEXT: .b8 83                                  // DW_AT_name
; CHECK-NEXT: .b8 72
; CHECK-NEXT: .b8 65
; CHECK-NEXT: .b8 82
; CHECK-NEXT: .b8 69
; CHECK-NEXT: .b8 68
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 111                                // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_external
; CHECK-NEXT: .b8 1                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 4                                   // DW_AT_decl_line
; CHECK-NEXT: .b8 8                                   // DW_AT_address_class
; CHECK-NEXT: .b8 9                                   // DW_AT_location
; CHECK-NEXT: .b8 3
; CHECK-NEXT: .b64 SHARED
; CHECK-NEXT: .b8 4                                   // Abbrev [4] 0x90:0x53 DW_TAG_subprogram
; CHECK-NEXT: .b64 $L__func_begin0                    // DW_AT_low_pc
; CHECK-NEXT: .b64 $L__func_end0                      // DW_AT_high_pc
; CHECK-NEXT: .b8 1                                   // DW_AT_frame_base
; CHECK-NEXT: .b8 156
; CHECK-NEXT: .b8 116                                 // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 115
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 116                                 // DW_AT_name
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 115
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 6                                   // DW_AT_decl_line
; CHECK-NEXT: .b8 1                                   // DW_AT_external
; CHECK-NEXT: .b8 5                                   // Abbrev [5] 0xb0:0x10 DW_TAG_formal_parameter
; CHECK-NEXT: .b8 2                                   // DW_AT_address_class
; CHECK-NEXT: .b8 5                                   // DW_AT_location
; CHECK-NEXT: .b8 144
; CHECK-NEXT: .b8 177
; CHECK-NEXT: .b8 204
; CHECK-NEXT: .b8 149
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b8 97                                  // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 6                                   // DW_AT_decl_line
; CHECK-NEXT: .b32 227                                // DW_AT_type
; CHECK-NEXT: .b8 6                                   // Abbrev [6] 0xc0:0x9 DW_TAG_formal_parameter
; CHECK-NEXT: .b8 120                                 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 6                                   // DW_AT_decl_line
; CHECK-NEXT: .b32 236                                // DW_AT_type
; CHECK-NEXT: .b8 6                                   // Abbrev [6] 0xc9:0x9 DW_TAG_formal_parameter
; CHECK-NEXT: .b8 121                                 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 6                                   // DW_AT_decl_line
; CHECK-NEXT: .b32 236                                // DW_AT_type
; CHECK-NEXT: .b8 5                                   // Abbrev [5] 0xd2:0x10 DW_TAG_formal_parameter
; CHECK-NEXT: .b8 2                                   // DW_AT_address_class
; CHECK-NEXT: .b8 5                                   // DW_AT_location
; CHECK-NEXT: .b8 144
; CHECK-NEXT: .b8 177
; CHECK-NEXT: .b8 228
; CHECK-NEXT: .b8 149
; CHECK-NEXT: .b8 1
; CHECK-NEXT: .b8 105                                 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 6                                   // DW_AT_decl_line
; CHECK-NEXT: .b32 111                                // DW_AT_type
; CHECK-NEXT: .b8 0                                   // End Of Children Mark
; CHECK-NEXT: .b8 3                                   // Abbrev [3] 0xe3:0x9 DW_TAG_base_type
; CHECK-NEXT: .b8 102                                 // DW_AT_name
; CHECK-NEXT: .b8 108
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                   // DW_AT_encoding
; CHECK-NEXT: .b8 4                                   // DW_AT_byte_size
; CHECK-NEXT: .b8 7                                   // Abbrev [7] 0xec:0x5 DW_TAG_pointer_type
; CHECK-NEXT: .b32 227                                // DW_AT_type
; CHECK-NEXT: .b8 0                                   // End Of Children Mark
; CHECK-NEXT: 	}
; CHECK-NEXT: 	.section	.debug_macinfo	{	}
; CHECK-NOT: debug_
