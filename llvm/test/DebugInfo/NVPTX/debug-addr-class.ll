; RUN: llc -mtriple=nvptx64-nvidia-cuda < %s | FileCheck %s

@GLOBAL = addrspace(1) externally_initialized global i32 0, align 4, !dbg !0
@SHARED = addrspace(3) externally_initialized global i32 undef, align 4, !dbg !6

define void @test(float, float*, float*, i32) !dbg !17 {
  %5 = alloca float, align 4
  %6 = alloca float*, align 8
  %7 = alloca float*, align 8
  %8 = alloca i32, align 4
  store float %0, float* %5, align 4
  call void @llvm.dbg.declare(metadata float* %5, metadata !22, metadata !DIExpression()), !dbg !23
  store float* %1, float** %6, align 8
  call void @llvm.dbg.declare(metadata float** %6, metadata !24, metadata !DIExpression()), !dbg !25
  store float* %2, float** %7, align 8
  call void @llvm.dbg.declare(metadata float** %7, metadata !26, metadata !DIExpression()), !dbg !27
  store i32 %3, i32* %8, align 4
  call void @llvm.dbg.declare(metadata i32* %8, metadata !28, metadata !DIExpression()), !dbg !29
  %9 = load float, float* %5, align 4, !dbg !30
  %10 = load float*, float** %6, align 8, !dbg !31
  %11 = load i32, i32* %8, align 4, !dbg !32
  %12 = sext i32 %11 to i64, !dbg !31
  %13 = getelementptr inbounds float, float* %10, i64 %12, !dbg !31
  %14 = load float, float* %13, align 4, !dbg !31
  %15 = fmul contract float %9, %14, !dbg !33
  %16 = load float*, float** %7, align 8, !dbg !34
  %17 = load i32, i32* %8, align 4, !dbg !35
  %18 = sext i32 %17 to i64, !dbg !34
  %19 = getelementptr inbounds float, float* %16, i64 %18, !dbg !34
  store float %15, float* %19, align 4, !dbg !36
  store i32 0, i32* addrspacecast (i32 addrspace(1)* @GLOBAL to i32*), align 4, !dbg !37
  store i32 0, i32* addrspacecast (i32 addrspace(3)* @SHARED to i32*), align 4, !dbg !38
  ret void, !dbg !39
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!2}
!nvvm.annotations = !{!10}
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
!10 = !{void (float, float*, float*, i32)* @test, !"kernel", i32 1}
!11 = !{i32 2, !"Dwarf Version", i32 2}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 1, !"wchar_size", i32 4}
!14 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!15 = !{i32 7, !"PIC Level", i32 2}
!16 = !{!"clang version 9.0.0 (trunk 351969) (llvm/trunk 351973)"}
!17 = distinct !DISubprogram(name: "test", linkageName: "test", scope: !8, file: !8, line: 6, type: !18, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
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

; CHECK: .section .debug_abbrev
; CHECK-NEXT: {
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
; CHECK-NEXT: .b8 17                                  // DW_AT_low_pc
; CHECK-NEXT: .b8 1                                   // DW_FORM_addr
; CHECK-NEXT: .b8 18                                  // DW_AT_high_pc
; CHECK-NEXT: .b8 1                                   // DW_FORM_addr
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
; CHECK-NEXT: .b8 135,64                              // DW_AT_MIPS_linkage_name
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
; CHECK-NEXT: .b8 0                                   // EOM(3)
; CHECK-NEXT: }
; CHECK-NEXT: .section .debug_info
; CHECK-NEXT: {
; CHECK-NEXT: .b32 217                                // Length of Unit
; CHECK-NEXT: .b8 2                                   // DWARF version number
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 .debug_abbrev                      // Offset Into Abbrev. Section
; CHECK-NEXT: .b8 8                                   // Address Size (in bytes)
; CHECK-NEXT: .b8 1                                   // Abbrev [1] 0xb:0xd2 DW_TAG_compile_unit
; CHECK-NEXT: .b8 99,108,97,110,103,32,118,101,114,115,105,111,110,32,57,46,48,46,48,32,40,116,114,117,110,107,32,51,53,49,57,54,57,41,32,40,108,108,118,109 // DW_AT_producer
; CHECK-NEXT: .b8 47,116,114,117,110,107,32,51,53,49,57,55,51,41
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                   // DW_AT_language
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 110,101,119,46,99,99                // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 .debug_line                        // DW_AT_stmt_list
; CHECK-NEXT: .b8 47,116,109,112                      // DW_AT_comp_dir
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b64 Lfunc_begin0                       // DW_AT_low_pc
; CHECK-NEXT: .b64 Lfunc_end0                         // DW_AT_high_pc
; CHECK-NEXT: .b8 2                                   // Abbrev [2] 0x65:0x1a DW_TAG_variable
; CHECK-NEXT: .b8 71,76,79,66,65,76                   // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 127                                // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_external
; CHECK-NEXT: .b8 1                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 3                                   // DW_AT_decl_line
; CHECK-NEXT: .b8 5                                   // DW_AT_address_class
; CHECK-NEXT: .b8 9                                   // DW_AT_location
; CHECK-NEXT: .b8 3
; CHECK-NEXT: .b64 GLOBAL
; CHECK-NEXT: .b8 3                                   // Abbrev [3] 0x7f:0x7 DW_TAG_base_type
; CHECK-NEXT: .b8 105,110,116                         // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 5                                   // DW_AT_encoding
; CHECK-NEXT: .b8 4                                   // DW_AT_byte_size
; CHECK-NEXT: .b8 2                                   // Abbrev [2] 0x86:0x1a DW_TAG_variable
; CHECK-NEXT: .b8 83,72,65,82,69,68                   // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 127                                // DW_AT_type
; CHECK-NEXT: .b8 1                                   // DW_AT_external
; CHECK-NEXT: .b8 1                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 4                                   // DW_AT_decl_line
; CHECK-NEXT: .b8 8                                   // DW_AT_address_class
; CHECK-NEXT: .b8 9                                   // DW_AT_location
; CHECK-NEXT: .b8 3
; CHECK-NEXT: .b64 SHARED
; CHECK-NEXT: .b8 4                                   // Abbrev [4] 0xa0:0x33 DW_TAG_subprogram
; CHECK-NEXT: .b64 Lfunc_begin0                       // DW_AT_low_pc
; CHECK-NEXT: .b64 Lfunc_end0                         // DW_AT_high_pc
; CHECK-NEXT: .b8 1                                   // DW_AT_frame_base
; CHECK-NEXT: .b8 156
; CHECK-NEXT: .b8 116,101,115,116                     // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 116,101,115,116                     // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 6                                   // DW_AT_decl_line
; CHECK-NEXT: .b8 1                                   // DW_AT_external
; CHECK-NEXT: .b8 5                                   // Abbrev [5] 0xc0:0x9 DW_TAG_formal_parameter
; CHECK-NEXT: .b8 97                                  // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 6                                   // DW_AT_decl_line
; CHECK-NEXT: .b32 211                                // DW_AT_type
; CHECK-NEXT: .b8 5                                   // Abbrev [5] 0xc9:0x9 DW_TAG_formal_parameter
; CHECK-NEXT: .b8 105                                 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                   // DW_AT_decl_file
; CHECK-NEXT: .b8 6                                   // DW_AT_decl_line
; CHECK-NEXT: .b32 127                                // DW_AT_type
; CHECK-NEXT: .b8 0                                   // End Of Children Mark
; CHECK-NEXT: .b8 3                                   // Abbrev [3] 0xd3:0x9 DW_TAG_base_type
; CHECK-NEXT: .b8 102,108,111,97,116                  // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                   // DW_AT_encoding
; CHECK-NEXT: .b8 4                                   // DW_AT_byte_size
; CHECK-NEXT: .b8 0                                   // End Of Children Mark
; CHECK-NEXT: }
; CHECK-NEXT: .section .debug_macinfo
; CHECK-NEXT: {
; CHECK-NEXT: .b8 0                                   // End Of Macro List Mark
; CHECK:      }

