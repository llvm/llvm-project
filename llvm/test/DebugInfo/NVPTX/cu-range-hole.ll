; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda | FileCheck %s

; CHECK: .target sm_{{[0-9]+}}//, debug

; CHECK: .visible .func  (.param .b32 func_retval0) b(
; CHECK: .param .b32 b_param_0
; CHECK: )
; CHECK: {
; CHECK: Lfunc_begin0:
; CHECK: .loc 1 1 0
; CHECK: .loc 1 1 0
; CHECK: ret;
; CHECK: Lfunc_end0:
; CHECK: }

; CHECK: .visible .func  (.param .b32 func_retval0) a(
; CHECK: .param .b32 a_param_0
; CHECK: )
; CHECK: {
; CHECK: Lfunc_begin1:
; CHECK-NOT: .loc
; CHECK: ret;
; CHECK: Lfunc_end1:
; CHECK: }

; CHECK: .visible .func  (.param .b32 func_retval0) d(
; CHECK: .param .b32 d_param_0
; CHECK: )
; CHECK: {
; CHECK: Lfunc_begin2:
; CHECK: .loc 1 3 0
; CHECK: ret;
; CHECK: Lfunc_end2:
; CHECK: }

; CHECK: .file 1 "{{.*}}b.c"

; Function Attrs: nounwind uwtable
define i32 @b(i32 %c) #0 !dbg !5 {
entry:
  %c.addr = alloca i32, align 4
  store i32 %c, i32* %c.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %c.addr, metadata !13, metadata !DIExpression()), !dbg !14
  %0 = load i32, i32* %c.addr, align 4, !dbg !14
  %add = add nsw i32 %0, 1, !dbg !14
  ret i32 %add, !dbg !14
}

; Function Attrs: nounwind uwtable
define i32 @a(i32 %b) #0 {
entry:
  %b.addr = alloca i32, align 4
  store i32 %b, i32* %b.addr, align 4
  %0 = load i32, i32* %b.addr, align 4
  %add = add nsw i32 %0, 1
  ret i32 %add
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind uwtable
define i32 @d(i32 %e) #0 !dbg !10 {
entry:
  %e.addr = alloca i32, align 4
  store i32 %e, i32* %e.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %e.addr, metadata !15, metadata !DIExpression()), !dbg !16
  %0 = load i32, i32* %e.addr, align 4, !dbg !16
  %add = add nsw i32 %0, 1, !dbg !16
  ret i32 %add, !dbg !16
}

; CHECK: // .section .debug_abbrev
; CHECK: // {
; CHECK: // .b8 1                                // Abbreviation Code
; CHECK: // .b8 17                               // DW_TAG_compile_unit
; CHECK: // .b8 1                                // DW_CHILDREN_yes
; CHECK: // .b8 37                               // DW_AT_producer
; CHECK: // .b8 8                                // DW_FORM_string
; CHECK: // .b8 19                               // DW_AT_language
; CHECK: // .b8 5                                // DW_FORM_data2
; CHECK: // .b8 3                                // DW_AT_name
; CHECK: // .b8 8                                // DW_FORM_string
; CHECK: // .b8 16                               // DW_AT_stmt_list
; CHECK: // .b8 6                                // DW_FORM_data4
; CHECK: // .b8 27                               // DW_AT_comp_dir
; CHECK: // .b8 8                                // DW_FORM_string
; CHECK: // .b8 17                               // DW_AT_low_pc
; CHECK: // .b8 1                                // DW_FORM_addr
; CHECK: // .b8 18                               // DW_AT_high_pc
; CHECK: // .b8 1                                // DW_FORM_addr
; CHECK: // .b8 0                                // EOM(1)
; CHECK: // .b8 0                                // EOM(2)
; CHECK: // .b8 2                                // Abbreviation Code
; CHECK: // .b8 46                               // DW_TAG_subprogram
; CHECK: // .b8 1                                // DW_CHILDREN_yes
; CHECK: // .b8 17                               // DW_AT_low_pc
; CHECK: // .b8 1                                // DW_FORM_addr
; CHECK: // .b8 18                               // DW_AT_high_pc
; CHECK: // .b8 1                                // DW_FORM_addr
; CHECK: // .b8 3                                // DW_AT_name
; CHECK: // .b8 8                                // DW_FORM_string
; CHECK: // .b8 58                               // DW_AT_decl_file
; CHECK: // .b8 11                               // DW_FORM_data1
; CHECK: // .b8 59                               // DW_AT_decl_line
; CHECK: // .b8 11                               // DW_FORM_data1
; CHECK: // .b8 39                               // DW_AT_prototyped
; CHECK: // .b8 12                               // DW_FORM_flag
; CHECK: // .b8 73                               // DW_AT_type
; CHECK: // .b8 19                               // DW_FORM_ref4
; CHECK: // .b8 63                               // DW_AT_external
; CHECK: // .b8 12                               // DW_FORM_flag
; CHECK: // .b8 0                                // EOM(1)
; CHECK: // .b8 0                                // EOM(2)
; CHECK: // .b8 3                                // Abbreviation Code
; CHECK: // .b8 5                                // DW_TAG_formal_parameter
; CHECK: // .b8 0                                // DW_CHILDREN_no
; CHECK: // .b8 3                                // DW_AT_name
; CHECK: // .b8 8                                // DW_FORM_string
; CHECK: // .b8 58                               // DW_AT_decl_file
; CHECK: // .b8 11                               // DW_FORM_data1
; CHECK: // .b8 59                               // DW_AT_decl_line
; CHECK: // .b8 11                               // DW_FORM_data1
; CHECK: // .b8 73                               // DW_AT_type
; CHECK: // .b8 19                               // DW_FORM_ref4
; CHECK: // .b8 0                                // EOM(1)
; CHECK: // .b8 0                                // EOM(2)
; CHECK: // .b8 4                                // Abbreviation Code
; CHECK: // .b8 36                               // DW_TAG_base_type
; CHECK: // .b8 0                                // DW_CHILDREN_no
; CHECK: // .b8 3                                // DW_AT_name
; CHECK: // .b8 8                                // DW_FORM_string
; CHECK: // .b8 62                               // DW_AT_encoding
; CHECK: // .b8 11                               // DW_FORM_data1
; CHECK: // .b8 11                               // DW_AT_byte_size
; CHECK: // .b8 11                               // DW_FORM_data1
; CHECK: // .b8 0                                // EOM(1)
; CHECK: // .b8 0                                // EOM(2)
; CHECK: // .b8 0                                // EOM(3)
; CHECK: // }
; CHECK: // .section .debug_info
; CHECK: // {
; CHECK: // .b32 179                             // Length of Unit
; CHECK: // .b8 2                                // DWARF version number
; CHECK: // .b8 0
; CHECK: // .b32 .debug_abbrev                   // Offset Into Abbrev. Section
; CHECK: // .b8 8                                // Address Size (in bytes)
; CHECK: // .b8 1                                // Abbrev [1] 0xb:0xac DW_TAG_compile_unit
; CHECK: // .b8 99                               // DW_AT_producer
; CHECK: // .b8 108
; CHECK: // .b8 97
; CHECK: // .b8 110
; CHECK: // .b8 103
; CHECK: // .b8 32
; CHECK: // .b8 118
; CHECK: // .b8 101
; CHECK: // .b8 114
; CHECK: // .b8 115
; CHECK: // .b8 105
; CHECK: // .b8 111
; CHECK: // .b8 110
; CHECK: // .b8 32
; CHECK: // .b8 51
; CHECK: // .b8 46
; CHECK: // .b8 53
; CHECK: // .b8 46
; CHECK: // .b8 48
; CHECK: // .b8 32
; CHECK: // .b8 40
; CHECK: // .b8 116
; CHECK: // .b8 114
; CHECK: // .b8 117
; CHECK: // .b8 110
; CHECK: // .b8 107
; CHECK: // .b8 32
; CHECK: // .b8 50
; CHECK: // .b8 48
; CHECK: // .b8 52
; CHECK: // .b8 49
; CHECK: // .b8 54
; CHECK: // .b8 52
; CHECK: // .b8 41
; CHECK: // .b8 32
; CHECK: // .b8 40
; CHECK: // .b8 108
; CHECK: // .b8 108
; CHECK: // .b8 118
; CHECK: // .b8 109
; CHECK: // .b8 47
; CHECK: // .b8 116
; CHECK: // .b8 114
; CHECK: // .b8 117
; CHECK: // .b8 110
; CHECK: // .b8 107
; CHECK: // .b8 32
; CHECK: // .b8 50
; CHECK: // .b8 48
; CHECK: // .b8 52
; CHECK: // .b8 49
; CHECK: // .b8 56
; CHECK: // .b8 51
; CHECK: // .b8 41
; CHECK: // .b8 0
; CHECK: // .b8 12                               // DW_AT_language
; CHECK: // .b8 0
; CHECK: // .b8 98                               // DW_AT_name
; CHECK: // .b8 46
; CHECK: // .b8 99
; CHECK: // .b8 0
; CHECK: // .b32 .debug_line                     // DW_AT_stmt_list
; CHECK: // .b8 47                               // DW_AT_comp_dir
; CHECK: // .b8 115
; CHECK: // .b8 111
; CHECK: // .b8 117
; CHECK: // .b8 114
; CHECK: // .b8 99
; CHECK: // .b8 101
; CHECK: // .b8 0
; CHECK: // .b64 Lfunc_begin0                    // DW_AT_low_pc
; CHECK: // .b64 Lfunc_end2                      // DW_AT_high_pc
; CHECK: // .b8 2                                // Abbrev [2] 0x65:0x25 DW_TAG_subprogram
; CHECK: // .b64 Lfunc_begin0                    // DW_AT_low_pc
; CHECK: // .b64 Lfunc_end0                      // DW_AT_high_pc
; CHECK: // .b8 98                               // DW_AT_name
; CHECK: // .b8 0
; CHECK: // .b8 1                                // DW_AT_decl_file
; CHECK: // .b8 1                                // DW_AT_decl_line
; CHECK: // .b8 1                                // DW_AT_prototyped
; CHECK: // .b32 175                             // DW_AT_type
; CHECK: // .b8 1                                // DW_AT_external
; CHECK: // .b8 3                                // Abbrev [3] 0x80:0x9 DW_TAG_formal_parameter
; CHECK: // .b8 99                               // DW_AT_name
; CHECK: // .b8 0
; CHECK: // .b8 1                                // DW_AT_decl_file
; CHECK: // .b8 1                                // DW_AT_decl_line
; CHECK: // .b32 175                             // DW_AT_type
; CHECK: // .b8 0                                // End Of Children Mark
; CHECK: // .b8 2                                // Abbrev [2] 0x8a:0x25 DW_TAG_subprogram
; CHECK: // .b64 Lfunc_begin2                    // DW_AT_low_pc
; CHECK: // .b64 Lfunc_end2                      // DW_AT_high_pc
; CHECK: // .b8 100                              // DW_AT_name
; CHECK: // .b8 0
; CHECK: // .b8 1                                // DW_AT_decl_file
; CHECK: // .b8 3                                // DW_AT_decl_line
; CHECK: // .b8 1                                // DW_AT_prototyped
; CHECK: // .b32 175                             // DW_AT_type
; CHECK: // .b8 1                                // DW_AT_external
; CHECK: // .b8 3                                // Abbrev [3] 0xa5:0x9 DW_TAG_formal_parameter
; CHECK: // .b8 101                              // DW_AT_name
; CHECK: // .b8 0
; CHECK: // .b8 1                                // DW_AT_decl_file
; CHECK: // .b8 3                                // DW_AT_decl_line
; CHECK: // .b32 175                             // DW_AT_type
; CHECK: // .b8 0                                // End Of Children Mark
; CHECK: // .b8 4                                // Abbrev [4] 0xaf:0x7 DW_TAG_base_type
; CHECK: // .b8 105                              // DW_AT_name
; CHECK: // .b8 110
; CHECK: // .b8 116
; CHECK: // .b8 0
; CHECK: // .b8 5                                // DW_AT_encoding
; CHECK: // .b8 4                                // DW_AT_byte_size
; CHECK: // .b8 0                                // End Of Children Mark
; CHECK: // }
; CHECK: // .section .debug_macinfo
; CHECK: // {
; CHECK: // .b8 0                                // End Of Macro List Mark
; CHECK: // }

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.ident = !{!0, !0}
!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!11, !12}

!0 = !{!"clang version 3.5.0 (trunk 204164) (llvm/trunk 204183)"}
!1 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.5.0 (trunk 204164) (llvm/trunk 204183)", isOptimized: false, emissionKind: FullDebug, file: !2, enums: !3, retainedTypes: !3, globals: !3, imports: !3)
!2 = !DIFile(filename: "b.c", directory: "/source")
!3 = !{}
!5 = distinct !DISubprogram(name: "b", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !1, scopeLine: 1, file: !2, scope: !6, type: !7, retainedNodes: !3)
!6 = !DIFile(filename: "b.c", directory: "/source")
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !9}
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = distinct !DISubprogram(name: "d", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !1, scopeLine: 3, file: !2, scope: !6, type: !7, retainedNodes: !3)
!11 = !{i32 2, !"Dwarf Version", i32 2}
!12 = !{i32 1, !"Debug Info Version", i32 3}
!13 = !DILocalVariable(name: "c", line: 1, arg: 1, scope: !5, file: !6, type: !9)
!14 = !DILocation(line: 1, scope: !5)
!15 = !DILocalVariable(name: "e", line: 3, arg: 1, scope: !10, file: !6, type: !9)
!16 = !DILocation(line: 3, scope: !10)
