; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda | FileCheck %s

; CHECK: .target sm_{{[0-9]+}}, debug

; CHECK: .visible .func  (.param .b32 func_retval0) b(
; CHECK: .param .b32 b_param_0
; CHECK: )
; CHECK: {
; CHECK: .loc 1 1 0
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
; CHECK: .loc 1 3 0
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
; CHECK-NEXT: .b8 73                               // DW_AT_type
; CHECK-NEXT: .b8 19                               // DW_FORM_ref4
; CHECK-NEXT: .b8 63                               // DW_AT_external
; CHECK-NEXT: .b8 12                               // DW_FORM_flag
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 3                                // Abbreviation Code
; CHECK-NEXT: .b8 5                                // DW_TAG_formal_parameter
; CHECK-NEXT: .b8 0                                // DW_CHILDREN_no
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
; CHECK-NEXT: .b32 183                             // Length of Unit
; CHECK-NEXT: .b8 2                                // DWARF version number
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 .debug_abbrev                   // Offset Into Abbrev. Section
; CHECK-NEXT: .b8 8                                // Address Size (in bytes)
; CHECK-NEXT: .b8 1                                // Abbrev [1] 0xb:0xb0 DW_TAG_compile_unit
; CHECK-NEXT: .b8 99                               // DW_AT_producer
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
; CHECK-NEXT: .b8 51
; CHECK-NEXT: .b8 46
; CHECK-NEXT: .b8 53
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
; CHECK-NEXT: .b8 50
; CHECK-NEXT: .b8 48
; CHECK-NEXT: .b8 52
; CHECK-NEXT: .b8 49
; CHECK-NEXT: .b8 54
; CHECK-NEXT: .b8 52
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
; CHECK-NEXT: .b8 50
; CHECK-NEXT: .b8 48
; CHECK-NEXT: .b8 52
; CHECK-NEXT: .b8 49
; CHECK-NEXT: .b8 56
; CHECK-NEXT: .b8 51
; CHECK-NEXT: .b8 41
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 12                               // DW_AT_language
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 98                               // DW_AT_name
; CHECK-NEXT: .b8 46
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 .debug_line                     // DW_AT_stmt_list
; CHECK-NEXT: .b8 47                               // DW_AT_comp_dir
; CHECK-NEXT: .b8 115
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b64 Lfunc_begin0                    // DW_AT_low_pc
; CHECK-NEXT: .b64 Lfunc_end2                      // DW_AT_high_pc
; CHECK-NEXT: .b8 2                                // Abbrev [2] 0x65:0x27 DW_TAG_subprogram
; CHECK-NEXT: .b64 Lfunc_begin0                    // DW_AT_low_pc
; CHECK-NEXT: .b64 Lfunc_end0                      // DW_AT_high_pc
; CHECK-NEXT: .b8 1                                // DW_AT_frame_base
; CHECK-NEXT: .b8 156
; CHECK-NEXT: .b8 98                               // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 1                                // DW_AT_decl_line
; CHECK-NEXT: .b8 1                                // DW_AT_prototyped
; CHECK-NEXT: .b32 179                             // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x82:0x9 DW_TAG_formal_parameter
; CHECK-NEXT: .b8 99                               // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 1                                // DW_AT_decl_line
; CHECK-NEXT: .b32 179                             // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 2                                // Abbrev [2] 0x8c:0x27 DW_TAG_subprogram
; CHECK-NEXT: .b64 Lfunc_begin2                    // DW_AT_low_pc
; CHECK-NEXT: .b64 Lfunc_end2                      // DW_AT_high_pc
; CHECK-NEXT: .b8 1                                // DW_AT_frame_base
; CHECK-NEXT: .b8 156
; CHECK-NEXT: .b8 100                              // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 3                                // DW_AT_decl_line
; CHECK-NEXT: .b8 1                                // DW_AT_prototyped
; CHECK-NEXT: .b32 179                             // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0xa9:0x9 DW_TAG_formal_parameter
; CHECK-NEXT: .b8 101                              // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 3                                // DW_AT_decl_line
; CHECK-NEXT: .b32 179                             // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0xb3:0x7 DW_TAG_base_type
; CHECK-NEXT: .b8 105                              // DW_AT_name
; CHECK-NEXT: .b8 110
; CHECK-NEXT: .b8 116
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 5                                // DW_AT_encoding
; CHECK-NEXT: .b8 4                                // DW_AT_byte_size
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: }
; CHECK-NEXT: .section .debug_loc { }
; CHECK-NOT: debug_

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.ident = !{!0, !0}
!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!11, !12}

!0 = !{!"clang version 3.5.0 (trunk 204164) (llvm/trunk 204183)"}
!1 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.5.0 (trunk 204164) (llvm/trunk 204183)", isOptimized: false, emissionKind: FullDebug, file: !2, enums: !3, retainedTypes: !3, globals: !3, imports: !3, nameTableKind: None)
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
