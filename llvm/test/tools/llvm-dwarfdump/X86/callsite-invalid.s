# RUN: llvm-mc -triple x86_64-apple-darwin %s -filetype=obj -o %t.o
# RUN: not llvm-dwarfdump -verify %t.o 2>&1 | FileCheck %s

# CHECK: error: Subprogram with call site entry has no DW_AT_call attribute:
# CHECK: DW_TAG_subprogram
# CHECK:   DW_AT_name ("main")
# CHECK: DW_TAG_call_site
# CHECK:   DW_AT_call_origin
# CHECK: Errors detected.

# Source:
## define void @foo() !dbg !25 {
##   ret void, !dbg !28
## }
##
## define i32 @main() !dbg !29 {
##   call void @foo(), !dbg !32
##   ret i32 0, !dbg !33
## }
##
## !llvm.dbg.cu = !{!2}
## !llvm.module.flags = !{!8, !9, !10, !11}
## !llvm.ident = !{!12}
##
## !0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
## !1 = distinct !DIGlobalVariable(name: "sink", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
## !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 7.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
## !3 = !DIFile(filename: "/Users/vsk/src/llvm.org-tailcall/tail2.cc", directory: "/Users/vsk/src/builds/llvm-project-tailcall-RA", checksumkind: CSK_MD5, checksum: "3b61952c21b7f657ddb7c0ad44cf5529")
## !4 = !{}
## !5 = !{!0}
## !6 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !7)
## !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
## !8 = !{i32 2, !"Dwarf Version", i32 5}
## !9 = !{i32 2, !"Debug Info Version", i32 3}
## !10 = !{i32 1, !"wchar_size", i32 4}
## !11 = !{i32 7, !"PIC Level", i32 2}
## !12 = !{!"clang version 7.0.0 "}
## !13 = distinct !DISubprogram(name: "bat", linkageName: "_Z3batv", scope: !3, file: !3, line: 2, type: !14, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, isOptimized: true, unit: !2, retainedNodes: !4)
## !14 = !DISubroutineType(types: !15)
## !15 = !{null}
## !16 = !DILocation(line: 2, column: 44, scope: !13)
## !17 = !{!18, !18, i64 0}
## !18 = !{!"int", !19, i64 0}
## !19 = !{!"omnipotent char", !20, i64 0}
## !20 = !{!"Simple C++ TBAA"}
## !21 = !DILocation(line: 2, column: 48, scope: !13)
## !22 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barv", scope: !3, file: !3, line: 3, type: !14, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, isOptimized: true, unit: !2, retainedNodes: !4)
## !23 = !DILocation(line: 3, column: 44, scope: !22)
## !24 = !DILocation(line: 3, column: 48, scope: !22)
## !25 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !3, file: !3, line: 4, type: !14, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, isOptimized: true, unit: !2, retainedNodes: !4)
## !26 = !DILocation(line: 5, column: 3, scope: !25)
## !27 = !DILocation(line: 6, column: 3, scope: !25)
## !28 = !DILocation(line: 7, column: 1, scope: !25)
## !29 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 8, type: !30, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, isOptimized: true, unit: !2, retainedNodes: !4)
## !30 = !DISubroutineType(types: !31)
## !31 = !{!7}
## !32 = !DILocation(line: 8, column: 50, scope: !29)
## !33 = !DILocation(line: 8, column: 57, scope: !29)

        .section        __TEXT,__text,regular,pure_instructions
        .globl  _foo                    ## -- Begin function foo
_foo:                                   ## @foo
Lfunc_begin0:
        .cfi_startproc
## %bb.0:
        retq
Ltmp0:
Lfunc_end0:
        .cfi_endproc
                                        ## -- End function
        .globl  _main                   ## -- Begin function main
_main:                                  ## @main
Lfunc_begin1:
        .cfi_startproc
## %bb.0:
        pushq   %rax
        .cfi_def_cfa_offset 16
Ltmp1:
        callq   _foo
        xorl    %eax, %eax
        popq    %rcx
        retq
Ltmp2:
Lfunc_end1:
        .cfi_endproc
                                        ## -- End function
        .section        __DWARF,__debug_str_offs,regular,debug
Lsection_str_off:
        .long   36
        .short  5
        .short  0
Lstr_offsets_base0:
        .section        __DWARF,__debug_str,regular,debug
Linfo_string:
        .asciz  "clang version 7.0.0 "  ## string offset=0
        .asciz  "/Users/vsk/src/llvm.org-tailcall/tail2.cc" ## string offset=21
        .asciz  "/Users/vsk/src/builds/llvm-project-tailcall-RA" ## string offset=63
        .asciz  "sink"                  ## string offset=110
        .asciz  "int"                   ## string offset=115
        .asciz  "foo"                   ## string offset=119
        .asciz  "_Z3foov"               ## string offset=123
        .asciz  "main"                  ## string offset=131
        .section        __DWARF,__debug_str_offs,regular,debug
        .long   0
        .long   21
        .long   63
        .long   110
        .long   115
        .long   119
        .long   123
        .long   131
        .section        __DWARF,__debug_abbrev,regular,debug
Lsection_abbrev:
        .byte   1                       ## Abbreviation Code
        .byte   17                      ## DW_TAG_compile_unit
        .byte   1                       ## DW_CHILDREN_yes
        .byte   37                      ## DW_AT_producer
        .byte   37                      ## DW_FORM_strx1
        .byte   19                      ## DW_AT_language
        .byte   5                       ## DW_FORM_data2
        .byte   3                       ## DW_AT_name
        .byte   37                      ## DW_FORM_strx1
        .byte   114                     ## DW_AT_str_offsets_base
        .byte   23                      ## DW_FORM_sec_offset
        .byte   16                      ## DW_AT_stmt_list
        .byte   23                      ## DW_FORM_sec_offset
        .byte   27                      ## DW_AT_comp_dir
        .byte   37                      ## DW_FORM_strx1
        .ascii  "\341\177"              ## DW_AT_APPLE_optimized
        .byte   25                      ## DW_FORM_flag_present
        .byte   17                      ## DW_AT_low_pc
        .byte   1                       ## DW_FORM_addr
        .byte   18                      ## DW_AT_high_pc
        .byte   6                       ## DW_FORM_data4
        .byte   0                       ## EOM(1)
        .byte   0                       ## EOM(2)
        .byte   2                       ## Abbreviation Code
        .byte   52                      ## DW_TAG_variable
        .byte   0                       ## DW_CHILDREN_no
        .byte   3                       ## DW_AT_name
        .byte   37                      ## DW_FORM_strx1
        .byte   73                      ## DW_AT_type
        .byte   19                      ## DW_FORM_ref4
        .byte   63                      ## DW_AT_external
        .byte   25                      ## DW_FORM_flag_present
        .byte   58                      ## DW_AT_decl_file
        .byte   11                      ## DW_FORM_data1
        .byte   59                      ## DW_AT_decl_line
        .byte   11                      ## DW_FORM_data1
        .byte   0                       ## EOM(1)
        .byte   0                       ## EOM(2)
        .byte   3                       ## Abbreviation Code
        .byte   53                      ## DW_TAG_volatile_type
        .byte   0                       ## DW_CHILDREN_no
        .byte   73                      ## DW_AT_type
        .byte   19                      ## DW_FORM_ref4
        .byte   0                       ## EOM(1)
        .byte   0                       ## EOM(2)
        .byte   4                       ## Abbreviation Code
        .byte   36                      ## DW_TAG_base_type
        .byte   0                       ## DW_CHILDREN_no
        .byte   3                       ## DW_AT_name
        .byte   37                      ## DW_FORM_strx1
        .byte   62                      ## DW_AT_encoding
        .byte   11                      ## DW_FORM_data1
        .byte   11                      ## DW_AT_byte_size
        .byte   11                      ## DW_FORM_data1
        .byte   0                       ## EOM(1)
        .byte   0                       ## EOM(2)
        .byte   5                       ## Abbreviation Code
        .byte   46                      ## DW_TAG_subprogram
        .byte   0                       ## DW_CHILDREN_no
        .byte   17                      ## DW_AT_low_pc
        .byte   1                       ## DW_FORM_addr
        .byte   18                      ## DW_AT_high_pc
        .byte   6                       ## DW_FORM_data4
        .ascii  "\347\177"              ## DW_AT_APPLE_omit_frame_ptr
        .byte   25                      ## DW_FORM_flag_present
        .byte   64                      ## DW_AT_frame_base
        .byte   24                      ## DW_FORM_exprloc
##        .byte   122                     ## DW_AT_call_all_calls
##        .byte   25                      ## DW_FORM_flag_present
        .byte   110                     ## DW_AT_linkage_name
        .byte   37                      ## DW_FORM_strx1
        .byte   3                       ## DW_AT_name
        .byte   37                      ## DW_FORM_strx1
        .byte   58                      ## DW_AT_decl_file
        .byte   11                      ## DW_FORM_data1
        .byte   59                      ## DW_AT_decl_line
        .byte   11                      ## DW_FORM_data1
        .byte   63                      ## DW_AT_external
        .byte   25                      ## DW_FORM_flag_present
        .ascii  "\341\177"              ## DW_AT_APPLE_optimized
        .byte   25                      ## DW_FORM_flag_present
        .byte   0                       ## EOM(1)
        .byte   0                       ## EOM(2)
        .byte   6                       ## Abbreviation Code
        .byte   46                      ## DW_TAG_subprogram
        .byte   1                       ## DW_CHILDREN_yes
        .byte   17                      ## DW_AT_low_pc
        .byte   1                       ## DW_FORM_addr
        .byte   18                      ## DW_AT_high_pc
        .byte   6                       ## DW_FORM_data4
        .ascii  "\347\177"              ## DW_AT_APPLE_omit_frame_ptr
        .byte   25                      ## DW_FORM_flag_present
        .byte   64                      ## DW_AT_frame_base
        .byte   24                      ## DW_FORM_exprloc
##        .byte   122                     ## DW_AT_call_all_calls
##        .byte   25                      ## DW_FORM_flag_present
        .byte   3                       ## DW_AT_name
        .byte   37                      ## DW_FORM_strx1
        .byte   58                      ## DW_AT_decl_file
        .byte   11                      ## DW_FORM_data1
        .byte   59                      ## DW_AT_decl_line
        .byte   11                      ## DW_FORM_data1
        .byte   73                      ## DW_AT_type
        .byte   19                      ## DW_FORM_ref4
        .byte   63                      ## DW_AT_external
        .byte   25                      ## DW_FORM_flag_present
        .ascii  "\341\177"              ## DW_AT_APPLE_optimized
        .byte   25                      ## DW_FORM_flag_present
        .byte   0                       ## EOM(1)
        .byte   0                       ## EOM(2)
        .byte   7                       ## Abbreviation Code
        .byte   72                      ## DW_TAG_call_site
        .byte   0                       ## DW_CHILDREN_no
        .byte   127                     ## DW_AT_call_origin
        .byte   19                      ## DW_FORM_ref4
        .byte   0                       ## EOM(1)
        .byte   0                       ## EOM(2)
        .byte   0                       ## EOM(3)
        .section        __DWARF,__debug_info,regular,debug
Lsection_info:
Lcu_begin0:
        .long   99                      ## Length of Unit
        .short  5                       ## DWARF version number
        .byte   1                       ## DWARF Unit Type
        .byte   8                       ## Address Size (in bytes)
.set Lset0, Lsection_abbrev-Lsection_abbrev ## Offset Into Abbrev. Section
        .long   Lset0
        .byte   1                       ## Abbrev [1] 0xc:0x5b DW_TAG_compile_unit
        .byte   0                       ## DW_AT_producer
        .short  4                       ## DW_AT_language
        .byte   1                       ## DW_AT_name
.set Lset1, Lstr_offsets_base0-Lsection_str_off ## DW_AT_str_offsets_base
        .long   Lset1
.set Lset2, Lline_table_start0-Lsection_line ## DW_AT_stmt_list
        .long   Lset2
        .byte   2                       ## DW_AT_comp_dir
                                        ## DW_AT_APPLE_optimized
        .quad   Lfunc_begin0            ## DW_AT_low_pc
.set Lset3, Lfunc_end1-Lfunc_begin0     ## DW_AT_high_pc
        .long   Lset3
        .byte   2                       ## Abbrev [2] 0x26:0x8 DW_TAG_variable
        .byte   3                       ## DW_AT_name
        .long   46                      ## DW_AT_type
                                        ## DW_AT_external
        .byte   1                       ## DW_AT_decl_file
        .byte   1                       ## DW_AT_decl_line
        .byte   3                       ## Abbrev [3] 0x2e:0x5 DW_TAG_volatile_type
        .long   51                      ## DW_AT_type
        .byte   4                       ## Abbrev [4] 0x33:0x4 DW_TAG_base_type
        .byte   4                       ## DW_AT_name
        .byte   5                       ## DW_AT_encoding
        .byte   4                       ## DW_AT_byte_size
        .byte   5                       ## Abbrev [5] 0x37:0x13 DW_TAG_subprogram
        .quad   Lfunc_begin0            ## DW_AT_low_pc
.set Lset4, Lfunc_end0-Lfunc_begin0     ## DW_AT_high_pc
        .long   Lset4
                                        ## DW_AT_APPLE_omit_frame_ptr
        .byte   1                       ## DW_AT_frame_base
        .byte   87
                                        ## DW_AT_call_all_calls
        .byte   6                       ## DW_AT_linkage_name
        .byte   5                       ## DW_AT_name
        .byte   1                       ## DW_AT_decl_file
        .byte   4                       ## DW_AT_decl_line
                                        ## DW_AT_external
                                        ## DW_AT_APPLE_optimized
        .byte   6                       ## Abbrev [6] 0x4a:0x1c DW_TAG_subprogram
        .quad   Lfunc_begin1            ## DW_AT_low_pc
.set Lset5, Lfunc_end1-Lfunc_begin1     ## DW_AT_high_pc
        .long   Lset5
                                        ## DW_AT_APPLE_omit_frame_ptr
        .byte   1                       ## DW_AT_frame_base
        .byte   87
                                        ## DW_AT_call_all_calls
        .byte   7                       ## DW_AT_name
        .byte   1                       ## DW_AT_decl_file
        .byte   8                       ## DW_AT_decl_line
        .long   51                      ## DW_AT_type
                                        ## DW_AT_external
                                        ## DW_AT_APPLE_optimized
        .byte   7                       ## Abbrev [7] 0x60:0x5 DW_TAG_call_site
        .long   55                      ## DW_AT_call_origin
        .byte   0                       ## End Of Children Mark
        .byte   0                       ## End Of Children Mark
        .section        __DWARF,__debug_macinfo,regular,debug
Ldebug_macinfo:
        .byte   0                       ## End Of Macro List Mark
        .section        __DWARF,__debug_names,regular,debug
Ldebug_names_begin:
.set Lset6, Lnames_end0-Lnames_start0   ## Header: unit length
        .long   Lset6
Lnames_start0:
        .short  5                       ## Header: version
        .short  0                       ## Header: padding
        .long   1                       ## Header: compilation unit count
        .long   0                       ## Header: local type unit count
        .long   0                       ## Header: foreign type unit count
        .long   4                       ## Header: bucket count
        .long   4                       ## Header: name count
.set Lset7, Lnames_abbrev_end0-Lnames_abbrev_start0 ## Header: abbreviation table size
        .long   Lset7
        .long   8                       ## Header: augmentation string size
        .ascii  "LLVM0700"              ## Header: augmentation string
.set Lset8, Lcu_begin0-Lsection_info    ## Compilation unit 0
        .long   Lset8
        .long   1                       ## Bucket 0
        .long   2                       ## Bucket 1
        .long   3                       ## Bucket 2
        .long   4                       ## Bucket 3
        .long   193495088               ## Hash in Bucket 0
        .long   193491849               ## Hash in Bucket 1
        .long   2090499946              ## Hash in Bucket 2
        .long   -1257882357             ## Hash in Bucket 3
        .long   115                     ## String in Bucket 0: int
        .long   119                     ## String in Bucket 1: foo
        .long   131                     ## String in Bucket 2: main
        .long   123                     ## String in Bucket 3: _Z3foov
.set Lset9, Lnames3-Lnames_entries0     ## Offset in Bucket 0
        .long   Lset9
.set Lset10, Lnames0-Lnames_entries0    ## Offset in Bucket 1
        .long   Lset10
.set Lset11, Lnames1-Lnames_entries0    ## Offset in Bucket 2
        .long   Lset11
.set Lset12, Lnames2-Lnames_entries0    ## Offset in Bucket 3
        .long   Lset12
Lnames_abbrev_start0:
        .byte   46                      ## Abbrev code
        .byte   46                      ## DW_TAG_subprogram
        .byte   3                       ## DW_IDX_die_offset
        .byte   19                      ## DW_FORM_ref4
        .byte   0                       ## End of abbrev
        .byte   0                       ## End of abbrev
        .byte   36                      ## Abbrev code
        .byte   36                      ## DW_TAG_base_type
        .byte   3                       ## DW_IDX_die_offset
        .byte   19                      ## DW_FORM_ref4
        .byte   0                       ## End of abbrev
        .byte   0                       ## End of abbrev
        .byte   0                       ## End of abbrev list
Lnames_abbrev_end0:
Lnames_entries0:
Lnames3:
        .byte   36                      ## Abbreviation code
        .long   51                      ## DW_IDX_die_offset
        .long   0                       ## End of list: int
Lnames0:
        .byte   46                      ## Abbreviation code
        .long   55                      ## DW_IDX_die_offset
        .long   0                       ## End of list: foo
Lnames1:
        .byte   46                      ## Abbreviation code
        .long   74                      ## DW_IDX_die_offset
        .long   0                       ## End of list: main
Lnames2:
        .byte   46                      ## Abbreviation code
        .long   55                      ## DW_IDX_die_offset
        .long   0                       ## End of list: _Z3foov
Lnames_end0:

.subsections_via_symbols
        .section        __DWARF,__debug_line,regular,debug
Lsection_line:
Lline_table_start0:
