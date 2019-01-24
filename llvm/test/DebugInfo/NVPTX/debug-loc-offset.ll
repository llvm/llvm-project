; RUN: llc -mtriple=nvptx64-nvidia-cuda < %s | FileCheck %s

; CHECK: .target sm_{{[0-9]+}}, debug

; CHECK: .extern .func  (.param .b32 func_retval0) _ZN1A3fooEv
; CHECK: (
; CHECK: .param .b64 _ZN1A3fooEv_param_0
; CHECK: )

%struct.A = type { i32 (...)**, i32 }

; CHECK: .visible .func  (.param .b32 func_retval0) _Z3bari(
; CHECK: {
; CHECK: .loc [[CU1:[0-9]+]] 1 0
; CHECK: Lfunc_begin0:
; CHECK: .loc [[CU1]] 1 0

; CHECK: //DEBUG_VALUE: bar:b <- {{[0-9]+}}
; CHECK: //DEBUG_VALUE: bar:b <- {{[0-9]+}}
; CHECK: .loc [[CU1]] 2 0
; CHECK: ret;
; CHECK: }

; Function Attrs: nounwind
define i32 @_Z3bari(i32 %b) #0 !dbg !4 {
entry:
  %b.addr = alloca i32, align 4
  store i32 %b, i32* %b.addr, align 4
  call void @llvm.dbg.value(metadata i32 0, metadata !21, metadata !DIExpression()), !dbg !22
  %0 = load i32, i32* %b.addr, align 4, !dbg !23
  call void @llvm.dbg.value(metadata i32 1, metadata !21, metadata !DIExpression()), !dbg !22
  %add = add nsw i32 %0, 4, !dbg !23
  ret i32 %add, !dbg !23
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @llvm.dbg.value(metadata, metadata, metadata) #1

; CHECK: .visible .func _Z3baz1A(
; CHECK: {
; CHECK: .loc [[CU2:[0-9]+]] 6 0
; CHECK: Lfunc_begin1:
; CHECK: .loc [[CU2]] 6 0
; CHECK: //DEBUG_VALUE: baz:z <- {{[0-9]+}}
; CHECK: //DEBUG_VALUE: baz:z <- {{[0-9]+}}
; CHECK: .loc [[CU2]] 10 0
; CHECK: ret;
; CHECK: }

define void @_Z3baz1A(%struct.A* %a) #2 !dbg !14 {
entry:
  %z = alloca i32, align 4
  call void @llvm.dbg.declare(metadata %struct.A* %a, metadata !24, metadata !DIExpression(DW_OP_deref)), !dbg !25
  call void @llvm.dbg.declare(metadata i32* %z, metadata !26, metadata !DIExpression()), !dbg !27
  store i32 2, i32* %z, align 4, !dbg !27
  %var = getelementptr inbounds %struct.A, %struct.A* %a, i32 0, i32 1, !dbg !28
  %0 = load i32, i32* %var, align 4, !dbg !28
  %cmp = icmp sgt i32 %0, 2, !dbg !28
  br i1 %cmp, label %if.then, label %if.end, !dbg !28

if.then:                                          ; preds = %entry
  %1 = load i32, i32* %z, align 4, !dbg !30
  %inc = add nsw i32 %1, 1, !dbg !30
  store i32 %inc, i32* %z, align 4, !dbg !30
  br label %if.end, !dbg !30

if.end:                                           ; preds = %if.then, %entry
  %call = call signext i8 @_ZN1A3fooEv(%struct.A* %a), !dbg !31
  %conv = sext i8 %call to i32, !dbg !31
  %cmp1 = icmp eq i32 %conv, 97, !dbg !31
  br i1 %cmp1, label %if.then2, label %if.end4, !dbg !31

if.then2:                                         ; preds = %if.end
  %2 = load i32, i32* %z, align 4, !dbg !33
  %inc3 = add nsw i32 %2, 1, !dbg !33
  store i32 %inc3, i32* %z, align 4, !dbg !33
  br label %if.end4, !dbg !33

if.end4:                                          ; preds = %if.then2, %if.end
  ret void, !dbg !34
}

; CHECK-DAG: .file [[CU1]] "/llvm_cmake_gcc{{/|\\\\}}debug-loc-offset1.cc"
; CHECK-DAG: .file [[CU2]] "/llvm_cmake_gcc{{/|\\\\}}debug-loc-offset2.cc"

declare signext i8 @_ZN1A3fooEv(%struct.A*) #2

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0, !9}
!llvm.module.flags = !{!18, !19}
!llvm.ident = !{!20, !20}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0 (210479)", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2, nameTableKind: None)
!1 = !DIFile(filename: "debug-loc-offset1.cc", directory: "/llvm_cmake_gcc")
!2 = !{}
!4 = distinct !DISubprogram(name: "bar", linkageName: "_Z3bari", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 1, file: !1, scope: !5, type: !6, retainedNodes: !2)
!5 = !DIFile(filename: "debug-loc-offset1.cc", directory: "/llvm_cmake_gcc")
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !8}
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0 (210479)", isOptimized: false, emissionKind: FullDebug, file: !10, enums: !2, retainedTypes: !11, globals: !2, imports: !2, nameTableKind: None)
!10 = !DIFile(filename: "debug-loc-offset2.cc", directory: "/llvm_cmake_gcc")
!11 = !{!12}
!12 = !DICompositeType(tag: DW_TAG_structure_type, name: "A", line: 1, flags: DIFlagFwdDecl, file: !10, identifier: "_ZTS1A")
!14 = distinct !DISubprogram(name: "baz", linkageName: "_Z3baz1A", line: 6, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !9, scopeLine: 6, file: !10, scope: !15, type: !16, retainedNodes: !2)
!15 = !DIFile(filename: "debug-loc-offset2.cc", directory: "/llvm_cmake_gcc")
!16 = !DISubroutineType(types: !17)
!17 = !{null, !12}
!18 = !{i32 2, !"Dwarf Version", i32 4}
!19 = !{i32 2, !"Debug Info Version", i32 3}
!20 = !{!"clang version 3.5.0 (210479)"}
!21 = !DILocalVariable(name: "b", line: 1, arg: 1, scope: !4, file: !5, type: !8)
!22 = !DILocation(line: 1, scope: !4)
!23 = !DILocation(line: 2, scope: !4)
!24 = !DILocalVariable(name: "a", line: 6, arg: 1, scope: !14, file: !15, type: !12)
!25 = !DILocation(line: 6, scope: !14)
!26 = !DILocalVariable(name: "z", line: 7, scope: !14, file: !15, type: !8)
!27 = !DILocation(line: 7, scope: !14)
!28 = !DILocation(line: 8, scope: !29)
!29 = distinct !DILexicalBlock(line: 8, column: 0, file: !10, scope: !14)
!30 = !DILocation(line: 9, scope: !29)
!31 = !DILocation(line: 10, scope: !32)
!32 = distinct !DILexicalBlock(line: 10, column: 0, file: !10, scope: !14)
!33 = !DILocation(line: 11, scope: !32)
!34 = !DILocation(line: 12, scope: !14)

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
; CHECK-NEXT: .b8 19                               // DW_TAG_structure_type
; CHECK-NEXT: .b8 0                                // DW_CHILDREN_no
; CHECK-NEXT: .b8 3                                // DW_AT_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 60                               // DW_AT_declaration
; CHECK-NEXT: .b8 12                               // DW_FORM_flag
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 3                                // Abbreviation Code
; CHECK-NEXT: .b8 46                               // DW_TAG_subprogram
; CHECK-NEXT: .b8 1                                // DW_CHILDREN_yes
; CHECK-NEXT: .b8 17                               // DW_AT_low_pc
; CHECK-NEXT: .b8 1                                // DW_FORM_addr
; CHECK-NEXT: .b8 18                               // DW_AT_high_pc
; CHECK-NEXT: .b8 1                                // DW_FORM_addr
; CHECK-NEXT: .b8 64                               // DW_AT_frame_base
; CHECK-NEXT: .b8 10                               // DW_FORM_block1
; CHECK-NEXT: .b8 135,64                           // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 3                                // DW_AT_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 58                               // DW_AT_decl_file
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 59                               // DW_AT_decl_line
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 63                               // DW_AT_external
; CHECK-NEXT: .b8 12                               // DW_FORM_flag
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 4                                // Abbreviation Code
; CHECK-NEXT: .b8 52                               // DW_TAG_variable
; CHECK-NEXT: .b8 0                                // DW_CHILDREN_no
; CHECK-NEXT: .b8 3                                // DW_AT_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 58                               // DW_AT_decl_file
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 59                               // DW_AT_decl_line
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 73                               // DW_AT_type
; CHECK-NEXT: .b8 16                               // DW_FORM_ref_addr
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 5                                // Abbreviation Code
; CHECK-NEXT: .b8 46                               // DW_TAG_subprogram
; CHECK-NEXT: .b8 1                                // DW_CHILDREN_yes
; CHECK-NEXT: .b8 17                               // DW_AT_low_pc
; CHECK-NEXT: .b8 1                                // DW_FORM_addr
; CHECK-NEXT: .b8 18                               // DW_AT_high_pc
; CHECK-NEXT: .b8 1                                // DW_FORM_addr
; CHECK-NEXT: .b8 64                               // DW_AT_frame_base
; CHECK-NEXT: .b8 10                               // DW_FORM_block1
; CHECK-NEXT: .b8 135,64                           // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 3                                // DW_AT_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 58                               // DW_AT_decl_file
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 59                               // DW_AT_decl_line
; CHECK-NEXT: .b8 11                               // DW_FORM_data1
; CHECK-NEXT: .b8 73                               // DW_AT_type
; CHECK-NEXT: .b8 19                               // DW_FORM_ref4
; CHECK-NEXT: .b8 63                               // DW_AT_external
; CHECK-NEXT: .b8 12                               // DW_FORM_flag
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 6                                // Abbreviation Code
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
; CHECK-NEXT: .b8 7                                // Abbreviation Code
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
; CHECK-NEXT: .b32 150                             // Length of Unit
; CHECK-NEXT: .b8 2                                // DWARF version number
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 .debug_abbrev                   // Offset Into Abbrev. Section
; CHECK-NEXT: .b8 8                                // Address Size (in bytes)
; CHECK-NEXT: .b8 1                                // Abbrev [1] 0xb:0x8f DW_TAG_compile_unit
; CHECK-NEXT: .b8 99,108,97,110,103,32,118,101,114,115,105,111,110,32,51,46,53,46,48,32,40,50,49,48,52,55,57,41 // DW_AT_producer
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_language
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 100,101,98,117,103,45,108,111,99,45,111,102,102,115,101,116,50,46,99,99 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 .debug_line                     // DW_AT_stmt_list
; CHECK-NEXT: .b8 47,108,108,118,109,95,99,109,97,107,101,95,103,99,99 // DW_AT_comp_dir
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b64 Lfunc_begin1                    // DW_AT_low_pc
; CHECK-NEXT: .b64 Lfunc_end1                      // DW_AT_high_pc
; CHECK-NEXT: .b8 2                                // Abbrev [2] 0x64:0x4 DW_TAG_structure_type
; CHECK-NEXT: .b8 65                               // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_declaration
; CHECK-NEXT: .b8 3                                // Abbrev [3] 0x68:0x31 DW_TAG_subprogram
; CHECK-NEXT: .b64 Lfunc_begin1                    // DW_AT_low_pc
; CHECK-NEXT: .b64 Lfunc_end1                      // DW_AT_high_pc
; CHECK-NEXT: .b8 1                                // DW_AT_frame_base
; CHECK-NEXT: .b8 156
; CHECK-NEXT: .b8 95,90,51,98,97,122,49,65         // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 98,97,122                        // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                // DW_AT_decl_file
; CHECK-NEXT: .b8 6                                // DW_AT_decl_line
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 4                                // Abbrev [4] 0x8b:0xd DW_TAG_variable
; CHECK-NEXT: .b8 122                              // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 2                                // DW_AT_decl_file
; CHECK-NEXT: .b8 7                                // DW_AT_decl_line
; CHECK-NEXT: .b64 .debug_info+302                 // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b32 152                             // Length of Unit
; CHECK-NEXT: .b8 2                                // DWARF version number
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 .debug_abbrev                   // Offset Into Abbrev. Section
; CHECK-NEXT: .b8 8                                // Address Size (in bytes)
; CHECK-NEXT: .b8 1                                // Abbrev [1] 0xb:0x91 DW_TAG_compile_unit
; CHECK-NEXT: .b8 99,108,97,110,103,32,118,101,114,115,105,111,110,32,51,46,53,46,48,32,40,50,49,48,52,55,57,41 // DW_AT_producer
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 4                                // DW_AT_language
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 100,101,98,117,103,45,108,111,99,45,111,102,102,115,101,116,49,46,99,99 // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 .debug_line                     // DW_AT_stmt_list
; CHECK-NEXT: .b8 47,108,108,118,109,95,99,109,97,107,101,95,103,99,99 // DW_AT_comp_dir
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b64 Lfunc_begin0                    // DW_AT_low_pc
; CHECK-NEXT: .b64 Lfunc_end0                      // DW_AT_high_pc
; CHECK-NEXT: .b8 5                                // Abbrev [5] 0x64:0x30 DW_TAG_subprogram
; CHECK-NEXT: .b64 Lfunc_begin0                    // DW_AT_low_pc
; CHECK-NEXT: .b64 Lfunc_end0                      // DW_AT_high_pc
; CHECK-NEXT: .b8 1                                // DW_AT_frame_base
; CHECK-NEXT: .b8 156
; CHECK-NEXT: .b8 95,90,51,98,97,114,105           // DW_AT_MIPS_linkage_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 98,97,114                        // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 1                                // DW_AT_decl_line
; CHECK-NEXT: .b32 148                             // DW_AT_type
; CHECK-NEXT: .b8 1                                // DW_AT_external
; CHECK-NEXT: .b8 6                                // Abbrev [6] 0x8a:0x9 DW_TAG_formal_parameter
; CHECK-NEXT: .b8 98                               // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 1                                // DW_AT_decl_file
; CHECK-NEXT: .b8 1                                // DW_AT_decl_line
; CHECK-NEXT: .b32 148                             // DW_AT_type
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: .b8 7                                // Abbrev [7] 0x94:0x7 DW_TAG_base_type
; CHECK-NEXT: .b8 105,110,116                      // DW_AT_name
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 5                                // DW_AT_encoding
; CHECK-NEXT: .b8 4                                // DW_AT_byte_size
; CHECK-NEXT: .b8 0                                // End Of Children Mark
; CHECK-NEXT: }
; CHECK-NEXT: .section .debug_macinfo
; CHECK-NEXT: {
; CHECK-NEXT: .b8 0                                // End Of Macro List Mark
; CHECK:      }
