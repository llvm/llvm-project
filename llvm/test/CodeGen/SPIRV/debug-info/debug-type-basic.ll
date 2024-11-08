; RUN: llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info --print-after=spirv-nonsemantic-debug-info -O0 -mtriple=spirv64-unknown-unknown %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-MIR
; RUN: llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --check-prefix=CHECK-OPTION
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-MIR-DAG: [[type_void:%[0-9]+\:type]] = OpTypeVoid
; CHECK-MIR-DAG: [[type_i32:%[0-9]+\:type]] = OpTypeInt 32, 0
; CHECK-MIR-DAG: [[encoding_signedchar:%[0-9]+\:iid\(s32\)]] = OpConstantI [[type_i32]], 5
; CHECK-MIR-DAG: [[encoding_float:%[0-9]+\:iid\(s32\)]] = OpConstantI [[type_i32]], 3
; CHECK-MIR-DAG: [[flag_zero:%[0-9]+\:iid\(s32\)]] = OpConstantNull [[type_i32]]
; CHECK-MIR-DAG: [[str_bool:%[0-9]+\:id\(s32\)]] = OpString 1819242338, 0
; CHECK-MIR-DAG: [[size_8bits:%[0-9]+\:iid\(s32\)]] = OpConstantI [[type_i32]], 8
; CHECK-MIR-DAG: [[encoding_boolean:%[0-9]+\:iid\(s32\)]] = OpConstantI [[type_i32]], 2
; CHECK-MIR: OpExtInst [[type_void]], 3, 2, [[str_bool]], [[size_8bits]], [[encoding_boolean]], [[flag_zero]]
; CHECK-MIR-DAG: [[str_int:%[0-9]+\:id\(s32\)]] = OpString 7630441
; CHECK-MIR-DAG: [[size_32bits:%[0-9]+\:iid\(s32\)]] = OpConstantI [[type_i32]], 32
; CHECK-MIR-DAG: [[encoding_signed:%[0-9]+\:iid\(s32\)]] = OpConstantI [[type_i32]], 4
; CHECK-MIR: OpExtInst [[type_void]], 3, 2, [[str_int]], [[size_32bits]], [[encoding_signed]], [[flag_zero]]
; CHECK-MIR-DAG: [[str_short:%[0-9]+\:id\(s32\)]] = OpString 1919903859, 116
; CHECK-MIR-DAG: [[size_16bits:%[0-9]+\:iid\(s32\)]] = OpConstantI [[type_i32]], 16
; CHECK-MIR: OpExtInst [[type_void]], 3, 2, [[str_short]], [[size_16bits]], [[encoding_signed]], [[flag_zero]]
; CHECK-MIR-DAG: [[str_char:%[0-9]+\:id\(s32\)]] = OpString 1918986339, 0
; CHECK-MIR: OpExtInst [[type_void]], 3, 2, [[str_char]], [[size_8bits]], [[encoding_signedchar]], [[flag_zero]]
; CHECK-MIR-DAG: [[str_long:%[0-9]+\:id\(s32\)]] = OpString 1735290732, 0
; CHECK-MIR-DAG: [[size_64bits:%[0-9]+\:iid\(s32\)]] = OpConstantI [[type_i32]], 64
; CHECK-MIR: OpExtInst [[type_void]], 3, 2, [[str_long]], [[size_64bits]], [[encoding_signed]], [[flag_zero]]
; CHECK-MIR-DAG: [[str_uint:%[0-9]+\:id\(s32\)]] = OpString 1769172597, 1684368999, 1953392928, 0
; CHECK-MIR-DAG: [[encoding_unsigned:%[0-9]+\:iid\(s32\)]] = OpConstantI [[type_i32]], 6
; CHECK-MIR: OpExtInst [[type_void]], 3, 2, [[str_uint]], [[size_32bits]], [[encoding_unsigned]], [[flag_zero]]
; CHECK-MIR-DAG: [[str_ushort:%[0-9]+\:id\(s32\)]] = OpString 1769172597, 1684368999, 1869116192, 29810
; CHECK-MIR: OpExtInst [[type_void]], 3, 2, [[str_ushort]], [[size_16bits]], [[encoding_unsigned]], [[flag_zero]]
; CHECK-MIR-DAG: [[str_uchar:%[0-9]+\:id\(s32\)]] = OpString 1769172597, 1684368999, 1634231072, 114
; CHECK-MIR-DAG: [[encoding_unsignedchar:%[0-9]+\:iid\(s32\)]] = OpConstantI [[type_i32]], 7
; CHECK-MIR: OpExtInst [[type_void]], 3, 2, [[str_uchar]], [[size_8bits]], [[encoding_unsignedchar]], [[flag_zero]]
; CHECK-MIR-DAG: [[str_ulong:%[0-9]+\:id\(s32\)]] = OpString 1769172597, 1684368999, 1852795936, 103
; CHECK-MIR: OpExtInst [[type_void]], 3, 2, [[str_ulong]], [[size_64bits]], [[encoding_unsigned]], [[flag_zero]]
; CHECK-MIR-DAG: [[str_float:%[0-9]+\:id\(s32\)]] = OpString 1634692198, 116
; CHECK-MIR: OpExtInst [[type_void]], 3, 2, [[str_float]], [[size_32bits]], [[encoding_float]], [[flag_zero]]
; CHECK-MIR-DAG: [[str_double:%[0-9]+\:id\(s32\)]] = OpString 1651863396, 25964
; CHECK-MIR: OpExtInst [[type_void]], 3, 2, [[str_double]], [[size_64bits]], [[encoding_float]], [[flag_zero]]

; CHECK-SPIRV: [[ext_inst_non_semantic:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-SPIRV-DAG: [[str_bool:%[0-9]+]] = OpString "bool"
; CHECK-SPIRV-DAG: [[str_int:%[0-9]+]] = OpString "int"
; CHECK-SPIRV-DAG: [[str_short:%[0-9]+]] = OpString "short"
; CHECK-SPIRV-DAG: [[str_char:%[0-9]+]] = OpString "char"
; CHECK-SPIRV-DAG: [[str_long:%[0-9]+]] = OpString "long"
; CHECK-SPIRV-DAG: [[str_uint:%[0-9]+]] = OpString "unsigned int"
; CHECK-SPIRV-DAG: [[str_ushort:%[0-9]+]] = OpString "unsigned short"
; CHECK-SPIRV-DAG: [[str_uchar:%[0-9]+]] = OpString "unsigned char"
; CHECK-SPIRV-DAG: [[str_ulong:%[0-9]+]] = OpString "unsigned long"
; CHECK-SPIRV-DAG: [[str_float:%[0-9]+]] = OpString "float"
; CHECK-SPIRV-DAG: [[str_double:%[0-9]+]] = OpString "double"
; CHECK-SPIRV-NOT: ----------------
; CHECK-SPIRV-DAG: [[type_void:%[0-9]+]] = OpTypeVoid
; CHECK-SPIRV-DAG: [[type_float64:%[0-9]+]] = OpTypeFloat 64
; CHECK-SPIRV-DAG: [[type_float32:%[0-9]+]] = OpTypeFloat 32
; CHECK-SPIRV-DAG: [[type_int64:%[0-9]+]] = OpTypeInt 64 0
; CHECK-SPIRV-DAG: [[type_int8:%[0-9]+]] = OpTypeInt 8 0
; CHECK-SPIRV-DAG: [[type_int16:%[0-9]+]] = OpTypeInt 16 0
; CHECK-SPIRV-DAG: [[type_int32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: [[encoding_signedchar:%[0-9]+]] = OpConstant [[type_int32]] 5
; CHECK-SPIRV-DAG: [[flag_zero:%[0-9]+]] = OpConstantNull [[type_int32]]
; CHECK-SPIRV-DAG: [[encoding_float:%[0-9]+]] = OpConstant [[type_int32]] 3
; CHECK-SPIRV-DAG: [[size_8bit:%[0-9]+]] = OpConstant [[type_int32]] 8
; CHECK-SPIRV-DAG: [[encoding_boolean:%[0-9]+]] = OpConstant [[type_int32]] 2
; CHECK-SPIRV-DAG: [[size_32bit:%[0-9]+]] = OpConstant [[type_int32]] 32
; CHECK-SPIRV-DAG: [[encoding_signed:%[0-9]+]] = OpConstant [[type_int32]] 4
; CHECK-SPIRV-DAG: [[size_16bit:%[0-9]+]] = OpConstant [[type_int32]] 16
; CHECK-SPIRV-DAG: [[size_64bit:%[0-9]+]] = OpConstant [[type_int32]] 64
; CHECK-SPIRV-DAG: [[encoding_unsigned:%[0-9]+]] = OpConstant [[type_int32]] 6
; CHECK-SPIRV-DAG: [[encoding_unsignedchar:%[0-9]+]] = OpConstant [[type_int32]] 7
; CHECK-SPIRV-DAG: OpExtInst [[type_void]] [[ext_inst_non_semantic]] DebugTypeBasic [[str_bool]] [[size_8bit]] [[encoding_boolean]] [[flag_zero]]
; CHECK-SPIRV-DAG: OpExtInst [[type_void]] [[ext_inst_non_semantic]] DebugTypeBasic [[str_int]] [[size_32bit]] [[encoding_signed]] [[flag_zero]]
; CHECK-SPIRV-DAG: OpExtInst [[type_void]] [[ext_inst_non_semantic]] DebugTypeBasic [[str_short]] [[size_16bit]] [[encoding_signed]] [[flag_zero]]
; CHECK-SPIRV-DAG: OpExtInst [[type_void]] [[ext_inst_non_semantic]] DebugTypeBasic [[str_char]] [[size_8bit]] [[encoding_signedchar]] [[flag_zero]]
; CHECK-SPIRV-DAG: OpExtInst [[type_void]] [[ext_inst_non_semantic]] DebugTypeBasic [[str_long]] [[size_64bit]] [[encoding_signed]] [[flag_zero]]
; CHECK-SPIRV-DAG: OpExtInst [[type_void]] [[ext_inst_non_semantic]] DebugTypeBasic [[str_uint]] [[size_32bit]] [[encoding_unsigned]] [[flag_zero]]
; CHECK-SPIRV-DAG: OpExtInst [[type_void]] [[ext_inst_non_semantic]] DebugTypeBasic [[str_ushort]] [[size_16bit]] [[encoding_unsigned]] [[flag_zero]]
; CHECK-SPIRV-DAG: OpExtInst [[type_void]] [[ext_inst_non_semantic]] DebugTypeBasic [[str_uchar]] [[size_8bit]] [[encoding_unsignedchar]] [[flag_zero]]
; CHECK-SPIRV-DAG: OpExtInst [[type_void]] [[ext_inst_non_semantic]] DebugTypeBasic [[str_ulong]] [[size_64bit]] [[encoding_unsigned]] [[flag_zero]]
; CHECK-SPIRV-DAG: OpExtInst [[type_void]] [[ext_inst_non_semantic]] DebugTypeBasic [[str_float]] [[size_32bit]] [[encoding_float]] [[flag_zero]]
; CHECK-SPIRV-DAG: OpExtInst [[type_void]] [[ext_inst_non_semantic]] DebugTypeBasic [[str_double]] [[size_64bit]] [[encoding_float]] [[flag_zero]]

; CHECK-OPTION-NOT: DebugTypeBasic

define spir_func void  @test1() !dbg !9 {
entry:
  %b0 = alloca i8, align 1
  %a0 = alloca i32, align 4
  %s0 = alloca i16, align 2
  %c0 = alloca i8, align 1
  %l0 = alloca i64, align 8
  %ua0 = alloca i32, align 4
  %us0 = alloca i16, align 2
  %uc0 = alloca i8, align 1
  %ul0 = alloca i64, align 8
  %f0 = alloca float, align 4
  %d0 = alloca double, align 8
    #dbg_declare(ptr %b0, !14, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !16)
  store i8 0, ptr %b0, align 1, !dbg !16
    #dbg_declare(ptr %a0, !17, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !18)
  store i32 1, ptr %a0, align 4, !dbg !18
    #dbg_declare(ptr %s0, !19, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !21)
  store i16 2, ptr %s0, align 2, !dbg !21
    #dbg_declare(ptr %c0, !22, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !24)
  store i8 3, ptr %c0, align 1, !dbg !24
    #dbg_declare(ptr %l0, !25, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !27)
  store i64 4, ptr %l0, align 8, !dbg !27
    #dbg_declare(ptr %ua0, !28, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !30)
  store i32 1, ptr %ua0, align 4, !dbg !30
    #dbg_declare(ptr %us0, !31, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !33)
  store i16 2, ptr %us0, align 2, !dbg !33
    #dbg_declare(ptr %uc0, !34, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !36)
  store i8 3, ptr %uc0, align 1, !dbg !36
    #dbg_declare(ptr %ul0, !37, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !39)
  store i64 4, ptr %ul0, align 8, !dbg !39
    #dbg_declare(ptr %f0, !40, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !42)
  store float 1.000000e+00, ptr %f0, align 4, !dbg !42
    #dbg_declare(ptr %d0, !43, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !45)
  store double 2.000000e+00, ptr %d0, align 8, !dbg !45
  ret void
}

define spir_func void @test2() !dbg !47 {
entry:
  %b1 = alloca i8, align 1
  %a1 = alloca i32, align 4
  %s1 = alloca i16, align 2
  %c1 = alloca i8, align 1
  %l1 = alloca i64, align 8
  %ua1 = alloca i32, align 4
  %us1 = alloca i16, align 2
  %uc1 = alloca i8, align 1
  %ul1 = alloca i64, align 8
  %f1 = alloca float, align 4
  %d1 = alloca double, align 8
    #dbg_declare(ptr %b1, !48, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !49)
  store i8 0, ptr %b1, align 1, !dbg !49
    #dbg_declare(ptr %a1, !50, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !51)
  store i32 1, ptr %a1, align 4, !dbg !51
    #dbg_declare(ptr %s1, !52, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !53)
  store i16 2, ptr %s1, align 2, !dbg !53
    #dbg_declare(ptr %c1, !54, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !55)
  store i8 3, ptr %c1, align 1, !dbg !55
    #dbg_declare(ptr %l1, !56, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !57)
  store i64 4, ptr %l1, align 8, !dbg !57
    #dbg_declare(ptr %ua1, !58, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !59)
  store i32 1, ptr %ua1, align 4, !dbg !59
    #dbg_declare(ptr %us1, !60, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !61)
  store i16 2, ptr %us1, align 2, !dbg !61
    #dbg_declare(ptr %uc1, !62, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !63)
  store i8 3, ptr %uc1, align 1, !dbg !63
    #dbg_declare(ptr %ul1, !64, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !65)
  store i64 4, ptr %ul1, align 8, !dbg !65
    #dbg_declare(ptr %f1, !66, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !67)
  store float 1.000000e+00, ptr %f1, align 4, !dbg !67
    #dbg_declare(ptr %d1, !68, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !69)
  store double 2.000000e+00, ptr %d1, align 8, !dbg !69
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_Zig, file: !1, producer: "clang version XX.X.XXXX (FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "example.cpp", directory: "/AAAAAAAAAA/BBBBBBBB/CCCCCCCCC", checksumkind: CSK_MD5, checksum: "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"frame-pointer", i32 2}
!9 = distinct !DISubprogram(name: "test1", linkageName: "XXXX", scope: !1, file: !1, line: 1, type: !10, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !13)
!10 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !{}
!14 = !DILocalVariable(name: "b0", scope: !9, file: !1, line: 2, type: !15)
!15 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!16 = !DILocation(line: 2, column: 8, scope: !9)
!17 = !DILocalVariable(name: "a0", scope: !9, file: !1, line: 3, type: !12)
!18 = !DILocation(line: 3, column: 7, scope: !9)
!19 = !DILocalVariable(name: "s0", scope: !9, file: !1, line: 4, type: !20)
!20 = !DIBasicType(name: "short", size: 16, encoding: DW_ATE_signed)
!21 = !DILocation(line: 4, column: 9, scope: !9)
!22 = !DILocalVariable(name: "c0", scope: !9, file: !1, line: 5, type: !23)
!23 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!24 = !DILocation(line: 5, column: 8, scope: !9)
!25 = !DILocalVariable(name: "l0", scope: !9, file: !1, line: 6, type: !26)
!26 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!27 = !DILocation(line: 6, column: 8, scope: !9)
!28 = !DILocalVariable(name: "ua0", scope: !9, file: !1, line: 7, type: !29)
!29 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!30 = !DILocation(line: 7, column: 16, scope: !9)
!31 = !DILocalVariable(name: "us0", scope: !9, file: !1, line: 8, type: !32)
!32 = !DIBasicType(name: "unsigned short", size: 16, encoding: DW_ATE_unsigned)
!33 = !DILocation(line: 8, column: 18, scope: !9)
!34 = !DILocalVariable(name: "uc0", scope: !9, file: !1, line: 9, type: !35)
!35 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
!36 = !DILocation(line: 9, column: 17, scope: !9)
!37 = !DILocalVariable(name: "ul0", scope: !9, file: !1, line: 10, type: !38)
!38 = !DIBasicType(name: "unsigned long", size: 64, encoding: DW_ATE_unsigned)
!39 = !DILocation(line: 10, column: 17, scope: !9)
!40 = !DILocalVariable(name: "f0", scope: !9, file: !1, line: 11, type: !41)
!41 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!42 = !DILocation(line: 11, column: 9, scope: !9)
!43 = !DILocalVariable(name: "d0", scope: !9, file: !1, line: 12, type: !44)
!44 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!45 = !DILocation(line: 12, column: 10, scope: !9)
!46 = !DILocation(line: 13, column: 3, scope: !9)
!47 = distinct !DISubprogram(name: "test2", linkageName: "YYYY", scope: !1, file: !1, line: 16, type: !10, scopeLine: 16, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !13)
!48 = !DILocalVariable(name: "b1", scope: !47, file: !1, line: 17, type: !15)
!49 = !DILocation(line: 17, column: 8, scope: !47)
!50 = !DILocalVariable(name: "a1", scope: !47, file: !1, line: 18, type: !12)
!51 = !DILocation(line: 18, column: 7, scope: !47)
!52 = !DILocalVariable(name: "s1", scope: !47, file: !1, line: 19, type: !20)
!53 = !DILocation(line: 19, column: 9, scope: !47)
!54 = !DILocalVariable(name: "c1", scope: !47, file: !1, line: 20, type: !23)
!55 = !DILocation(line: 20, column: 8, scope: !47)
!56 = !DILocalVariable(name: "l1", scope: !47, file: !1, line: 21, type: !26)
!57 = !DILocation(line: 21, column: 8, scope: !47)
!58 = !DILocalVariable(name: "ua1", scope: !47, file: !1, line: 22, type: !29)
!59 = !DILocation(line: 22, column: 16, scope: !47)
!60 = !DILocalVariable(name: "us1", scope: !47, file: !1, line: 23, type: !32)
!61 = !DILocation(line: 23, column: 18, scope: !47)
!62 = !DILocalVariable(name: "uc1", scope: !47, file: !1, line: 24, type: !35)
!63 = !DILocation(line: 24, column: 17, scope: !47)
!64 = !DILocalVariable(name: "ul1", scope: !47, file: !1, line: 25, type: !38)
!65 = !DILocation(line: 25, column: 17, scope: !47)
!66 = !DILocalVariable(name: "f1", scope: !47, file: !1, line: 26, type: !41)
!67 = !DILocation(line: 26, column: 9, scope: !47)
!68 = !DILocalVariable(name: "d1", scope: !47, file: !1, line: 27, type: !44)
!69 = !DILocation(line: 27, column: 10, scope: !47)
!70 = !DILocation(line: 28, column: 3, scope: !47)
