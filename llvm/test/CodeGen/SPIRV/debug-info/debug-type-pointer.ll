; RUN: llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info --print-after=spirv-nonsemantic-debug-info -O0 -mtriple=spirv64-unknown-unknown %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-MIR
; RUN: llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --check-prefix=CHECK-OPTION
; TODO(#109287): When type is void * the spirv-val raises an error when DebugInfoNone is set as <id> Base Type argument of DebugTypePointer.   
; DISABLED: %if spirv-tools %{ llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-MIR-DAG:   [[i32type:%[0-9]+\:type]] = OpTypeInt 32, 0
; CHECK-MIR-DAG:   [[void_type:%[0-9]+\:type\(s64\)]] = OpTypeVoid
; CHECK-MIR-DAG:   [[i32_8:%[0-9]+\:iid]] = OpConstantI [[i32type]], 8
; CHECK-MIR-DAG:   [[i32_0:%[0-9]+\:iid]] = OpConstantI [[i32type]], 0
; CHECK-MIR-DAG:   [[i32_5:%[0-9]+\:iid\(s32\)]] = OpConstantI [[i32type]], 5
; CHECK-MIR-DAG:   [[enc_float:%[0-9]+\:iid\(s32\)]] = OpConstantI [[i32type]], 3
; CHECK-MIR-DAG:   [[enc_boolean:%[0-9]+\:iid\(s32\)]] = OpConstantI [[i32type]], 2
; CHECK-MIR-DAG:   [[bool:%[0-9]+\:id\(s32\)]] = OpExtInst [[void_type]], 3, 2, {{%[0-9]+\:[a-z0-9\(\)]+}}, [[i32_8]], [[enc_boolean]], [[i32_0]]
; CHECK-MIR-DAG:   [[i32_16:%[0-9]+\:iid\(s32\)]] = OpConstantI [[i32type]], 16
; CHECK-MIR-DAG:   [[enc_signed:%[0-9]+\:iid\(s32\)]] = OpConstantI [[i32type]], 4
; CHECK-MIR-DAG:   [[short:%[0-9]+\:id\(s32\)]] = OpExtInst [[void_type]], 3, 2, {{%[0-9]+\:[a-z0-9\(\)]+}}, [[i32_16]], [[enc_signed]], [[i32_0]]
; CHECK-MIR-DAG:   [[char:%[0-9]+\:id\(s32\)]] = OpExtInst [[void_type]], 3, 2, {{%[0-9]+\:[a-z0-9\(\)]+}}, [[i32_8]], [[i32_5]], [[i32_0]]
; CHECK-MIR-DAG:   [[i32_64:%[0-9]+\:iid\(s32\)]] = OpConstantI [[i32type]], 64
; CHECK-MIR-DAG:   [[long:%[0-9]+\:id\(s32\)]] = OpExtInst [[void_type]], 3, 2, {{%[0-9]+\:[a-z0-9\(\)]+}}, [[i32_64]], [[enc_signed]], [[i32_0]]
; CHECK-MIR-DAG:   [[i32_32:%[0-9]+\:iid\(s32\)]] = OpConstantI [[i32type]], 32
; CHECK-MIR-DAG:   [[enc_unsigned:%[0-9]+\:iid\(s32\)]] = OpConstantI [[i32type]], 6
; CHECK-MIR-DAG:   [[unsigned_int:%[0-9]+\:id\(s32\)]] = OpExtInst [[void_type]], 3, 2, {{%[0-9]+\:[a-z0-9\(\)]+}}, [[i32_32]], [[enc_unsigned]], [[i32_0]]
; CHECK-MIR-DAG:   [[unsigned_short:%[0-9]+\:id\(s32\)]] = OpExtInst [[void_type]], 3, 2, {{%[0-9]+\:[a-z0-9\(\)]+}}, [[i32_16]], [[enc_unsigned]], [[i32_0]]
; CHECK-MIR-DAG:   [[enc_unsigned_char:%[0-9]+\:iid\(s32\)]] = OpConstantI [[i32type]], 7
; CHECK-MIR-DAG:   [[unsigned_char:%[0-9]+\:id\(s32\)]] = OpExtInst [[void_type]], 3, 2, {{%[0-9]+\:[a-z0-9\(\)]+}}, [[i32_8]], [[enc_unsigned_char]], [[i32_0]]
; CHECK-MIR-DAG:   [[unsigned_long:%[0-9]+\:id\(s32\)]] = OpExtInst [[void_type]], 3, 2, {{%[0-9]+\:[a-z0-9\(\)]+}}, [[i32_64]], [[enc_unsigned]], [[i32_0]]
; CHECK-MIR-DAG:   [[float:%[0-9]+\:id\(s32\)]] = OpExtInst [[void_type]], 3, 2, {{%[0-9]+\:[a-z0-9\(\)]+}}, [[i32_32]], [[enc_float]], [[i32_0]]
; CHECK-MIR-DAG:   [[double:%[0-9]+\:id\(s32\)]] = OpExtInst [[void_type]], 3, 2, {{%[0-9]+\:[a-z0-9\(\)]+}}, [[i32_64]], [[enc_float]], [[i32_0]]
; CHECK-MIR-DAG:   [[int:%[0-9]+\:id\(s32\)]] = OpExtInst [[void_type]], 3, 2, {{%[0-9]+\:[a-z0-9\(\)]+}}, [[i32_32]], [[enc_signed]], [[i32_0]]
; CHECK-MIR-DAG:   OpExtInst [[void_type]], 3, 3, [[bool]], [[i32_8]], [[i32_0]]
; CHECK-MIR-DAG:   OpExtInst [[void_type]], 3, 3, [[short]], [[i32_8]], [[i32_0]]
; CHECK-MIR-DAG:   OpExtInst [[void_type]], 3, 3, [[char]], [[i32_8]], [[i32_0]]
; CHECK-MIR-DAG:   OpExtInst [[void_type]], 3, 3, [[long]], [[i32_8]], [[i32_0]]
; CHECK-MIR-DAG:   OpExtInst [[void_type]], 3, 3, [[unsigned_int]], [[i32_8]], [[i32_0]]
; CHECK-MIR-DAG:   OpExtInst [[void_type]], 3, 3, [[unsigned_short]], [[i32_8]], [[i32_0]]
; CHECK-MIR-DAG:   OpExtInst [[void_type]], 3, 3, [[unsigned_char]], [[i32_8]], [[i32_0]]
; CHECK-MIR-DAG:   OpExtInst [[void_type]], 3, 3, [[unsigned_long]], [[i32_8]], [[i32_0]]
; CHECK-MIR-DAG:   OpExtInst [[void_type]], 3, 3, [[float]], [[i32_8]], [[i32_0]]
; CHECK-MIR-DAG:   OpExtInst [[void_type]], 3, 3, [[double]], [[i32_8]], [[i32_0]]
; CHECK-MIR-DAG:   OpExtInst [[void_type]], 3, 3, [[int]], [[i32_5]], [[i32_0]]
; CHECK-MIR-DAG:   [[debug_info_none:%[0-9]+\:id\(s32\)]] = OpExtInst [[void_type]], 3, 0
; CHECK-MIR-DAG:   OpExtInst [[void_type]], 3, 3, [[debug_info_none]], [[i32_5]], [[i32_0]]

; CHECK-SPIRV:	[[i32type:%[0-9]+]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG:	[[i32_8:%[0-9]+]] = OpConstant [[i32type]] 8
; CHECK-SPIRV-DAG:	[[i32_0:%[0-9]+]] = OpConstant [[i32type]] 0
; CHECK-SPIRV-DAG:	[[i32_5:%[0-9]+]] = OpConstant [[i32type]] 5
; CHECK-SPIRV-DAG:	[[enc_float:%[0-9]+]] = OpConstant [[i32type]] 3
; CHECK-SPIRV-DAG:	[[enc_boolean:%[0-9]+]] = OpConstant [[i32type]] 2
; CHECK-SPIRV-DAG:	[[i32_16:%[0-9]+]] = OpConstant [[i32type]] 16
; CHECK-SPIRV-DAG:	[[enc_signed:%[0-9]+]] = OpConstant [[i32type]] 4
; CHECK-SPIRV-DAG:	[[i32_64:%[0-9]+]] = OpConstant [[i32type]] 64
; CHECK-SPIRV-DAG:	[[i32_32:%[0-9]+]] = OpConstant [[i32type]] 32
; CHECK-SPIRV-DAG:	[[enc_unsigned:%[0-9]+]] = OpConstant [[i32type]] 6
; CHECK-SPIRV-DAG:	[[enc_unsigned_char:%[0-9]+]] = OpConstant [[i32type]] 7
; CHECK-SPIRV-DAG:	[[bool:%[0-9]+]] = OpExtInst {{%[0-9]+ %[0-9]+}} DebugTypeBasic {{%[0-9]+}} [[i32_8]] [[enc_boolean]] [[i32_0]]
; CHECK-SPIRV-DAG:	[[short:%[0-9]+]] = OpExtInst {{%[0-9]+ %[0-9]+}} DebugTypeBasic {{%[0-9]+}} [[i32_16]] [[enc_signed]] [[i32_0]]
; CHECK-SPIRV-DAG:	[[char:%[0-9]+]] = OpExtInst {{%[0-9]+ %[0-9]+}} DebugTypeBasic {{%[0-9]+}} [[i32_8]] [[i32_5]] [[i32_0]]
; CHECK-SPIRV-DAG:	[[long:%[0-9]+]] = OpExtInst {{%[0-9]+ %[0-9]+}} DebugTypeBasic {{%[0-9]+}} [[i32_64]] [[enc_signed]] [[i32_0]]
; CHECK-SPIRV-DAG:	[[unsigned_int:%[0-9]+]] = OpExtInst {{%[0-9]+ %[0-9]+}} DebugTypeBasic {{%[0-9]+}} [[i32_32]] [[enc_unsigned]] [[i32_0]]
; CHECK-SPIRV-DAG:	[[unsigned_short:%[0-9]+]] = OpExtInst {{%[0-9]+ %[0-9]+}} DebugTypeBasic {{%[0-9]+}} [[i32_16]] [[enc_unsigned]] [[i32_0]]
; CHECK-SPIRV-DAG:	[[unsigned_char:%[0-9]+]] = OpExtInst {{%[0-9]+ %[0-9]+}} DebugTypeBasic {{%[0-9]+}} [[i32_8]] [[enc_unsigned_char]] [[i32_0]]
; CHECK-SPIRV-DAG:	[[unsigned_long:%[0-9]+]] = OpExtInst {{%[0-9]+ %[0-9]+}} DebugTypeBasic {{%[0-9]+}} [[i32_64]] [[enc_unsigned]] [[i32_0]]
; CHECK-SPIRV-DAG:	[[float:%[0-9]+]] = OpExtInst {{%[0-9]+ %[0-9]+}} DebugTypeBasic {{%[0-9]+}} [[i32_32]] [[enc_float]] [[i32_0]]
; CHECK-SPIRV-DAG:	[[double:%[0-9]+]] = OpExtInst {{%[0-9]+ %[0-9]+}} DebugTypeBasic {{%[0-9]+}} [[i32_64]] [[enc_float]] [[i32_0]]
; CHECK-SPIRV-DAG:	[[int:%[0-9]+]] = OpExtInst {{%[0-9]+ %[0-9]+}} DebugTypeBasic {{%[0-9]+}} [[i32_32]] [[enc_signed]] [[i32_0]]
; CHECK-SPIRV-DAG:	OpExtInst {{%[0-9]+ %[0-9]+}} DebugTypePointer [[bool]] [[i32_8]] [[i32_0]]
; CHECK-SPIRV-DAG:	OpExtInst {{%[0-9]+ %[0-9]+}} DebugTypePointer [[short]] [[i32_8]] [[i32_0]]
; CHECK-SPIRV-DAG:	OpExtInst {{%[0-9]+ %[0-9]+}} DebugTypePointer [[char]] [[i32_8]] [[i32_0]]
; CHECK-SPIRV-DAG:	OpExtInst {{%[0-9]+ %[0-9]+}} DebugTypePointer [[long]] [[i32_8]] [[i32_0]]
; CHECK-SPIRV-DAG:	OpExtInst {{%[0-9]+ %[0-9]+}} DebugTypePointer [[unsigned_int]] [[i32_8]] [[i32_0]]
; CHECK-SPIRV-DAG:	OpExtInst {{%[0-9]+ %[0-9]+}} DebugTypePointer [[unsigned_short]] [[i32_8]] [[i32_0]]
; CHECK-SPIRV-DAG:	OpExtInst {{%[0-9]+ %[0-9]+}} DebugTypePointer [[unsigned_char]] [[i32_8]] [[i32_0]]
; CHECK-SPIRV-DAG:	OpExtInst {{%[0-9]+ %[0-9]+}} DebugTypePointer [[unsigned_long]] [[i32_8]] [[i32_0]]
; CHECK-SPIRV-DAG:	OpExtInst {{%[0-9]+ %[0-9]+}} DebugTypePointer [[float]] [[i32_8]] [[i32_0]]
; CHECK-SPIRV-DAG:	OpExtInst {{%[0-9]+ %[0-9]+}} DebugTypePointer [[double]] [[i32_8]] [[i32_0]]
; CHECK-SPIRV-DAG:	OpExtInst {{%[0-9]+ %[0-9]+}} DebugTypePointer [[int]] [[i32_5]] [[i32_0]]
; CHECK-SPIRV-DAG:	[[debug_info_none:%[0-9]+]] = OpExtInst {{%[0-9]+ %[0-9]+}} DebugInfoNone
; CHECK-SPIRV-DAG:	OpExtInst {{%[0-9]+ %[0-9]+}} DebugTypePointer [[debug_info_none]] [[i32_5]] [[i32_0]]

; CHECK-OPTION-NOT: DebugTypePointer

@gi0 = dso_local addrspace(1) global ptr addrspace(4) null, align 4, !dbg !0
@gv0 = dso_local addrspace(1) global ptr addrspace(4) null, align 4, !dbg !5

define spir_func i32 @test0() !dbg !17 {
  %1 = alloca ptr addrspace(4), align 4
  %2 = alloca ptr addrspace(4), align 4
  %3 = alloca ptr addrspace(4), align 4
  %4 = alloca ptr addrspace(4), align 4
  %5 = alloca ptr addrspace(4), align 4
  %6 = alloca ptr addrspace(4), align 4
  %7 = alloca ptr addrspace(4), align 4
  %8 = alloca ptr addrspace(4), align 4
  %9 = alloca ptr addrspace(4), align 4
  %10 = alloca ptr addrspace(4), align 4
  %11 = alloca ptr addrspace(4), align 4
  %12 = alloca ptr addrspace(4), align 4
  %13 = alloca [8 x i32], align 4
    #dbg_declare(ptr %1, !21, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !24)
  store ptr addrspace(4) null, ptr %1, align 4, !dbg !24
    #dbg_declare(ptr %2, !25, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !28)
  store ptr addrspace(4) null, ptr %2, align 4, !dbg !28
    #dbg_declare(ptr %3, !29, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !32)
  store ptr addrspace(4) null, ptr %3, align 4, !dbg !32
    #dbg_declare(ptr %4, !33, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !36)
  store ptr addrspace(4) null, ptr %4, align 4, !dbg !36
    #dbg_declare(ptr %5, !37, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !40)
  store ptr addrspace(4) null, ptr %5, align 4, !dbg !40
    #dbg_declare(ptr %6, !41, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !44)
  store ptr addrspace(4) null, ptr %6, align 4, !dbg !44
    #dbg_declare(ptr %7, !45, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !48)
  store ptr addrspace(4) null, ptr %7, align 4, !dbg !48
    #dbg_declare(ptr %8, !49, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !52)
  store ptr addrspace(4) null, ptr %8, align 4, !dbg !52
    #dbg_declare(ptr %9, !53, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !56)
  store ptr addrspace(4) null, ptr %9, align 4, !dbg !56
    #dbg_declare(ptr %10, !57, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !60)
  store ptr addrspace(4) null, ptr %10, align 4, !dbg !60
    #dbg_declare(ptr %11, !61, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !62)
  store ptr addrspace(4) null, ptr %11, align 4, !dbg !62
    #dbg_declare(ptr %12, !63, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !64)
  %14 = load ptr addrspace(4), ptr %11, align 4, !dbg !65
  store ptr addrspace(4) %14, ptr %12, align 4, !dbg !64
    #dbg_declare(ptr %13, !66, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !70)
  ret i32 0, !dbg !71
}

define spir_func i32 @test1() !dbg !72 {
  %1 = alloca ptr addrspace(4), align 4
  %2 = alloca ptr addrspace(4), align 4
  %3 = alloca ptr addrspace(4), align 4
  %4 = alloca ptr addrspace(4), align 4
  %5 = alloca ptr addrspace(4), align 4
  %6 = alloca ptr addrspace(4), align 4
  %7 = alloca ptr addrspace(4), align 4
  %8 = alloca ptr addrspace(4), align 4
  %9 = alloca ptr addrspace(4), align 4
  %10 = alloca ptr addrspace(4), align 4
  %11 = alloca ptr addrspace(4), align 4
  %12 = alloca ptr addrspace(4), align 4
  %13 = alloca [8 x i32], align 4
    #dbg_declare(ptr %1, !73, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !74)
  store ptr addrspace(4) null, ptr %1, align 4, !dbg !74
    #dbg_declare(ptr %2, !75, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !76)
  store ptr addrspace(4) null, ptr %2, align 4, !dbg !76
    #dbg_declare(ptr %3, !77, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !78)
  store ptr addrspace(4) null, ptr %3, align 4, !dbg !78
    #dbg_declare(ptr %4, !79, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !80)
  store ptr addrspace(4) null, ptr %4, align 4, !dbg !80
    #dbg_declare(ptr %5, !81, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !82)
  store ptr addrspace(4) null, ptr %5, align 4, !dbg !82
    #dbg_declare(ptr %6, !83, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !84)
  store ptr addrspace(4) null, ptr %6, align 4, !dbg !84
    #dbg_declare(ptr %7, !85, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !86)
  store ptr addrspace(4) null, ptr %7, align 4, !dbg !86
    #dbg_declare(ptr %8, !87, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !88)
  store ptr addrspace(4) null, ptr %8, align 4, !dbg !88
    #dbg_declare(ptr %9, !89, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !90)
  store ptr addrspace(4) null, ptr %9, align 4, !dbg !90
    #dbg_declare(ptr %10, !91, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !92)
  store ptr addrspace(4) null, ptr %10, align 4, !dbg !92
    #dbg_declare(ptr %11, !93, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !94)
  store ptr addrspace(4) null, ptr %11, align 4, !dbg !94
    #dbg_declare(ptr %12, !95, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !96)
  %14 = load ptr addrspace(4), ptr %11, align 4, !dbg !97
  store ptr addrspace(4) %14, ptr %12, align 4, !dbg !96
    #dbg_declare(ptr %13, !98, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !99)
  ret i32 0, !dbg !100
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10, !11, !12, !13}
!opencl.ocl.version = !{!14}
!opencl.cxx.version = !{!15}
!opencl.spir.version = !{!14}
!llvm.ident = !{!16}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression(DW_OP_constu, 1, DW_OP_swap, DW_OP_xderef))
!1 = distinct !DIGlobalVariable(name: "gi0", scope: !2, file: !3, line: 1, type: !8, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version XX.X.XXXX (FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "example.cpp", directory: "/AAAAAAAAAA/BBBBBBBB/CCCCCCCCC", checksumkind: CSK_MD5, checksum: "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
!4 = !{!0, !5}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression(DW_OP_constu, 1, DW_OP_swap, DW_OP_xderef))
!6 = distinct !DIGlobalVariable(name: "gv0", scope: !2, file: !3, line: 3, type: !7, isLocal: false, isDefinition: true)
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 32, dwarfAddressSpace: 1)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 32, dwarfAddressSpace: 1)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !{i32 7, !"Dwarf Version", i32 5}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{i32 7, !"frame-pointer", i32 2}
!14 = !{i32 2, i32 0}
!15 = !{i32 1, i32 0}
!16 = !{!"clang version 20.0.0git (https://github.com/bwlodarcz/llvm-project de1f5b96adcea52bf7c9670c46123fe1197050d2)"}
!17 = distinct !DISubprogram(name: "test0", linkageName: "test0", scope: !3, file: !3, line: 5, type: !18, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !20)
!18 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !19)
!19 = !{!9}
!20 = !{}
!21 = !DILocalVariable(name: "bp0", scope: !17, file: !3, line: 6, type: !22)
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !23, size: 32, dwarfAddressSpace: 4)
!23 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!24 = !DILocation(line: 6, column: 9, scope: !17)
!25 = !DILocalVariable(name: "sp0", scope: !17, file: !3, line: 7, type: !26)
!26 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !27, size: 32, dwarfAddressSpace: 4)
!27 = !DIBasicType(name: "short", size: 16, encoding: DW_ATE_signed)
!28 = !DILocation(line: 7, column: 10, scope: !17)
!29 = !DILocalVariable(name: "cp0", scope: !17, file: !3, line: 8, type: !30)
!30 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !31, size: 32, dwarfAddressSpace: 4)
!31 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!32 = !DILocation(line: 8, column: 9, scope: !17)
!33 = !DILocalVariable(name: "lp0", scope: !17, file: !3, line: 9, type: !34)
!34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !35, size: 32, dwarfAddressSpace: 4)
!35 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!36 = !DILocation(line: 9, column: 9, scope: !17)
!37 = !DILocalVariable(name: "uip0", scope: !17, file: !3, line: 10, type: !38)
!38 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !39, size: 32, dwarfAddressSpace: 4)
!39 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!40 = !DILocation(line: 10, column: 17, scope: !17)
!41 = !DILocalVariable(name: "usp0", scope: !17, file: !3, line: 11, type: !42)
!42 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !43, size: 32, dwarfAddressSpace: 4)
!43 = !DIBasicType(name: "unsigned short", size: 16, encoding: DW_ATE_unsigned)
!44 = !DILocation(line: 11, column: 19, scope: !17)
!45 = !DILocalVariable(name: "ucp0", scope: !17, file: !3, line: 12, type: !46)
!46 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !47, size: 32, dwarfAddressSpace: 4)
!47 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
!48 = !DILocation(line: 12, column: 18, scope: !17)
!49 = !DILocalVariable(name: "ulp0", scope: !17, file: !3, line: 13, type: !50)
!50 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !51, size: 32, dwarfAddressSpace: 4)
!51 = !DIBasicType(name: "unsigned long", size: 64, encoding: DW_ATE_unsigned)
!52 = !DILocation(line: 13, column: 18, scope: !17)
!53 = !DILocalVariable(name: "fp0", scope: !17, file: !3, line: 14, type: !54)
!54 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !55, size: 32, dwarfAddressSpace: 4)
!55 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!56 = !DILocation(line: 14, column: 10, scope: !17)
!57 = !DILocalVariable(name: "dp0", scope: !17, file: !3, line: 15, type: !58)
!58 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !59, size: 32, dwarfAddressSpace: 4)
!59 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!60 = !DILocation(line: 15, column: 11, scope: !17)
!61 = !DILocalVariable(name: "ip0", scope: !17, file: !3, line: 16, type: !8)
!62 = !DILocation(line: 16, column: 8, scope: !17)
!63 = !DILocalVariable(name: "addr0", scope: !17, file: !3, line: 17, type: !7)
!64 = !DILocation(line: 17, column: 9, scope: !17)
!65 = !DILocation(line: 17, column: 17, scope: !17)
!66 = !DILocalVariable(name: "arr0", scope: !17, file: !3, line: 18, type: !67)
!67 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 256, elements: !68)
!68 = !{!69}
!69 = !DISubrange(count: 8)
!70 = !DILocation(line: 18, column: 7, scope: !17)
!71 = !DILocation(line: 19, column: 3, scope: !17)
!72 = distinct !DISubprogram(name: "test1", linkageName: "test1", scope: !3, file: !3, line: 22, type: !18, scopeLine: 22, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !20)
!73 = !DILocalVariable(name: "bp1", scope: !72, file: !3, line: 23, type: !22)
!74 = !DILocation(line: 23, column: 9, scope: !72)
!75 = !DILocalVariable(name: "sp1", scope: !72, file: !3, line: 24, type: !26)
!76 = !DILocation(line: 24, column: 10, scope: !72)
!77 = !DILocalVariable(name: "cp1", scope: !72, file: !3, line: 25, type: !30)
!78 = !DILocation(line: 25, column: 9, scope: !72)
!79 = !DILocalVariable(name: "lp1", scope: !72, file: !3, line: 26, type: !34)
!80 = !DILocation(line: 26, column: 9, scope: !72)
!81 = !DILocalVariable(name: "uip1", scope: !72, file: !3, line: 27, type: !38)
!82 = !DILocation(line: 27, column: 17, scope: !72)
!83 = !DILocalVariable(name: "usp1", scope: !72, file: !3, line: 28, type: !42)
!84 = !DILocation(line: 28, column: 19, scope: !72)
!85 = !DILocalVariable(name: "ucp1", scope: !72, file: !3, line: 29, type: !46)
!86 = !DILocation(line: 29, column: 18, scope: !72)
!87 = !DILocalVariable(name: "ulp1", scope: !72, file: !3, line: 30, type: !50)
!88 = !DILocation(line: 30, column: 18, scope: !72)
!89 = !DILocalVariable(name: "fp1", scope: !72, file: !3, line: 31, type: !54)
!90 = !DILocation(line: 31, column: 10, scope: !72)
!91 = !DILocalVariable(name: "dp1", scope: !72, file: !3, line: 32, type: !58)
!92 = !DILocation(line: 32, column: 11, scope: !72)
!93 = !DILocalVariable(name: "ip1", scope: !72, file: !3, line: 33, type: !8)
!94 = !DILocation(line: 33, column: 8, scope: !72)
!95 = !DILocalVariable(name: "addr1", scope: !72, file: !3, line: 34, type: !7)
!96 = !DILocation(line: 34, column: 9, scope: !72)
!97 = !DILocation(line: 34, column: 17, scope: !72)
!98 = !DILocalVariable(name: "arr1", scope: !72, file: !3, line: 35, type: !67)
!99 = !DILocation(line: 35, column: 7, scope: !72)
!100 = !DILocation(line: 36, column: 3, scope: !72)
