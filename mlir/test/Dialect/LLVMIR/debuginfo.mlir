// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK: #[[TYPE:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "si64", sizeInBits = 0, encoding = DW_ATE_signed>
#si64 = #llvm.di_basic_type<
  tag = DW_TAG_base_type, name = "si64", sizeInBits = 0,
  encoding = DW_ATE_signed
>

// CHECK: #[[FILE:.*]] = #llvm.di_file<"debuginfo.mlir" in "/test/">
#file = #llvm.di_file<"debuginfo.mlir" in "/test/">

// CHECK: #[[CU:.*]] = #llvm.di_compile_unit<sourceLanguage = DW_LANG_C, file = #[[FILE]], producer = "MLIR", isOptimized = true, emissionKind = Full>
#cu = #llvm.di_compile_unit<
  sourceLanguage = DW_LANG_C, file = #file, producer = "MLIR",
  isOptimized = true, emissionKind = Full
>

// CHECK: #[[SPTYPE:.*]] = #llvm.di_subroutine_type<callingConvention = DW_CC_normal, types = #[[TYPE]]>
#spType = #llvm.di_subroutine_type<callingConvention = DW_CC_normal, types = #si64>

// CHECK: #[[SP:.*]] = #llvm.di_subprogram<compileUnit = #[[CU]], scope = #[[FILE]], name = "intrinsics", linkageName = "intrinsics", file = #[[FILE]], line = 3, scopeLine = 3, subprogramFlags = "Definition|Optimized", type = #[[SPTYPE]]>
#sp = #llvm.di_subprogram<
  compileUnit = #cu, scope = #file, name = "intrinsics", linkageName = "intrinsics",
  file = #file, line = 3, scopeLine = 3, subprogramFlags = "Definition|Optimized", type = #spType
>

// CHECK: #[[VAR:.*]] = #llvm.di_local_variable<scope = #[[SP]], name = "arg", file = #[[FILE]], line = 6, arg = 1, alignInBits = 0, type = #[[TYPE]]>
#variable = #llvm.di_local_variable<scope = #sp, name = "arg", file = #file, line = 6, arg = 1, alignInBits = 0, type = #si64>

// CHECK: llvm.func @intrinsics(%[[ARG:.*]]: i64)
llvm.func @intrinsics(%arg: i64) {
  // CHECK: %[[ALLOC:.*]] = llvm.alloca 
  %allocCount = llvm.mlir.constant(1 : i32) : i32
  %alloc = llvm.alloca %allocCount x i64 : (i32) -> !llvm.ptr<i64>

  // CHECK: llvm.dbg.value #[[VAR]] = %[[ARG]]
  // CHECK: llvm.dbg.addr #[[VAR]] = %[[ALLOC]]
  // CHECK: llvm.dbg.declare #[[VAR]] = %[[ALLOC]]
  llvm.dbg.value #variable = %arg : i64
  llvm.dbg.addr #variable = %alloc : !llvm.ptr<i64>
  llvm.dbg.declare #variable = %alloc : !llvm.ptr<i64>
  llvm.return
}
