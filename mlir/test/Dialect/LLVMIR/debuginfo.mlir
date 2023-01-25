// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-DAG: #[[FILE:.*]] = #llvm.di_file<"debuginfo.mlir" in "/test/">
#file = #llvm.di_file<"debuginfo.mlir" in "/test/">

// CHECK-DAG: #[[CU:.*]] = #llvm.di_compile_unit<sourceLanguage = DW_LANG_C, file = #[[FILE]], producer = "MLIR", isOptimized = true, emissionKind = Full>
#cu = #llvm.di_compile_unit<
  sourceLanguage = DW_LANG_C, file = #file, producer = "MLIR",
  isOptimized = true, emissionKind = Full
>

// CHECK-DAG: #[[VOID:.*]] = #llvm.di_void_result_type
#void = #llvm.di_void_result_type

// CHECK-DAG: #[[INT0:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int0">
#int0 = #llvm.di_basic_type<
  // Omit the optional sizeInBits and encoding parameters.
  tag = DW_TAG_base_type, name = "int0"
>

// CHECK-DAG: #[[INT1:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int1", sizeInBits = 32, encoding = DW_ATE_signed>
#int1 = #llvm.di_basic_type<
  tag = DW_TAG_base_type, name = "int1",
  sizeInBits = 32, encoding = DW_ATE_signed
>

// CHECK-DAG: #[[PTR0:.*]] = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #[[INT0]], sizeInBits = 64, alignInBits = 32, offsetInBits = 4>
#ptr0 = #llvm.di_derived_type<
  tag = DW_TAG_pointer_type, baseType = #int0,
  sizeInBits = 64, alignInBits = 32, offsetInBits = 4
>

// CHECK-DAG: #[[PTR1:.*]] = #llvm.di_derived_type<tag = DW_TAG_pointer_type, name = "ptr1", baseType = #[[INT0]]>
#ptr1 = #llvm.di_derived_type<
  // Specify the name parameter.
  tag = DW_TAG_pointer_type, name = "ptr1", baseType = #int0
>

// CHECK-DAG: #[[COMP0:.*]] = #llvm.di_composite_type<tag = DW_TAG_array_type, name = "array0", line = 10, sizeInBits = 128, alignInBits = 32>
#comp0 = #llvm.di_composite_type<
  tag = DW_TAG_array_type, name = "array0",
  line = 10, sizeInBits = 128, alignInBits = 32
>

// CHECK-DAG: #[[COMP1:.*]] = #llvm.di_composite_type<tag = DW_TAG_array_type, name = "array1", file = #[[FILE]], scope = #[[FILE]], baseType = #[[INT0]], elements = #llvm.di_subrange<count = 4 : i64>>
#comp1 = #llvm.di_composite_type<
  tag = DW_TAG_array_type, name = "array1", file = #file,
  scope = #file, baseType = #int0,
  // Specify the subrange count.
  elements = #llvm.di_subrange<count = 4>
>

// CHECK-DAG: #[[COMP2:.*]] = #llvm.di_composite_type<tag = DW_TAG_class_type, name = "class_name", file = #[[FILE]], scope = #[[FILE]], flags = "TypePassByReference|NonTrivial">
#comp2 = #llvm.di_composite_type<
  tag = DW_TAG_class_type, name = "class_name", file = #file, scope = #file,
  flags = "TypePassByReference|NonTrivial"
>

// CHECK-DAG: #[[SPTYPE0:.*]] = #llvm.di_subroutine_type<callingConvention = DW_CC_normal, types = #[[VOID]], #[[INT0]], #[[PTR0]], #[[PTR1]], #[[COMP0:.*]], #[[COMP1:.*]], #[[COMP2:.*]]>
#spType0 = #llvm.di_subroutine_type<
  callingConvention = DW_CC_normal, types = #void, #int0, #ptr0, #ptr1, #comp0, #comp1, #comp2
>

// CHECK-DAG: #[[SPTYPE1:.*]] = #llvm.di_subroutine_type<types = #[[INT1]], #[[INT1]]>
#spType1 = #llvm.di_subroutine_type<
  // Omit the optional callingConvention parameter.
  types = #int1, #int1
>

// CHECK-DAG: #[[SPTYPE2:.*]] = #llvm.di_subroutine_type<callingConvention = DW_CC_normal>
#spType2 = #llvm.di_subroutine_type<
  // Omit the optional types parameter array.
  callingConvention = DW_CC_normal
>

// CHECK-DAG: #[[SP0:.*]] = #llvm.di_subprogram<compileUnit = #[[CU]], scope = #[[FILE]], name = "addr", linkageName = "addr", file = #[[FILE]], line = 3, scopeLine = 3, subprogramFlags = "Definition|Optimized", type = #[[SPTYPE0]]>
#sp0 = #llvm.di_subprogram<
  compileUnit = #cu, scope = #file, name = "addr", linkageName = "addr",
  file = #file, line = 3, scopeLine = 3, subprogramFlags = "Definition|Optimized", type = #spType0
>

// CHECK-DAG: #[[SP1:.*]] = #llvm.di_subprogram<compileUnit = #[[CU]], scope = #[[COMP2]], name = "value", file = #[[FILE]], subprogramFlags = Definition, type = #[[SPTYPE1]]>
#sp1 = #llvm.di_subprogram<
  // Omit the optional linkageName parameter.
  compileUnit = #cu, scope = #comp2, name = "value",
  file = #file, subprogramFlags = "Definition", type = #spType1
>

// CHECK-DAG: #[[SP2:.*]] = #llvm.di_subprogram<compileUnit = #[[CU]], scope = #[[FILE]], name = "value", file = #[[FILE]], subprogramFlags = Definition, type = #[[SPTYPE2]]>
#sp2 = #llvm.di_subprogram<
  // Omit the optional linkageName parameter.
  compileUnit = #cu, scope = #file, name = "value",
  file = #file, subprogramFlags = "Definition", type = #spType2
>

// CHECK-DAG: #[[BLOCK0:.*]] = #llvm.di_lexical_block<scope = #[[SP0]], line = 1, column = 2>
#block0 = #llvm.di_lexical_block<scope = #sp0, line = 1, column = 2>

// CHECK-DAG: #[[BLOCK1:.*]] = #llvm.di_lexical_block<scope = #[[SP1]]>
#block1 = #llvm.di_lexical_block<scope = #sp1>

// CHECK-DAG: #[[BLOCK2:.*]] = #llvm.di_lexical_block<scope = #[[SP2]]>
#block2 = #llvm.di_lexical_block<scope = #sp2>

// CHECK-DAG: #[[VAR0:.*]] = #llvm.di_local_variable<scope = #[[BLOCK0]], name = "alloc", file = #[[FILE]], line = 6, arg = 1, alignInBits = 32, type = #[[INT0]]>
#var0 = #llvm.di_local_variable<
  scope = #block0, name = "alloc", file = #file,
  line = 6, arg = 1, alignInBits = 32, type = #int0
>

// CHECK-DAG: #[[VAR1:.*]] = #llvm.di_local_variable<scope = #[[BLOCK1]], name = "arg1">
#var1 = #llvm.di_local_variable<
  // Omit the optional parameters.
  scope = #block1, name = "arg1"
>

// CHECK-DAG: #[[VAR2:.*]] = #llvm.di_local_variable<scope = #[[BLOCK2]], name = "arg2">
#var2 = #llvm.di_local_variable<
  // Omit the optional parameters.
  scope = #block2, name = "arg2"
>

// CHECK: llvm.func @addr(%[[ARG:.*]]: i64)
llvm.func @addr(%arg: i64) {
  // CHECK: %[[ALLOC:.*]] = llvm.alloca
  %allocCount = llvm.mlir.constant(1 : i32) : i32
  %alloc = llvm.alloca %allocCount x i64 : (i32) -> !llvm.ptr<i64>

  // CHECK: llvm.intr.dbg.addr #[[VAR0]] = %[[ALLOC]]
  // CHECK: llvm.intr.dbg.declare #[[VAR0]] = %[[ALLOC]]
  llvm.intr.dbg.addr #var0 = %alloc : !llvm.ptr<i64>
  llvm.intr.dbg.declare #var0 = %alloc : !llvm.ptr<i64>
  llvm.return
}

// CHECK: llvm.func @value(%[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32)
llvm.func @value(%arg1: i32, %arg2: i32) {
  // CHECK: llvm.intr.dbg.value #[[VAR1]] = %[[ARG1]]
  llvm.intr.dbg.value #var1 = %arg1 : i32
  // CHECK: llvm.intr.dbg.value #[[VAR2]] = %[[ARG2]]
  llvm.intr.dbg.value #var2 = %arg2 : i32
  llvm.return
}
