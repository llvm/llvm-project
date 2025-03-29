// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-DAG: #[[FILE:.*]] = #llvm.di_file<"debuginfo.mlir" in "/test/">
#file = #llvm.di_file<"debuginfo.mlir" in "/test/">

// CHECK-DAG: #[[CU:.*]] = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = #[[FILE]], producer = "MLIR", isOptimized = true, emissionKind = Full>
#cu = #llvm.di_compile_unit<
  id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = #file,
  producer = "MLIR", isOptimized = true, emissionKind = Full
>

// CHECK-DAG: #[[NULL:.*]] = #llvm.di_null_type
#null = #llvm.di_null_type

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

// CHECK-DAG: #[[PTR0:.*]] = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #[[INT0]], sizeInBits = 64, alignInBits = 32, offsetInBits = 4, extraData = #[[INT1]]>
#ptr0 = #llvm.di_derived_type<
  tag = DW_TAG_pointer_type, baseType = #int0,
  sizeInBits = 64, alignInBits = 32, offsetInBits = 4,
  extraData = #int1
>

// CHECK-DAG: #[[PTR1:.*]] = #llvm.di_derived_type<tag = DW_TAG_pointer_type, name = "ptr1">
#ptr1 = #llvm.di_derived_type<
  // Specify the name parameter.
  tag = DW_TAG_pointer_type, name = "ptr1"
>

// CHECK-DAG: #[[PTR2:.*]] = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #[[INT0]], sizeInBits = 64, alignInBits = 32, offsetInBits = 4, dwarfAddressSpace = 3, extraData = #[[INT1]]>
#ptr2 = #llvm.di_derived_type<
  tag = DW_TAG_pointer_type, baseType = #int0,
  sizeInBits = 64, alignInBits = 32, offsetInBits = 4,
  dwarfAddressSpace = 3, extraData = #int1
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

// CHECK-DAG: #[[TOPLEVEL:.*]] = #llvm.di_namespace<name = "toplevel", exportSymbols = true>
#toplevel_namespace = #llvm.di_namespace<
  name = "toplevel", exportSymbols = true
>

// CHECK-DAG: #[[NESTED:.*]] = #llvm.di_namespace<name = "nested", scope = #[[TOPLEVEL]], exportSymbols = false>
#nested_namespace = #llvm.di_namespace<
  name = "nested", scope = #toplevel_namespace, exportSymbols = false
>

// CHECK-DAG: #[[ANONYMOUS_NS:.*]] = #llvm.di_namespace<scope = #[[FILE]], exportSymbols = false>
#anonymous_namespace = #llvm.di_namespace<
  scope = #file,
  exportSymbols = false
>

// CHECK-DAG: #[[COMP2:.*]] = #llvm.di_composite_type<tag = DW_TAG_class_type, name = "class_name", file = #[[FILE]], scope = #[[NESTED]], flags = "TypePassByReference|NonTrivial">
#comp2 = #llvm.di_composite_type<
  tag = DW_TAG_class_type, name = "class_name", file = #file, scope = #nested_namespace,
  flags = "TypePassByReference|NonTrivial"
>

// CHECK-DAG: #[[COMP3:.+]] = #llvm.di_composite_type<{{.*}}, name = "expr_elements2"{{.*}}elements = #llvm.di_generic_subrange<count = #llvm.di_expression<[DW_OP_push_object_address, DW_OP_plus_uconst(16), DW_OP_deref]>, lowerBound = #llvm.di_expression<[DW_OP_push_object_address, DW_OP_plus_uconst(24), DW_OP_deref]>, stride = #llvm.di_expression<[DW_OP_push_object_address, DW_OP_plus_uconst(32), DW_OP_deref]>>>
#exp1 =  #llvm.di_expression<[DW_OP_push_object_address, DW_OP_plus_uconst(16), DW_OP_deref]>
#exp2 =  #llvm.di_expression<[DW_OP_push_object_address, DW_OP_plus_uconst(24), DW_OP_deref]>
#exp3 =  #llvm.di_expression<[DW_OP_push_object_address, DW_OP_plus_uconst(32), DW_OP_deref]>
#comp3 = #llvm.di_composite_type<tag = DW_TAG_array_type,
 name = "expr_elements2", baseType = #int0, elements =
 #llvm.di_generic_subrange<count = #exp1, lowerBound = #exp2, stride = #exp3>>

// CHECK-DAG: #[[SPTYPE0:.*]] = #llvm.di_subroutine_type<callingConvention = DW_CC_normal, types = #[[NULL]], #[[INT0]], #[[PTR0]], #[[PTR1]], #[[PTR2]], #[[COMP0:.*]], #[[COMP1:.*]], #[[COMP2:.*]], #[[COMP3:.*]]>
#spType0 = #llvm.di_subroutine_type<
  callingConvention = DW_CC_normal, types = #null, #int0, #ptr0, #ptr1, #ptr2, #comp0, #comp1, #comp2, #comp3
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

// CHECK-DAG: #[[SP0:.*]] = #llvm.di_subprogram<compileUnit = #[[CU]], scope = #[[ANONYMOUS_NS]], name = "addr", linkageName = "addr", file = #[[FILE]], line = 3, scopeLine = 3, subprogramFlags = "Definition|Optimized", type = #[[SPTYPE0]]>
#sp0 = #llvm.di_subprogram<
  compileUnit = #cu, scope = #anonymous_namespace, name = "addr", linkageName = "addr",
  file = #file, line = 3, scopeLine = 3, subprogramFlags = "Definition|Optimized", type = #spType0
>

// CHECK-DAG: #[[SP1:.*]] = #llvm.di_subprogram<scope = #[[COMP2]], file = #[[FILE]], type = #[[SPTYPE1]]>
#sp1 = #llvm.di_subprogram<
  // Omit the optional parameters.
  scope = #comp2, file = #file, type = #spType1
>

// CHECK-DAG: #[[MODULE:.*]] = #llvm.di_module<file = #[[FILE]], scope = #[[FILE]], name = "module", configMacros = "bar", includePath = "/", apinotes = "/", line = 42, isDecl = true>
#module = #llvm.di_module<
  file = #file, scope = #file, name = "module",
  configMacros = "bar", includePath = "/",
  apinotes = "/", line = 42, isDecl = true
>

// CHECK-DAG: #[[SP2:.*]] = #llvm.di_subprogram<compileUnit = #[[CU]], scope = #[[MODULE]], name = "value", file = #[[FILE]], subprogramFlags = Definition, type = #[[SPTYPE2]], annotations = #llvm.di_annotation<name = "foo", value = "bar">
#sp2 = #llvm.di_subprogram<
  compileUnit = #cu, scope = #module, name = "value",
  file = #file, subprogramFlags = "Definition", type = #spType2,
  annotations = #llvm.di_annotation<name = "foo", value = "bar">
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

// CHECK-DAG: #[[LABEL1:.*]] =  #llvm.di_label<scope = #[[BLOCK1]], name = "label", file = #[[FILE]], line = 42>
#label1 = #llvm.di_label<scope = #block1, name = "label", file = #file, line = 42>

// CHECK-DAG: #[[LABEL2:.*]] =  #llvm.di_label<scope = #[[BLOCK2]]>
#label2 = #llvm.di_label<scope = #block2>

// CHECK-DAG: #llvm.di_common_block<scope = #[[SP1]], name = "block", file = #[[FILE]], line = 3>
#di_common_block = #llvm.di_common_block<scope = #sp1, name = "block", file = #file, line = 3>
#global_var = #llvm.di_global_variable<scope = #di_common_block, name = "a",
 file = #file, line = 2, type = #int0>
#var_expression = #llvm.di_global_variable_expression<var = #global_var,
 expr = <>>
#global_var1 = #llvm.di_global_variable<scope = #di_common_block, name = "b",
 file = #file, line = 3, type = #int0>
#var_expression1 = #llvm.di_global_variable_expression<var = #global_var1,
 expr = <>>
llvm.mlir.global @data() {dbg_exprs = [#var_expression, #var_expression1]} : i64

// CHECK-DAG: llvm.mlir.global external @data() {{{.*}}dbg_exprs = [#[[EXP1:.*]], #[[EXP2:.*]]]} : i64
// CHECK-DAG: #[[EXP1]] = #llvm.di_global_variable_expression<var = #[[GV1:.*]], expr = <>>
// CHECK-DAG: #[[EXP2]] = #llvm.di_global_variable_expression<var = #[[GV2:.*]], expr = <>>
// CHECK-DAG: #[[GV1]] = #llvm.di_global_variable<{{.*}}name = "a"{{.*}}>
// CHECK-DAG: #[[GV2]] = #llvm.di_global_variable<{{.*}}name = "b"{{.*}}>


// CHECK: llvm.func @addr(%[[ARG:.*]]: i64)
llvm.func @addr(%arg: i64) {
  // CHECK: %[[ALLOC:.*]] = llvm.alloca
  %allocCount = llvm.mlir.constant(1 : i32) : i32
  %alloc = llvm.alloca %allocCount x i64 : (i32) -> !llvm.ptr

  // CHECK: llvm.intr.dbg.declare #[[VAR0]] = %[[ALLOC]]
  llvm.intr.dbg.declare #var0 = %alloc : !llvm.ptr
  llvm.return
}

// CHECK: llvm.func @value(%[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32)
llvm.func @value(%arg1: i32, %arg2: i32) {
  // CHECK: llvm.intr.dbg.value #[[VAR1]] #llvm.di_expression<[DW_OP_LLVM_fragment(16, 8), DW_OP_plus_uconst(2), DW_OP_deref]> = %[[ARG1]]
  llvm.intr.dbg.value #var1 #llvm.di_expression<[DW_OP_LLVM_fragment(16, 8), DW_OP_plus_uconst(2), DW_OP_deref]> = %arg1 : i32
  // CHECK: llvm.intr.dbg.value #[[VAR2]] = %[[ARG2]]
  llvm.intr.dbg.value #var2 = %arg2 : i32
  // CHECK: llvm.intr.dbg.label #[[LABEL1]]
  llvm.intr.dbg.label #label1
  // CHECK: llvm.intr.dbg.label #[[LABEL2]]
  llvm.intr.dbg.label #label2
  llvm.return
}
