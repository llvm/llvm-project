// RUN: mlir-translate -mlir-to-llvmir --split-input-file %s | FileCheck %s

// CHECK-LABEL: define void @func_with_empty_named_info()
// Check that translation doens't crash in the presence of an inlineble call
// with a named loc that has no backing source info.
llvm.func @callee() {
  llvm.return
} loc("calleesource.cc":1:1)
llvm.func @func_with_empty_named_info() {
  llvm.call @callee() : () -> () loc("named with no line info")
  llvm.return
}

// CHECK-LABEL: define void @func_no_debug()
// CHECK-NOT: !dbg
llvm.func @func_no_debug() {
  llvm.return loc(unknown)
} loc(unknown)

#file = #llvm.di_file<"foo.mlir" in "/test/">
#si64 = #llvm.di_basic_type<
  // Omit the optional sizeInBits and encoding parameters.
  tag = DW_TAG_base_type, name = "si64"
>
#si32 = #llvm.di_basic_type<
  tag = DW_TAG_base_type, name = "si32",
  sizeInBits = 32, encoding = DW_ATE_signed
>
#ptr = #llvm.di_derived_type<
  tag = DW_TAG_pointer_type, baseType = #si32,
  sizeInBits = 64, alignInBits = 32, offsetInBits = 8
>
#named = #llvm.di_derived_type<
  // Specify the name parameter.
  tag = DW_TAG_pointer_type, name = "named", baseType = #si32
>
#cu = #llvm.di_compile_unit<
  id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = #file,
  producer = "MLIR", isOptimized = true, emissionKind = Full
>
#composite = #llvm.di_composite_type<
  tag = DW_TAG_structure_type, name = "composite", file = #file,
  line = 42, sizeInBits = 64, alignInBits = 32,
  elements = #llvm.di_subrange<count = 4>
>
#vector = #llvm.di_composite_type<
  tag = DW_TAG_array_type, name = "array", file = #file,
  baseType = #si64, flags = Vector,
  elements = #llvm.di_subrange<lowerBound = 0, upperBound = 4, stride = 1>
>
#null = #llvm.di_null_type
#spType0 = #llvm.di_subroutine_type<callingConvention = DW_CC_normal, types = #null, #si64, #ptr, #named, #composite, #vector>
#toplevel_namespace = #llvm.di_namespace<
  name = "toplevel", exportSymbols = true
>
#nested_namespace = #llvm.di_namespace<
  name = "nested", scope = #toplevel_namespace, exportSymbols = false
>
#sp0 = #llvm.di_subprogram<
  compileUnit = #cu, scope = #nested_namespace, name = "func_with_debug", linkageName = "func_with_debug",
  file = #file, line = 3, scopeLine = 3, subprogramFlags = "Definition|Optimized", type = #spType0
>
#calleeType = #llvm.di_subroutine_type<
  // Omit the optional callingConvention parameter.
  types = #si64, #si64>
#callee = #llvm.di_subprogram<
  // Omit the optional linkageName, line, and scopeLine parameters.
  compileUnit = #cu, scope = #composite, name = "callee",
  file = #file, subprogramFlags = "Definition", type = #calleeType
>
#fileScope = #llvm.di_lexical_block_file<scope = #sp0, file = #file, discriminator = 0>
#blockScope = #llvm.di_lexical_block<scope = #sp0>
#variable = #llvm.di_local_variable<scope = #fileScope, name = "arg", file = #file, line = 6, arg = 1, alignInBits = 32, type = #si64>
#variableAddr = #llvm.di_local_variable<scope = #blockScope, name = "alloc">
#noNameVariable = #llvm.di_local_variable<scope = #blockScope>
#module = #llvm.di_module<
  file = #file, scope = #file, name = "module",
  configMacros = "bar", includePath = "/",
  apinotes = "/", line = 42, isDecl = true
>
#spType1 = #llvm.di_subroutine_type<callingConvention = DW_CC_normal>
#sp1 = #llvm.di_subprogram<
  compileUnit = #cu, scope = #module, name = "empty_types",
  file = #file, subprogramFlags = "Definition", type = #spType1
>

// CHECK-LABEL: define void @func_with_debug(
// CHECK-SAME: i64 %[[ARG:.*]]) !dbg ![[FUNC_LOC:[0-9]+]]
llvm.func @func_with_debug(%arg: i64) {
  // CHECK: %[[ALLOC:.*]] = alloca
  %allocCount = llvm.mlir.constant(1 : i32) : i32
  %alloc = llvm.alloca %allocCount x i64 : (i32) -> !llvm.ptr

  // CHECK: call void @llvm.dbg.value(metadata i64 %[[ARG]], metadata ![[VAR_LOC:[0-9]+]], metadata !DIExpression(DW_OP_LLVM_fragment, 0, 1))
  llvm.intr.dbg.value #variable #llvm.di_expression<[DW_OP_LLVM_fragment(0, 1)]> = %arg : i64

  // CHECK: call void @llvm.dbg.declare(metadata ptr %[[ALLOC]], metadata ![[ADDR_LOC:[0-9]+]], metadata !DIExpression(DW_OP_deref, DW_OP_LLVM_convert, 4, DW_ATE_signed))
  llvm.intr.dbg.declare #variableAddr #llvm.di_expression<[DW_OP_deref, DW_OP_LLVM_convert(4, DW_ATE_signed)]> = %alloc : !llvm.ptr

  // CHECK: call void @llvm.dbg.value(metadata i64 %[[ARG]], metadata ![[NO_NAME_VAR:[0-9]+]], metadata !DIExpression())
  llvm.intr.dbg.value #noNameVariable = %arg : i64

  // CHECK: call void @func_no_debug(), !dbg ![[FILE_LOC:[0-9]+]]
  llvm.call @func_no_debug() : () -> () loc("foo.mlir":1:2)

  // CHECK: call void @func_no_debug(), !dbg ![[NAMED_LOC:[0-9]+]]
  llvm.call @func_no_debug() : () -> () loc("named"("foo.mlir":10:10))

  // CHECK: call void @func_no_debug(), !dbg ![[CALLSITE_LOC:[0-9]+]]
  llvm.call @func_no_debug() : () -> () loc(callsite("nodebug.cc":3:4 at "mysource.cc":5:6))

  // CHECK: call void @func_no_debug(), !dbg ![[CALLSITE_LOC:[0-9]+]]
  llvm.call @func_no_debug() : () -> () loc(callsite("nodebug.cc":3:4 at fused<#sp0>["mysource.cc":5:6]))

  // CHECK: call void @func_no_debug(), !dbg ![[FUSED_LOC:[0-9]+]]
  llvm.call @func_no_debug() : () -> () loc(fused[callsite(fused<#callee>["mysource.cc":5:6] at "mysource.cc":1:1), "mysource.cc":1:1])

  // CHECK: add i64 %[[ARG]], %[[ARG]], !dbg ![[FUSEDWITH_LOC:[0-9]+]]
  %sum = llvm.add %arg, %arg : i64 loc(callsite(fused<#callee>["foo.mlir":2:4] at fused<#sp0>["foo.mlir":28:5]))

  llvm.return
} loc(fused<#sp0>["foo.mlir":1:1])

// CHECK: define void @empty_types() !dbg ![[EMPTY_TYPES_LOC:[0-9]+]]
llvm.func @empty_types() {
  llvm.return
} loc(fused<#sp1>["foo.mlir":2:1])

// CHECK: ![[CU_LOC:.*]] = distinct !DICompileUnit(language: DW_LANG_C, file: ![[CU_FILE_LOC:.*]], producer: "MLIR", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK: ![[CU_FILE_LOC]] = !DIFile(filename: "foo.mlir", directory: "/test/")

// CHECK: ![[FUNC_LOC]] = distinct !DISubprogram(name: "func_with_debug", linkageName: "func_with_debug", scope: ![[NESTED_NAMESPACE:.*]], file: ![[CU_FILE_LOC]], line: 3, type: ![[FUNC_TYPE:.*]], scopeLine: 3, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: ![[CU_LOC]])
// CHECK: ![[NESTED_NAMESPACE]] = !DINamespace(name: "nested", scope: ![[TOPLEVEL_NAMESPACE:.*]])
// CHECK: ![[TOPLEVEL_NAMESPACE]] = !DINamespace(name: "toplevel", scope: null, exportSymbols: true)
// CHECK: ![[FUNC_TYPE]] = !DISubroutineType(cc: DW_CC_normal, types: ![[FUNC_ARGS:.*]])
// CHECK: ![[FUNC_ARGS]] = !{null, ![[ARG_TYPE:.*]], ![[PTR_TYPE:.*]], ![[NAMED_TYPE:.*]], ![[COMPOSITE_TYPE:.*]], ![[VECTOR_TYPE:.*]]}
// CHECK: ![[ARG_TYPE]] = !DIBasicType(name: "si64")
// CHECK: ![[PTR_TYPE]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[BASE_TYPE:.*]], size: 64, align: 32, offset: 8)
// CHECK: ![[BASE_TYPE]] = !DIBasicType(name: "si32", size: 32, encoding: DW_ATE_signed)
// CHECK: ![[NAMED_TYPE]] = !DIDerivedType(tag: DW_TAG_pointer_type, name: "named", baseType: ![[BASE_TYPE:.*]])
// CHECK: ![[COMPOSITE_TYPE]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "composite", file: ![[CU_FILE_LOC]], line: 42, size: 64, align: 32, elements: ![[COMPOSITE_ELEMENTS:.*]])
// CHECK: ![[COMPOSITE_ELEMENTS]] = !{![[COMPOSITE_ELEMENT:.*]]}
// CHECK: ![[COMPOSITE_ELEMENT]] = !DISubrange(count: 4)
// CHECK: ![[VECTOR_TYPE]] = !DICompositeType(tag: DW_TAG_array_type, name: "array", file: ![[CU_FILE_LOC]], baseType: ![[ARG_TYPE]], flags: DIFlagVector, elements: ![[VECTOR_ELEMENTS:.*]])
// CHECK: ![[VECTOR_ELEMENTS]] = !{![[VECTOR_ELEMENT:.*]]}
// CHECK: ![[VECTOR_ELEMENT]] = !DISubrange(lowerBound: 0, upperBound: 4, stride: 1)

// CHECK: ![[VAR_LOC]] = !DILocalVariable(name: "arg", arg: 1, scope: ![[VAR_SCOPE:.*]], file: ![[CU_FILE_LOC]], line: 6, type: ![[ARG_TYPE]], align: 32)
// CHECK: ![[VAR_SCOPE]] = distinct !DILexicalBlockFile(scope: ![[FUNC_LOC]], file: ![[CU_FILE_LOC]], discriminator: 0)
// CHECK: ![[ADDR_LOC]] = !DILocalVariable(name: "alloc", scope: ![[BLOCK_LOC:.*]])
// CHECK: ![[BLOCK_LOC]] = distinct !DILexicalBlock(scope: ![[FUNC_LOC]])
// CHECK: ![[NO_NAME_VAR]] = !DILocalVariable(scope: ![[BLOCK_LOC]])

// CHECK-DAG: ![[CALLSITE_LOC]] = !DILocation(line: 5, column: 6,
// CHECK-DAG: ![[FILE_LOC]] = !DILocation(line: 1, column: 2,
// CHECK-DAG: ![[NAMED_LOC]] = !DILocation(line: 10, column: 10
// CHECK-DAG: ![[FUSED_LOC]] = !DILocation(line: 1, column: 1

// CHECK: ![[FUSEDWITH_LOC]] = !DILocation(line: 2, column: 4, scope: ![[CALLEE_LOC:.*]], inlinedAt: ![[INLINE_LOC:.*]])
// CHECK: ![[CALLEE_LOC]] = distinct !DISubprogram(name: "callee", scope: ![[COMPOSITE_TYPE]], file: ![[CU_FILE_LOC]], type: ![[CALLEE_TYPE:.*]], spFlags: DISPFlagDefinition, unit: ![[CU_LOC]])
// CHECK: ![[CALLEE_TYPE]] = !DISubroutineType(types: ![[CALLEE_ARGS:.*]])
// CHECK: ![[CALLEE_ARGS]] = !{![[ARG_TYPE:.*]], ![[ARG_TYPE:.*]]}
// CHECK: ![[INLINE_LOC]] = !DILocation(line: 28, column: 5,

// CHECK: ![[EMPTY_TYPES_LOC]] = distinct !DISubprogram(name: "empty_types", scope: ![[MODULE:.*]], file: ![[CU_FILE_LOC]], type: ![[EMPTY_TYPES_TYPE:.*]], spFlags: DISPFlagDefinition
// CHECK: ![[MODULE]] = !DIModule(scope: ![[CU_FILE_LOC]], name: "module", configMacros: "bar", includePath: "/", apinotes: "/", file: ![[CU_FILE_LOC]], line: 42, isDecl: true)
// CHECK: ![[EMPTY_TYPES_TYPE]] = !DISubroutineType(cc: DW_CC_normal, types: ![[EMPTY_TYPES_ARGS:.*]])
// CHECK: ![[EMPTY_TYPES_ARGS]] = !{}

// -----

#di_file = #llvm.di_file<"foo.mlir" in "/test/">
#di_subprogram = #llvm.di_subprogram<
  scope = #di_file, name = "func_decl_with_subprogram", file = #di_file, subprogramFlags = "Optimized"
>

// CHECK-LABEL: declare !dbg
// CHECK-SAME: ![[SUBPROGRAM:.*]] i32 @func_decl_with_subprogram(
llvm.func @func_decl_with_subprogram() -> (i32) loc(fused<#di_subprogram>["foo.mlir":2:1])

// CHECK: ![[SUBPROGRAM]] = !DISubprogram(name: "func_decl_with_subprogram", scope: ![[FILE:.*]], file: ![[FILE]], spFlags: DISPFlagOptimized)
// CHECK: ![[FILE]] = !DIFile(filename: "foo.mlir", directory: "/test/")

// -----

#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_file = #llvm.di_file<"foo.mlir" in "/test/">
#di_compile_unit = #llvm.di_compile_unit<
  id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = #di_file,
  producer = "MLIR", isOptimized = true, emissionKind = Full
>
#di_subprogram = #llvm.di_subprogram<
  compileUnit = #di_compile_unit, scope = #di_file, name = "outer_func",
  file = #di_file, subprogramFlags = "Definition|Optimized"
>
#di_subprogram1 = #llvm.di_subprogram<
  compileUnit = #di_compile_unit, scope = #di_file, name = "inner_func",
  file = #di_file, subprogramFlags = "LocalToUnit|Definition|Optimized"
>
#di_local_variable0 = #llvm.di_local_variable<scope = #di_subprogram, name = "a", file = #di_file, type = #di_basic_type>
#di_lexical_block_file = #llvm.di_lexical_block_file<scope = #di_subprogram1, file = #di_file, discriminator = 0>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_lexical_block_file, name = "b", file = #di_file, type = #di_basic_type>
#di_label = #llvm.di_label<scope = #di_lexical_block_file, name = "label", file = #di_file, line = 42>

#loc0 = loc("foo.mlir":0:0)
#loc1 = loc(callsite(fused<#di_lexical_block_file>[#loc0] at fused<#di_subprogram>["foo.mlir":4:2]))

// CHECK-LABEL: define i32 @func_with_inlined_dbg_value(
// CHECK-SAME: i32 %[[ARG:.*]]) !dbg ![[OUTER_FUNC:[0-9]+]]
llvm.func @func_with_inlined_dbg_value(%arg0: i32) -> (i32) {
  // CHECK: call void @llvm.dbg.value(metadata i32 %[[ARG]], metadata ![[VAR_LOC0:[0-9]+]], metadata !DIExpression()), !dbg ![[DBG_LOC0:.*]]
  llvm.intr.dbg.value #di_local_variable0 = %arg0 : i32 loc(fused<#di_subprogram>[#loc0])
  // CHECK: call void @llvm.dbg.value(metadata i32 %[[ARG]], metadata ![[VAR_LOC1:[0-9]+]], metadata !DIExpression()), !dbg ![[DBG_LOC1:.*]]
  llvm.intr.dbg.value #di_local_variable1 = %arg0 : i32 loc(#loc1)
  // CHECK: call void @llvm.dbg.label(metadata ![[LABEL:[0-9]+]]), !dbg ![[DBG_LOC1:.*]]
  llvm.intr.dbg.label #di_label loc(#loc1)
  llvm.return %arg0 : i32
} loc(fused<#di_subprogram>["caller"])

// CHECK: ![[FILE:.*]] = !DIFile(filename: "foo.mlir", directory: "/test/")
// CHECK-DAG: ![[OUTER_FUNC]] = distinct !DISubprogram(name: "outer_func", scope: ![[FILE]]
// CHECK-DAG: ![[INNER_FUNC:.*]] = distinct !DISubprogram(name: "inner_func", scope: ![[FILE]]
// CHECK-DAG: ![[LEXICAL_BLOCK_FILE:.*]] = distinct !DILexicalBlockFile(scope: ![[INNER_FUNC]], file: ![[FILE]], discriminator: 0)
// CHECK-DAG: ![[VAR_LOC0]] = !DILocalVariable(name: "a", scope: ![[OUTER_FUNC]], file: ![[FILE]]
// CHECK-DAG: ![[VAR_LOC1]] = !DILocalVariable(name: "b", scope: ![[LEXICAL_BLOCK_FILE]], file: ![[FILE]]
// CHECK-DAG  ![[LABEL]] = !DILabel(scope: ![[LEXICAL_BLOCK_FILE]], name: "label", file: ![[FILE]], line: 42)

// -----

#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_file = #llvm.di_file<"foo.mlir" in "/test/">
#di_compile_unit = #llvm.di_compile_unit<
  id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = #di_file,
  producer = "MLIR", isOptimized = true, emissionKind = Full
>
#di_subprogram = #llvm.di_subprogram<
  compileUnit = #di_compile_unit, scope = #di_file, name = "func",
  file = #di_file, subprogramFlags = Definition>
#di_local_variable = #llvm.di_local_variable<scope = #di_subprogram, name = "a", file = #di_file, type = #di_basic_type>

#loc = loc("foo.mlir":0:0)

// CHECK-LABEL: define void @func_without_subprogram(
// CHECK-SAME: i32 %[[ARG:.*]])
llvm.func @func_without_subprogram(%0 : i32) {
  // CHECK: call void @llvm.dbg.value(metadata i32 %[[ARG]], metadata ![[VAR_LOC:[0-9]+]], metadata !DIExpression()), !dbg ![[DBG_LOC0:.*]]
  llvm.intr.dbg.value #di_local_variable = %0 : i32 loc(fused<#di_subprogram>[#loc])
  llvm.return
}

// CHECK: ![[FILE:.*]] = !DIFile(filename: "foo.mlir", directory: "/test/")
// CHECK-DAG: ![[FUNC:.*]] = distinct !DISubprogram(name: "func", scope: ![[FILE]]
// CHECK-DAG: ![[VAR_LOC]] = !DILocalVariable(name: "a", scope: ![[FUNC]], file: ![[FILE]]

// -----

// Ensures that debug intrinsics without a valid location are not exported to
// avoid broken LLVM IR.

#di_file = #llvm.di_file<"foo.mlir" in "/test/">
#di_compile_unit = #llvm.di_compile_unit<
  id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = #di_file,
  producer = "MLIR", isOptimized = true, emissionKind = Full
>
#di_subprogram = #llvm.di_subprogram<
  compileUnit = #di_compile_unit, scope = #di_file, name = "outer_func",
  file = #di_file, subprogramFlags = "Definition|Optimized"
>
#di_local_variable = #llvm.di_local_variable<scope = #di_subprogram, name = "a">
#declared_var = #llvm.di_local_variable<scope = #di_subprogram, name = "alloc">
#di_label = #llvm.di_label<scope = #di_subprogram, name = "label", file = #di_file, line = 42>

// CHECK-LABEL: define i32 @dbg_intrinsics_with_no_location(
llvm.func @dbg_intrinsics_with_no_location(%arg0: i32) -> (i32) {
  %allocCount = llvm.mlir.constant(1 : i32) : i32
  %alloc = llvm.alloca %allocCount x i64 : (i32) -> !llvm.ptr
  // CHECK-NOT: @llvm.dbg.value
  llvm.intr.dbg.value #di_local_variable = %arg0 : i32
  // CHECK-NOT: @llvm.dbg.declare
  llvm.intr.dbg.declare #declared_var = %alloc : !llvm.ptr
  // CHECK-NOT: @llvm.dbg.label
  llvm.intr.dbg.label #di_label
  llvm.return %arg0 : i32
}

// -----

// CHECK: @global_with_expr_1 = external global i64, !dbg {{.*}}
// CHECK: @global_with_expr_2 = external global i64, !dbg {{.*}}
// CHECK: !llvm.module.flags = !{{{.*}}}
// CHECK: !llvm.dbg.cu = !{{{.*}}}
// CHECK-DAG: ![[FILE:.*]] = !DIFile(filename: "not", directory: "existence")
// CHECK-DAG: ![[TYPE:.*]] = !DIBasicType(name: "uint64_t", size: 64, encoding: DW_ATE_unsigned)
// CHECK-DAG: ![[SCOPE:.*]] = distinct !DICompileUnit(language: DW_LANG_C, file: ![[FILE]], producer: "MLIR", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: ![[GVALS:.*]])
// CHECK-DAG: ![[GVAR0:.*]] = distinct !DIGlobalVariable(name: "global_with_expr_1", linkageName: "global_with_expr_1", scope: ![[SCOPE]], file: ![[FILE]], line: 370, type: ![[TYPE]], isLocal: false, isDefinition: false)
// CHECK-DAG: ![[GVAR1:.*]] = distinct !DIGlobalVariable(name: "global_with_expr_2", linkageName: "global_with_expr_2", scope: ![[SCOPE]], file: ![[FILE]], line: 371, type: ![[TYPE]], isLocal: true, isDefinition: true, align: 8)
// CHECK-DAG: ![[GEXPR0:.*]] = !DIGlobalVariableExpression(var: ![[GVAR0]], expr: !DIExpression())
// CHECK-DAG: ![[GEXPR1:.*]] = !DIGlobalVariableExpression(var: ![[GVAR1]], expr: !DIExpression())
// CHECK-DAG: ![[GVALS]] = !{![[GEXPR0]], ![[GEXPR1]]}

#di_file_2 = #llvm.di_file<"not" in "existence">
#di_compile_unit_2 = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = #di_file_2, producer = "MLIR", isOptimized = true, emissionKind = Full>
#di_basic_type_2 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "uint64_t", sizeInBits = 64, encoding = DW_ATE_unsigned>
llvm.mlir.global external @global_with_expr_1() {addr_space = 0 : i32, dbg_expr = #llvm.di_global_variable_expression<var = <scope = #di_compile_unit_2, name = "global_with_expr_1", linkageName = "global_with_expr_1", file = #di_file_2, line = 370, type = #di_basic_type_2>, expr = <>>} : i64
llvm.mlir.global external @global_with_expr_2() {addr_space = 0 : i32, dbg_expr = #llvm.di_global_variable_expression<var = <scope = #di_compile_unit_2, name = "global_with_expr_2", linkageName = "global_with_expr_2", file = #di_file_2, line = 371, type = #di_basic_type_2, isLocalToUnit = true, isDefined = true, alignInBits = 8>, expr = <>>} : i64

// -----

// Nameless and scopeless global constant.

// CHECK-LABEL: @.str.1 = external constant [10 x i8]
// CHECK-SAME: !dbg ![[GLOBAL_VAR_EXPR:.*]]
// CHECK-DAG: ![[GLOBAL_VAR_EXPR]] = !DIGlobalVariableExpression(var: ![[GLOBAL_VAR:.*]], expr: !DIExpression())
// CHECK-DAG: ![[GLOBAL_VAR]] = distinct !DIGlobalVariable(scope: null, file: !{{[0-9]+}}, line: 268, type: !{{[0-9]+}}, isLocal: true, isDefinition: true)

#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "char", sizeInBits = 8, encoding = DW_ATE_signed_char>
#di_file = #llvm.di_file<"file.c" in "/path/to/file">
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_basic_type>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type, sizeInBits = 80>
#di_global_variable = #llvm.di_global_variable<file = #di_file, line = 268, type = #di_composite_type, isLocalToUnit = true, isDefined = true>
#di_global_variable_expression = #llvm.di_global_variable_expression<var = #di_global_variable, expr = <>>

llvm.mlir.global external constant @".str.1"() {addr_space = 0 : i32, dbg_expr = #di_global_variable_expression} : !llvm.array<10 x i8>

// -----

// CHECK-DAG: ![[FILE1:.*]] = !DIFile(filename: "foo1.mlir", directory: "/test/")
#di_file_1 = #llvm.di_file<"foo1.mlir" in "/test/">
// CHECK-DAG: ![[FILE2:.*]] = !DIFile(filename: "foo2.mlir", directory: "/test/")
#di_file_2 = #llvm.di_file<"foo2.mlir" in "/test/">
// CHECK-DAG: ![[SCOPE2:.*]] = distinct !DICompileUnit(language: DW_LANG_C, file: ![[FILE2]], producer: "MLIR", isOptimized: true, runtimeVersion: 0, emissionKind: DebugDirectivesOnly)
#di_compile_unit_1 = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = #di_file_1, producer = "MLIR", isOptimized = true, emissionKind = LineTablesOnly>
// CHECK-DAG: ![[SCOPE1:.*]] = distinct !DICompileUnit(language: DW_LANG_C, file: ![[FILE1]], producer: "MLIR", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
#di_compile_unit_2 = #llvm.di_compile_unit<id = distinct[1]<>, sourceLanguage = DW_LANG_C, file = #di_file_2, producer = "MLIR", isOptimized = true, emissionKind = DebugDirectivesOnly>
#di_subprogram_1 = #llvm.di_subprogram<compileUnit = #di_compile_unit_1, scope = #di_file_1, name = "func1", file = #di_file_1, subprogramFlags = "Definition|Optimized">
#di_subprogram_2 = #llvm.di_subprogram<compileUnit = #di_compile_unit_2, scope = #di_file_2, name = "func2", file = #di_file_2, subprogramFlags = "Definition|Optimized">

llvm.func @func_line_tables() {
  llvm.return
} loc(fused<#di_subprogram_1>["foo1.mlir":0:0])

llvm.func @func_debug_directives() {
  llvm.return
} loc(fused<#di_subprogram_2>["foo2.mlir":0:0])
