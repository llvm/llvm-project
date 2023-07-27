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
  sourceLanguage = DW_LANG_C, file = #file, producer = "MLIR",
  isOptimized = true, emissionKind = Full
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

#spType1 = #llvm.di_subroutine_type<callingConvention = DW_CC_normal>
#sp1 = #llvm.di_subprogram<
  compileUnit = #cu, scope = #file, name = "empty_types",
  file = #file, subprogramFlags = "Definition", type = #spType1
>

// CHECK-LABEL: define void @func_with_debug(
// CHECK-SAME: i64 %[[ARG:.*]]) !dbg ![[FUNC_LOC:[0-9]+]]
llvm.func @func_with_debug(%arg: i64) {
  // CHECK: %[[ALLOC:.*]] = alloca
  %allocCount = llvm.mlir.constant(1 : i32) : i32
  %alloc = llvm.alloca %allocCount x i64 : (i32) -> !llvm.ptr<i64>

  // CHECK: call void @llvm.dbg.value(metadata i64 %[[ARG]], metadata ![[VAR_LOC:[0-9]+]], metadata !DIExpression())
  // CHECK: call void @llvm.dbg.declare(metadata ptr %[[ALLOC]], metadata ![[ADDR_LOC:[0-9]+]], metadata !DIExpression())
  // CHECK: call void @llvm.dbg.value(metadata i64 %[[ARG]], metadata ![[NO_NAME_VAR:[0-9]+]], metadata !DIExpression())
  llvm.intr.dbg.value #variable = %arg : i64
  llvm.intr.dbg.declare #variableAddr = %alloc : !llvm.ptr<i64>
  llvm.intr.dbg.value #noNameVariable= %arg : i64

  // CHECK: call void @func_no_debug(), !dbg ![[CALLSITE_LOC:[0-9]+]]
  llvm.call @func_no_debug() : () -> () loc(callsite("mysource.cc":3:4 at "mysource.cc":5:6))

  // CHECK: call void @func_no_debug(), !dbg ![[FILE_LOC:[0-9]+]]
  llvm.call @func_no_debug() : () -> () loc("foo.mlir":1:2)

  // CHECK: call void @func_no_debug(), !dbg ![[NAMED_LOC:[0-9]+]]
  llvm.call @func_no_debug() : () -> () loc("named"("foo.mlir":10:10))

  // CHECK: call void @func_no_debug(), !dbg ![[FUSED_LOC:[0-9]+]]
  llvm.call @func_no_debug() : () -> () loc(fused[callsite("mysource.cc":5:6 at "mysource.cc":1:1), "mysource.cc":1:1])

  // CHECK: add i64 %[[ARG]], %[[ARG]], !dbg ![[FUSEDWITH_LOC:[0-9]+]]
  %sum = llvm.add %arg, %arg : i64 loc(fused<#callee>[callsite("foo.mlir":2:4 at fused<#sp0>["foo.mlir":28:5])])

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

// CHECK-DAG: ![[CALLSITE_LOC]] = !DILocation(line: 3, column: 4,
// CHECK-DAG: ![[FILE_LOC]] = !DILocation(line: 1, column: 2,
// CHECK-DAG: ![[NAMED_LOC]] = !DILocation(line: 10, column: 10
// CHECK-DAG: ![[FUSED_LOC]] = !DILocation(line: 1, column: 1

// CHECK: ![[FUSEDWITH_LOC]] = !DILocation(line: 2, column: 4, scope: ![[CALLEE_LOC:.*]], inlinedAt: ![[INLINE_LOC:.*]])
// CHECK: ![[CALLEE_LOC]] = distinct !DISubprogram(name: "callee", scope: ![[COMPOSITE_TYPE]], file: ![[CU_FILE_LOC]], type: ![[CALLEE_TYPE:.*]], spFlags: DISPFlagDefinition, unit: ![[CU_LOC]])
// CHECK: ![[CALLEE_TYPE]] = !DISubroutineType(types: ![[CALLEE_ARGS:.*]])
// CHECK: ![[CALLEE_ARGS]] = !{![[ARG_TYPE:.*]], ![[ARG_TYPE:.*]]}
// CHECK: ![[INLINE_LOC]] = !DILocation(line: 28, column: 5,

// CHECK: ![[EMPTY_TYPES_LOC]] = distinct !DISubprogram(name: "empty_types", scope: ![[CU_FILE_LOC]], file: ![[CU_FILE_LOC]], type: ![[EMPTY_TYPES_TYPE:.*]], spFlags: DISPFlagDefinition
// CHECK: ![[EMPTY_TYPES_TYPE]] = !DISubroutineType(cc: DW_CC_normal, types: ![[EMPTY_TYPES_ARGS:.*]])
// CHECK: ![[EMPTY_TYPES_ARGS]] = !{}

// -----

#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_file = #llvm.di_file<"foo.mlir" in "/test/">
#di_compile_unit = #llvm.di_compile_unit<
  sourceLanguage = DW_LANG_C, file = #di_file, producer = "MLIR",
  isOptimized = true, emissionKind = Full
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
#loc1 = loc(callsite(#loc0 at fused<#di_subprogram>["foo.mlir":4:2]))

// CHECK-LABEL: define i32 @func_with_inlined_dbg_value(
// CHECK-SAME: i32 %[[ARG:.*]]) !dbg ![[OUTER_FUNC:[0-9]+]]
llvm.func @func_with_inlined_dbg_value(%arg0: i32) -> (i32) {
  // CHECK: call void @llvm.dbg.value(metadata i32 %[[ARG]], metadata ![[VAR_LOC0:[0-9]+]], metadata !DIExpression()), !dbg ![[DBG_LOC0:.*]]
  llvm.intr.dbg.value #di_local_variable0 = %arg0 : i32 loc(fused<#di_subprogram>[#loc0])
  // CHECK: call void @llvm.dbg.value(metadata i32 %[[ARG]], metadata ![[VAR_LOC1:[0-9]+]], metadata !DIExpression()), !dbg ![[DBG_LOC1:.*]]
  llvm.intr.dbg.value #di_local_variable1 = %arg0 : i32 loc(fused<#di_lexical_block_file>[#loc1])
  // CHECK: call void @llvm.dbg.label(metadata ![[LABEL:[0-9]+]]), !dbg ![[DBG_LOC1:.*]]
  llvm.intr.dbg.label #di_label loc(fused<#di_lexical_block_file>[#loc1])
  llvm.return %arg0 : i32
} loc(fused<#di_subprogram>["caller"])

// CHECK: ![[FILE:.*]] = !DIFile(filename: "foo.mlir", directory: "/test/")
// CHECK-DAG: ![[OUTER_FUNC]] = distinct !DISubprogram(name: "outer_func", scope: ![[FILE]]
// CHECK-DAG: ![[INNER_FUNC:.*]] = distinct !DISubprogram(name: "inner_func", scope: ![[FILE]]
// CHECK-DAG: ![[LEXICAL_BLOCK_FILE:.*]] = distinct !DILexicalBlockFile(scope: ![[INNER_FUNC]], file: ![[FILE]], discriminator: 0)
// CHECK-DAG: ![[VAR_LOC0]] = !DILocalVariable(name: "a", scope: ![[OUTER_FUNC]], file: ![[FILE]]
// CHECK-DAG: ![[VAR_LOC1]] = !DILocalVariable(name: "b", scope: ![[LEXICAL_BLOCK_FILE]], file: ![[FILE]]
// CHECK-DAG  ![[LABEL]] = !DILabel(scope: ![[LEXICAL_BLOCK_FILE]], name: "label", file: ![[FILE]], line: 42)
