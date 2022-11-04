// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

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


#si64 = #llvm.di_basic_type<
  tag = DW_TAG_base_type, name = "si64", sizeInBits = 0,
  encoding = DW_ATE_signed
>
#file = #llvm.di_file<"foo.mlir" in "/test/">
#cu = #llvm.di_compile_unit<
  sourceLanguage = DW_LANG_C, file = #file, producer = "MLIR",
  isOptimized = true, emissionKind = Full
>
#composite = #llvm.di_composite_type<
  tag = DW_TAG_structure_type, name = "composite", file = #file,
  line = 0, sizeInBits = 0, alignInBits = 0,
  elements = #llvm.di_subrange<count = 4>
>
#spType = #llvm.di_subroutine_type<callingConvention = DW_CC_normal, types = #si64, #composite>
#sp = #llvm.di_subprogram<
  compileUnit = #cu, scope = #file, name = "intrinsics", linkageName = "intrinsics",
  file = #file, line = 3, scopeLine = 3, subprogramFlags = "Definition|Optimized", type = #spType
>
#fileScope = #llvm.di_lexical_block_file<scope = #sp, file = #file, descriminator = 0>
#variable = #llvm.di_local_variable<scope = #fileScope, name = "arg", file = #file, line = 6, arg = 1, alignInBits = 0, type = #si64>

// CHECK-LABEL: define void @func_with_debug(
// CHECK-SAME: i64 %[[ARG:.*]]) !dbg ![[FUNC_LOC:[0-9]+]]
llvm.func @func_with_debug(%arg: i64) {
  // CHECK: %[[ALLOC:.*]] = alloca
  %allocCount = llvm.mlir.constant(1 : i32) : i32
  %alloc = llvm.alloca %allocCount x i64 : (i32) -> !llvm.ptr<i64>

  // CHECK: call void @llvm.dbg.value(metadata i64 %[[ARG]], metadata ![[VAR_LOC:[0-9]+]], metadata !DIExpression())
  // CHECK: call void @llvm.dbg.addr(metadata ptr %[[ALLOC]], metadata ![[VAR_LOC]], metadata !DIExpression())
  // CHECK: call void @llvm.dbg.declare(metadata ptr %[[ALLOC]], metadata ![[VAR_LOC]], metadata !DIExpression())
  llvm.dbg.value #variable = %arg : i64
  llvm.dbg.addr #variable = %alloc : !llvm.ptr<i64>
  llvm.dbg.declare #variable = %alloc : !llvm.ptr<i64>

  // CHECK: call void @func_no_debug(), !dbg ![[CALLSITE_LOC:[0-9]+]]
  llvm.call @func_no_debug() : () -> () loc(callsite("mysource.cc":3:4 at "mysource.cc":5:6))

  // CHECK: call void @func_no_debug(), !dbg ![[FILE_LOC:[0-9]+]]
  llvm.call @func_no_debug() : () -> () loc("foo.mlir":1:2)

  // CHECK: call void @func_no_debug(), !dbg ![[NAMED_LOC:[0-9]+]]
  llvm.call @func_no_debug() : () -> () loc("named"("foo.mlir":10:10))

  // CHECK: call void @func_no_debug(), !dbg ![[FUSED_LOC:[0-9]+]]
  llvm.call @func_no_debug() : () -> () loc(fused[callsite("mysource.cc":1:1 at "mysource.cc":5:6), "mysource.cc":1:1])

  llvm.return
} loc(fused<#sp>["foo.mlir":1:1])

// CHECK: ![[CU_LOC:.*]] = distinct !DICompileUnit(language: DW_LANG_C, file: ![[CU_FILE_LOC:.*]], producer: "MLIR", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK: ![[CU_FILE_LOC]] = !DIFile(filename: "foo.mlir", directory: "/test/")

// CHECK: ![[FUNC_LOC]] = distinct !DISubprogram(name: "intrinsics", linkageName: "intrinsics", scope: ![[CU_FILE_LOC]], file: ![[CU_FILE_LOC]], line: 3, type: ![[FUNC_TYPE:.*]], scopeLine: 3, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: ![[CU_LOC]])
// CHECK: ![[FUNC_TYPE]] = !DISubroutineType(cc: DW_CC_normal, types: ![[ARG_TYPES:.*]])
// CHECK: ![[ARG_TYPES]] = !{![[ARG_TYPE:.*]], ![[COMPOSITE_TYPE:.*]]}
// CHECK: ![[ARG_TYPE]] = !DIBasicType(name: "si64", encoding: DW_ATE_signed)
// CHECK: ![[COMPOSITE_TYPE]] = !DICompositeType(tag: DW_TAG_structure_type, name: "composite", file: ![[CU_FILE_LOC]], elements: ![[COMPOSITE_ELEMENTS:.*]])
// CHECK: ![[COMPOSITE_ELEMENTS]] = !{![[COMPOSITE_ELEMENT:.*]]}
// CHECK: ![[COMPOSITE_ELEMENT]] = !DISubrange(count: 4)

// CHECK: ![[VAR_LOC]] = !DILocalVariable(name: "arg", arg: 1, scope: ![[VAR_SCOPE:.*]], file: ![[CU_FILE_LOC]], line: 6, type: ![[ARG_TYPE]])
// CHECK: ![[VAR_SCOPE]] = distinct !DILexicalBlockFile(scope: ![[FUNC_LOC]], file: ![[CU_FILE_LOC]], discriminator: 0)

// CHECK-DAG: ![[CALLSITE_LOC]] = !DILocation(line: 3, column: 4,
// CHECK-DAG: ![[FILE_LOC]] = !DILocation(line: 1, column: 2,
// CHECK-DAG: ![[NAMED_LOC]] = !DILocation(line: 10, column: 10
// CHECK-DAG: ![[FUSED_LOC]] = !DILocation(line: 1, column: 1
