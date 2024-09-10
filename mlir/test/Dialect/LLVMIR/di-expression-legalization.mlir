// RUN: mlir-opt -llvm-legalize-for-export --split-input-file  %s | FileCheck %s -check-prefix=CHECK-OPT
// RUN: mlir-translate -mlir-to-llvmir --split-input-file %s | FileCheck %s -check-prefix=CHECK-TRANSLATE

#di_file = #llvm.di_file<"foo.c" in "/mlir/">
#di_compile_unit = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = #di_file, producer = "MLIR", isOptimized = true, emissionKind = Full>
#di_subprogram = #llvm.di_subprogram<compileUnit = #di_compile_unit, scope = #di_file, name = "simplify", file = #di_file, subprogramFlags = Definition>
#i32_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "i32", sizeInBits = 32, encoding = DW_ATE_unsigned>
#i8_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "i8", sizeInBits = 8, encoding = DW_ATE_unsigned>

// struct0: {i8, i32}
#struct0_first = #llvm.di_derived_type<tag = DW_TAG_member, name = "struct0_first", baseType = #i8_type, sizeInBits = 8, alignInBits = 8>
#struct0_second = #llvm.di_derived_type<tag = DW_TAG_member, name = "struct0_second", baseType = #i32_type, sizeInBits = 32, alignInBits = 32, offsetInBits = 32>
#struct0 = #llvm.di_composite_type<tag = DW_TAG_structure_type, name = "struct0", sizeInBits = 64, alignInBits = 32, elements = #struct0_first, #struct0_second>

// struct1: {i8, struct0}
#struct1_first = #llvm.di_derived_type<tag = DW_TAG_member, name = "struct1_first", baseType = #i8_type, sizeInBits = 8, alignInBits = 8>
#struct1_second = #llvm.di_derived_type<tag = DW_TAG_member, name = "struct1_second", baseType = #struct0, sizeInBits = 64, alignInBits = 32>
#struct1 = #llvm.di_composite_type<tag = DW_TAG_structure_type, name = "struct1", sizeInBits = 96, alignInBits = 32, elements = #struct1_first, #struct1_second>

// struct2: {i32, struct1}
#struct2_first = #llvm.di_derived_type<tag = DW_TAG_member, name = "struct2_first", baseType = #i32_type, sizeInBits = 32, alignInBits = 32>
#struct2_second = #llvm.di_derived_type<tag = DW_TAG_member, name = "struct2_second", baseType = #struct1, sizeInBits = 96, alignInBits = 32>
#struct2 = #llvm.di_composite_type<tag = DW_TAG_structure_type, name = "struct2", sizeInBits = 128, alignInBits = 32, elements = #struct2_first, #struct2_second>

#var0 = #llvm.di_local_variable<scope = #di_subprogram, name = "struct0_var", file = #di_file, line = 10, alignInBits = 32, type = #struct0>
#var1 = #llvm.di_local_variable<scope = #di_subprogram, name = "struct1_var", file = #di_file, line = 10, alignInBits = 32, type = #struct1>
#var2 = #llvm.di_local_variable<scope = #di_subprogram, name = "struct2_var", file = #di_file, line = 10, alignInBits = 32, type = #struct2>

#loc = loc("test.mlir":0:0)

llvm.func @merge_fragments(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
  // CHECK-OPT: #llvm.di_expression<[DW_OP_deref, DW_OP_LLVM_fragment(32, 32)]>
  // CHECK-TRANSLATE: !DIExpression(DW_OP_deref, DW_OP_LLVM_fragment, 32, 32),
  llvm.intr.dbg.value #var0 #llvm.di_expression<[DW_OP_deref, DW_OP_LLVM_fragment(32, 32)]> = %arg0 : !llvm.ptr loc(fused<#di_subprogram>[#loc])
  // CHECK-OPT: #llvm.di_expression<[DW_OP_deref, DW_OP_LLVM_fragment(64, 32)]>
  // CHECK-TRANSLATE: !DIExpression(DW_OP_deref, DW_OP_LLVM_fragment, 64, 32),
  llvm.intr.dbg.value #var1 #llvm.di_expression<[DW_OP_deref, DW_OP_LLVM_fragment(32, 32), DW_OP_LLVM_fragment(32, 64)]> = %arg1 : !llvm.ptr loc(fused<#di_subprogram>[#loc])
  // CHECK-OPT: #llvm.di_expression<[DW_OP_deref, DW_OP_LLVM_fragment(96, 32)]>
  // CHECK-TRANSLATE: !DIExpression(DW_OP_deref, DW_OP_LLVM_fragment, 96, 32),
  llvm.intr.dbg.value #var2 #llvm.di_expression<[DW_OP_deref, DW_OP_LLVM_fragment(32, 32), DW_OP_LLVM_fragment(32, 64), DW_OP_LLVM_fragment(32, 96)]> = %arg2 : !llvm.ptr loc(fused<#di_subprogram>[#loc])
  llvm.return
}
