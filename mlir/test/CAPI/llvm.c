//===- llvm.c - Test of llvm APIs -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: mlir-capi-llvm-test 2>&1 | FileCheck %s

#include "mlir-c/Dialect/LLVM.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "llvm-c/Core.h"
#include "llvm-c/DebugInfo.h"

#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// CHECK-LABEL: testTypeCreation()
static void testTypeCreation(MlirContext ctx) {
  fprintf(stderr, "testTypeCreation()\n");
  MlirType i8 = mlirIntegerTypeGet(ctx, 8);
  MlirType i32 = mlirIntegerTypeGet(ctx, 32);
  MlirType i64 = mlirIntegerTypeGet(ctx, 64);

  const char *ptr_text = "!llvm.ptr";
  MlirType ptr = mlirLLVMPointerTypeGet(ctx, 0);
  MlirType ptr_ref =
      mlirTypeParseGet(ctx, mlirStringRefCreateFromCString(ptr_text));
  // CHECK: !llvm.ptr: 1
  fprintf(stderr, "%s: %d\n", ptr_text, mlirTypeEqual(ptr, ptr_ref));

  const char *ptr_addr_text = "!llvm.ptr<42>";
  MlirType ptr_addr = mlirLLVMPointerTypeGet(ctx, 42);
  MlirType ptr_addr_ref =
      mlirTypeParseGet(ctx, mlirStringRefCreateFromCString(ptr_addr_text));
  // CHECK: !llvm.ptr<42>: 1
  fprintf(stderr, "%s: %d\n", ptr_addr_text,
          mlirTypeEqual(ptr_addr, ptr_addr_ref));

  const char *voidt_text = "!llvm.void";
  MlirType voidt = mlirLLVMVoidTypeGet(ctx);
  MlirType voidt_ref =
      mlirTypeParseGet(ctx, mlirStringRefCreateFromCString(voidt_text));
  // CHECK: !llvm.void: 1
  fprintf(stderr, "%s: %d\n", voidt_text, mlirTypeEqual(voidt, voidt_ref));

  const char *i32_4_text = "!llvm.array<4 x i32>";
  MlirType i32_4 = mlirLLVMArrayTypeGet(i32, 4);
  MlirType i32_4_ref =
      mlirTypeParseGet(ctx, mlirStringRefCreateFromCString(i32_4_text));
  // CHECK: !llvm.array<4 x i32>: 1
  fprintf(stderr, "%s: %d\n", i32_4_text, mlirTypeEqual(i32_4, i32_4_ref));

  const char *i8_i32_i64_text = "!llvm.func<i8 (i32, i64)>";
  const MlirType i32_i64_arr[] = {i32, i64};
  MlirType i8_i32_i64 = mlirLLVMFunctionTypeGet(i8, 2, i32_i64_arr, false);
  MlirType i8_i32_i64_ref =
      mlirTypeParseGet(ctx, mlirStringRefCreateFromCString(i8_i32_i64_text));
  // CHECK: !llvm.func<i8 (i32, i64)>: 1
  fprintf(stderr, "%s: %d\n", i8_i32_i64_text,
          mlirTypeEqual(i8_i32_i64, i8_i32_i64_ref));

  const char *i32_i64_s_text = "!llvm.struct<(i32, i64)>";
  MlirType i32_i64_s = mlirLLVMStructTypeLiteralGet(ctx, 2, i32_i64_arr, false);
  MlirType i32_i64_s_ref =
      mlirTypeParseGet(ctx, mlirStringRefCreateFromCString(i32_i64_s_text));
  // CHECK: !llvm.struct<(i32, i64)>: 1
  fprintf(stderr, "%s: %d\n", i32_i64_s_text,
          mlirTypeEqual(i32_i64_s, i32_i64_s_ref));
}

// CHECK-LABEL: testStructTypeCreation
static int testStructTypeCreation(MlirContext ctx) {
  fprintf(stderr, "testStructTypeCreation\n");

  // CHECK: !llvm.struct<()>
  mlirTypeDump(mlirLLVMStructTypeLiteralGet(ctx, /*nFieldTypes=*/0,
                                            /*fieldTypes=*/NULL,
                                            /*isPacked=*/false));

  MlirType i8 = mlirIntegerTypeGet(ctx, 8);
  MlirType i32 = mlirIntegerTypeGet(ctx, 32);
  MlirType i64 = mlirIntegerTypeGet(ctx, 64);
  MlirType i8_i32_i64[] = {i8, i32, i64};
  // CHECK: !llvm.struct<(i8, i32, i64)>
  mlirTypeDump(
      mlirLLVMStructTypeLiteralGet(ctx, sizeof(i8_i32_i64) / sizeof(MlirType),
                                   i8_i32_i64, /*isPacked=*/false));
  // CHECK: !llvm.struct<(i32)>
  mlirTypeDump(mlirLLVMStructTypeLiteralGet(ctx, 1, &i32, /*isPacked=*/false));
  MlirType i32_i32[] = {i32, i32};
  // CHECK: !llvm.struct<packed (i32, i32)>
  mlirTypeDump(mlirLLVMStructTypeLiteralGet(
      ctx, sizeof(i32_i32) / sizeof(MlirType), i32_i32, /*isPacked=*/true));

  MlirType literal =
      mlirLLVMStructTypeLiteralGet(ctx, sizeof(i8_i32_i64) / sizeof(MlirType),
                                   i8_i32_i64, /*isPacked=*/false);
  // CHECK: num elements: 3
  // CHECK: i8
  // CHECK: i32
  // CHECK: i64
  fprintf(stderr, "num elements: %" PRIdPTR "\n",
          mlirLLVMStructTypeGetNumElementTypes(literal));
  for (intptr_t i = 0; i < 3; ++i) {
    mlirTypeDump(mlirLLVMStructTypeGetElementType(literal, i));
  }

  if (!mlirTypeEqual(
          mlirLLVMStructTypeLiteralGet(ctx, 1, &i32, /*isPacked=*/false),
          mlirLLVMStructTypeLiteralGet(ctx, 1, &i32, /*isPacked=*/false))) {
    return 1;
  }
  if (mlirTypeEqual(
          mlirLLVMStructTypeLiteralGet(ctx, 1, &i32, /*isPacked=*/false),
          mlirLLVMStructTypeLiteralGet(ctx, 1, &i64, /*isPacked=*/false))) {
    return 2;
  }

  // CHECK: !llvm.struct<"foo", opaque>
  // CHECK: !llvm.struct<"bar", opaque>
  mlirTypeDump(mlirLLVMStructTypeIdentifiedGet(
      ctx, mlirStringRefCreateFromCString("foo")));
  mlirTypeDump(mlirLLVMStructTypeIdentifiedGet(
      ctx, mlirStringRefCreateFromCString("bar")));

  if (!mlirTypeEqual(mlirLLVMStructTypeIdentifiedGet(
                         ctx, mlirStringRefCreateFromCString("foo")),
                     mlirLLVMStructTypeIdentifiedGet(
                         ctx, mlirStringRefCreateFromCString("foo")))) {
    return 3;
  }
  if (mlirTypeEqual(mlirLLVMStructTypeIdentifiedGet(
                        ctx, mlirStringRefCreateFromCString("foo")),
                    mlirLLVMStructTypeIdentifiedGet(
                        ctx, mlirStringRefCreateFromCString("bar")))) {
    return 4;
  }

  MlirType fooStruct = mlirLLVMStructTypeIdentifiedGet(
      ctx, mlirStringRefCreateFromCString("foo"));
  MlirStringRef name = mlirLLVMStructTypeGetIdentifier(fooStruct);
  if (memcmp(name.data, "foo", name.length))
    return 5;
  if (!mlirLLVMStructTypeIsOpaque(fooStruct))
    return 6;

  MlirType i32_i64[] = {i32, i64};
  MlirLogicalResult result =
      mlirLLVMStructTypeSetBody(fooStruct, sizeof(i32_i64) / sizeof(MlirType),
                                i32_i64, /*isPacked=*/false);
  if (!mlirLogicalResultIsSuccess(result))
    return 7;

  // CHECK: !llvm.struct<"foo", (i32, i64)>
  mlirTypeDump(fooStruct);
  if (mlirLLVMStructTypeIsOpaque(fooStruct))
    return 8;
  if (mlirLLVMStructTypeIsPacked(fooStruct))
    return 9;
  if (!mlirTypeEqual(mlirLLVMStructTypeIdentifiedGet(
                         ctx, mlirStringRefCreateFromCString("foo")),
                     fooStruct)) {
    return 10;
  }

  MlirType barStruct = mlirLLVMStructTypeIdentifiedGet(
      ctx, mlirStringRefCreateFromCString("bar"));
  result = mlirLLVMStructTypeSetBody(barStruct, 1, &i32, /*isPacked=*/true);
  if (!mlirLogicalResultIsSuccess(result))
    return 11;

  // CHECK: !llvm.struct<"bar", packed (i32)>
  mlirTypeDump(barStruct);
  if (!mlirLLVMStructTypeIsPacked(barStruct))
    return 12;

  // Same body, should succeed.
  result =
      mlirLLVMStructTypeSetBody(fooStruct, sizeof(i32_i64) / sizeof(MlirType),
                                i32_i64, /*isPacked=*/false);
  if (!mlirLogicalResultIsSuccess(result))
    return 13;

  // Different body, should fail.
  result = mlirLLVMStructTypeSetBody(fooStruct, 1, &i32, /*isPacked=*/false);
  if (mlirLogicalResultIsSuccess(result))
    return 14;

  // Packed flag differs, should fail.
  result = mlirLLVMStructTypeSetBody(barStruct, 1, &i32, /*isPacked=*/false);
  if (mlirLogicalResultIsSuccess(result))
    return 15;

  // Should have a different name.
  // CHECK: !llvm.struct<"foo{{[^"]+}}
  mlirTypeDump(mlirLLVMStructTypeIdentifiedNewGet(
      ctx, mlirStringRefCreateFromCString("foo"), /*nFieldTypes=*/0,
      /*fieldTypes=*/NULL, /*isPacked=*/false));

  // Two freshly created "new" types must differ.
  if (mlirTypeEqual(
          mlirLLVMStructTypeIdentifiedNewGet(
              ctx, mlirStringRefCreateFromCString("foo"), /*nFieldTypes=*/0,
              /*fieldTypes=*/NULL, /*isPacked=*/false),
          mlirLLVMStructTypeIdentifiedNewGet(
              ctx, mlirStringRefCreateFromCString("foo"), /*nFieldTypes=*/0,
              /*fieldTypes=*/NULL, /*isPacked=*/false))) {
    return 16;
  }

  MlirType opaque = mlirLLVMStructTypeOpaqueGet(
      ctx, mlirStringRefCreateFromCString("opaque"));
  // CHECK: !llvm.struct<"opaque", opaque>
  mlirTypeDump(opaque);
  if (!mlirLLVMStructTypeIsOpaque(opaque))
    return 17;

  return 0;
}

// CHECK-LABEL: testLLVMAttributes
static void testLLVMAttributes(MlirContext ctx) {
  fprintf(stderr, "testLLVMAttributes\n");

  // CHECK: #llvm.linkage<internal>
  mlirAttributeDump(mlirLLVMLinkageAttrGet(ctx, MlirLLVMLinkageInternal));
  // CHECK: #llvm.cconv<ccc>
  mlirAttributeDump(mlirLLVMCConvAttrGet(ctx, MlirLLVMCConvC));
  // CHECK: #llvm<comdat any>
  mlirAttributeDump(mlirLLVMComdatAttrGet(ctx, MlirLLVMComdatAny));
}

// CHECK-LABEL: testDebugInfoAttributes
static void testDebugInfoAttributes(MlirContext ctx) {
  fprintf(stderr, "testDebugInfoAttributes\n");

  MlirAttribute foo =
      mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("foo"));
  MlirAttribute bar =
      mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("bar"));
  MlirAttribute id = mlirDisctinctAttrCreate(foo);

  // CHECK: #llvm.di_null_type
  mlirAttributeDump(mlirLLVMDINullTypeAttrGet(ctx));

  // CHECK: #llvm.di_basic_type<tag = DW_TAG_null, name = "foo", sizeInBits =
  // CHECK-SAME: 64, encoding = DW_ATE_signed>
  MlirAttribute di_type =
      mlirLLVMDIBasicTypeAttrGet(ctx, 0, foo, 64, MlirLLVMTypeEncodingSigned);
  mlirAttributeDump(di_type);

  MlirAttribute file = mlirLLVMDIFileAttrGet(ctx, foo, bar);

  // CHECK: #llvm.di_file<"foo" in "bar">
  mlirAttributeDump(file);

  MlirAttribute compile_unit = mlirLLVMDICompileUnitAttrGet(
      ctx, id, LLVMDWARFSourceLanguageC99, file, foo, false,
      MlirLLVMDIEmissionKindFull, MlirLLVMDINameTableKindDefault);

  // CHECK: #llvm.di_compile_unit<{{.*}}>
  mlirAttributeDump(compile_unit);

  MlirAttribute di_module = mlirLLVMDIModuleAttrGet(
      ctx, file, compile_unit, foo,
      mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("")), bar, foo, 1,
      0);
  // CHECK: #llvm.di_module<{{.*}}>
  mlirAttributeDump(di_module);

  // CHECK: #llvm.di_compile_unit<{{.*}}>
  mlirAttributeDump(mlirLLVMDIModuleAttrGetScope(di_module));

  // CHECK: 1 : i32
  mlirAttributeDump(mlirLLVMDIFlagsAttrGet(ctx, 0x1));

  // CHECK: #llvm.di_lexical_block<{{.*}}>
  mlirAttributeDump(
      mlirLLVMDILexicalBlockAttrGet(ctx, compile_unit, file, 1, 2));

  // CHECK: #llvm.di_lexical_block_file<{{.*}}>
  mlirAttributeDump(
      mlirLLVMDILexicalBlockFileAttrGet(ctx, compile_unit, file, 3));

  // CHECK: #llvm.di_local_variable<{{.*}}>
  MlirAttribute local_var = mlirLLVMDILocalVariableAttrGet(
      ctx, compile_unit, foo, file, 1, 0, 8, di_type);
  mlirAttributeDump(local_var);
  // CHECK: #llvm.di_derived_type<{{.*}}>
  // CHECK-NOT: dwarfAddressSpace
  mlirAttributeDump(mlirLLVMDIDerivedTypeAttrGet(
      ctx, 0, bar, di_type, 64, 8, 0, MLIR_CAPI_DWARF_ADDRESS_SPACE_NULL,
      di_type));

  // CHECK: #llvm.di_derived_type<{{.*}} dwarfAddressSpace = 3{{.*}}>
  mlirAttributeDump(
      mlirLLVMDIDerivedTypeAttrGet(ctx, 0, bar, di_type, 64, 8, 0, 3, di_type));

  MlirAttribute subroutine_type =
      mlirLLVMDISubroutineTypeAttrGet(ctx, 0x0, 1, &di_type);

  // CHECK: #llvm.di_subroutine_type<{{.*}}>
  mlirAttributeDump(subroutine_type);

  MlirAttribute di_subprogram =
      mlirLLVMDISubprogramAttrGet(ctx, id, compile_unit, compile_unit, foo, bar,
                                  file, 1, 2, 0, subroutine_type);
  // CHECK: #llvm.di_subprogram<{{.*}}>
  mlirAttributeDump(di_subprogram);

  // CHECK: #llvm.di_compile_unit<{{.*}}>
  mlirAttributeDump(mlirLLVMDISubprogramAttrGetScope(di_subprogram));

  // CHECK: #llvm.di_file<{{.*}}>
  mlirAttributeDump(mlirLLVMDISubprogramAttrGetFile(di_subprogram));

  // CHECK: #llvm.di_subroutine_type<{{.*}}>
  mlirAttributeDump(mlirLLVMDISubprogramAttrGetType(di_subprogram));

  MlirAttribute expression_elem =
      mlirLLVMDIExpressionElemAttrGet(ctx, 1, 1, &(uint64_t){1});

  // CHECK: #llvm<di_expression_elem(1)>
  mlirAttributeDump(expression_elem);

  MlirAttribute expression =
      mlirLLVMDIExpressionAttrGet(ctx, 1, &expression_elem);
  // CHECK: #llvm.di_expression<[(1)]>
  mlirAttributeDump(expression);

  MlirAttribute string_type =
      mlirLLVMDIStringTypeAttrGet(ctx, 0x0, foo, 16, 0, local_var, expression,
                                  expression, MlirLLVMTypeEncodingSigned);
  // CHECK: #llvm.di_string_type<{{.*}}>
  mlirAttributeDump(string_type);

  // CHECK: #llvm.di_composite_type<{{.*}}>
  mlirAttributeDump(mlirLLVMDICompositeTypeAttrGet(
      ctx, 0, id, foo, file, 1, compile_unit, di_type, 0, 64, 8, 1, &di_type,
      expression, expression, expression, expression));
}

int main(void) {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__llvm__(), ctx);
  mlirContextGetOrLoadDialect(ctx, mlirStringRefCreateFromCString("llvm"));
  testTypeCreation(ctx);
  int result = testStructTypeCreation(ctx);
  testLLVMAttributes(ctx);
  testDebugInfoAttributes(ctx);
  mlirContextDestroy(ctx);
  if (result)
    fprintf(stderr, "FAILED: code %d", result);
  return result;
}
