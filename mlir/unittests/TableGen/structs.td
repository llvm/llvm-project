//===-- structs.td - StructsGen test definition file -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "test";
}

def Test_Struct : StructAttr<"TestStruct", Test_Dialect, [
                StructFieldAttr<"sample_integer", I32Attr>,
                StructFieldAttr<"sample_float", F32Attr>,
                StructFieldAttr<"sample_elements", I32ElementsAttr>,
                StructFieldAttr<"sample_optional_integer",
                                OptionalAttr<I32Attr>>,
                StructFieldAttr<"sample_default_valued_integer",
                                DefaultValuedAttr<I32Attr, "42">>] > {
  let summary = "Structure for test data";
}
