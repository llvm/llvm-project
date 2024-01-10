//===-- PartitionAttrDefs.td - attributes definitions ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PARTTENSOR_ATTRDEFS
#define PARTTENSOR_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/PartTensor/IR/PartTensorBase.td"
include "mlir/Dialect/SparseTensor/IR/SparseTensorAttrDefs.td"
include "mlir/IR/TensorEncoding.td"

// All of the partition tensor attributes will extend this class.
class PartTensor_Attr<string name, list<Trait> traits = []>
    : AttrDef<PartTensor_Dialect, name, traits>;

// PartTensor attribute that will always return 1

def PartTensorEncodingAttr : PartTensor_Attr<"PartTensorEncoding", []> {
  let mnemonic = "encoding";
  let description = [{ an attribute to show if there is partition or not }];
  let parameters = (
      ins
      "int64_t" : $partConst,
      SparseTensorEncodingAttr: $sparseAttributes
  );
  let assemblyFormat = "`<` `{` `partConst` `=` $partConst `,` `sparseAttributes` `=` $sparseAttributes `}` `>`";
}

#endif // PARTTENSOR_ATTRDEFS
