// ===- PartTensorDialect.cpp - part_tensor dialect implementation -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/PartTensor/IR/PartTensor.h"
#include <utility>

#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/PartTensor/IR/PartTensorAttrDefs.cpp.inc"

using namespace mlir;
using namespace mlir::part_tensor;

void PartTensorDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/PartTensor/IR/PartTensorAttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/PartTensor/IR/PartTensorOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/PartTensor/IR/PartTensorOps.cpp.inc"

#include "mlir/Dialect/PartTensor/IR/PartTensorOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// convenience methods.
//===----------------------------------------------------------------------===//

PartTensorEncodingAttr mlir::part_tensor::getPartTensorEncoding(Type type) {
  if (auto ttp = llvm::dyn_cast<RankedTensorType>(type))
    return llvm::dyn_cast_or_null<PartTensorEncodingAttr>(ttp.getEncoding());
  return nullptr;
}

/*
void PartitionEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<{";
  printer << "partConst = " << getpartConst();
  printer << " }>";
}

LogicalResult PartitionEncodingAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    unsigned partConst) {
        return success();
}
*/
