//===- PtrDialect.cpp - Pointer dialect ---------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Pointer dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::ptr;

//===----------------------------------------------------------------------===//
// Pointer dialect
//===----------------------------------------------------------------------===//

void PtrDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Ptr/IR/PtrOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/Ptr/IR/PtrOpsAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Ptr/IR/PtrOpsTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// FromPtrOp
//===----------------------------------------------------------------------===//

OpFoldResult FromPtrOp::fold(FoldAdaptor adaptor) {
  // Fold the pattern:
  // %ptr = ptr.to_ptr %v : type -> ptr
  // (%mda = ptr.get_metadata %v : type)?
  // %val = ptr.from_ptr %ptr (metadata %mda)? : ptr -> type
  // To:
  // %val -> %v
  auto toPtr = dyn_cast_or_null<ToPtrOp>(getPtr().getDefiningOp());
  // Cannot fold if it's not a `to_ptr` op or the initial and final types are
  // different.
  if (!toPtr || toPtr.getPtr().getType() != getType())
    return nullptr;
  Value md = getMetadata();
  if (!md)
    return toPtr.getPtr();
  // Fold if the metadata can be verified to be equal.
  if (auto mdOp = dyn_cast_or_null<GetMetadataOp>(md.getDefiningOp());
      mdOp && mdOp.getPtr() == toPtr.getPtr())
    return toPtr.getPtr();
  return nullptr;
}

LogicalResult FromPtrOp::verify() {
  if (isa<PtrType>(getType()))
    return emitError() << "the result type cannot be `!ptr.ptr`";
  if (getType().getMemorySpace() != getPtr().getType().getMemorySpace()) {
    return emitError()
           << "expected the input and output to have the same memory space";
  }
  bool hasMD = getMetadata() != Value();
  bool hasTrivialMD = getHasTrivialMetadata();
  if (hasMD && hasTrivialMD) {
    return emitError() << "expected either a metadata argument or the "
                          "`trivial_metadata` flag, not both";
  }
  if (getType().hasPtrMetadata() && !(hasMD || hasTrivialMD)) {
    return emitError() << "expected either a metadata argument or the "
                          "`trivial_metadata` flag to be set";
  }
  if (!getType().hasPtrMetadata() && (hasMD || hasTrivialMD))
    return emitError() << "expected no metadata specification";
  return success();
}

//===----------------------------------------------------------------------===//
// PtrAddOp
//===----------------------------------------------------------------------===//

/// Fold: ptradd ptr + 0 ->  ptr
OpFoldResult PtrAddOp::fold(FoldAdaptor adaptor) {
  Attribute attr = adaptor.getOffset();
  if (!attr)
    return nullptr;
  if (llvm::APInt value; m_ConstantInt(&value).match(attr) && value.isZero())
    return getBase();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// ToPtrOp
//===----------------------------------------------------------------------===//

OpFoldResult ToPtrOp::fold(FoldAdaptor adaptor) {
  // Fold the pattern:
  // %val = ptr.from_ptr %p (metadata ...)? : ptr -> type
  // %ptr = ptr.to_ptr %val : type -> ptr
  // To:
  // %ptr -> %p
  auto fromPtr = dyn_cast_or_null<FromPtrOp>(getPtr().getDefiningOp());
  // Cannot fold if it's not a `from_ptr` op.
  if (!fromPtr)
    return nullptr;
  return fromPtr.getPtr();
}

LogicalResult ToPtrOp::verify() {
  if (isa<PtrType>(getPtr().getType()))
    return emitError() << "the input value cannot be of type `!ptr.ptr`";
  if (getType().getMemorySpace() != getPtr().getType().getMemorySpace()) {
    return emitError()
           << "expected the input and output to have the same memory space";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TypeOffsetOp
//===----------------------------------------------------------------------===//

llvm::TypeSize TypeOffsetOp::getTypeSize(std::optional<DataLayout> layout) {
  if (layout)
    return layout->getTypeSize(getElementType());
  DataLayout dl = DataLayout::closest(*this);
  return dl.getTypeSize(getElementType());
}

//===----------------------------------------------------------------------===//
// Pointer API.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Ptr/IR/PtrOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Ptr/IR/PtrOpsAttrs.cpp.inc"

#include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.cpp.inc"

#include "mlir/Dialect/Ptr/IR/MemorySpaceAttrInterfaces.cpp.inc"

#include "mlir/Dialect/Ptr/IR/PtrOpsEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Ptr/IR/PtrOpsTypes.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Ptr/IR/PtrOps.cpp.inc"
