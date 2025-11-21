//===- PtrAttrs.cpp - Pointer dialect attributes ----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Ptr dialect attributes.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Ptr/IR/PtrAttrs.h"

using namespace mlir;
using namespace mlir::ptr;

constexpr const static unsigned kBitsInByte = 8;

//===----------------------------------------------------------------------===//
// GenericSpaceAttr
//===----------------------------------------------------------------------===//

bool GenericSpaceAttr::isValidLoad(
    Type type, ptr::AtomicOrdering ordering, std::optional<int64_t> alignment,
    const ::mlir::DataLayout *dataLayout,
    function_ref<InFlightDiagnostic()> emitError) const {
  return true;
}

bool GenericSpaceAttr::isValidStore(
    Type type, ptr::AtomicOrdering ordering, std::optional<int64_t> alignment,
    const ::mlir::DataLayout *dataLayout,
    function_ref<InFlightDiagnostic()> emitError) const {
  return true;
}

bool GenericSpaceAttr::isValidAtomicOp(
    ptr::AtomicBinOp op, Type type, ptr::AtomicOrdering ordering,
    std::optional<int64_t> alignment, const ::mlir::DataLayout *dataLayout,
    function_ref<InFlightDiagnostic()> emitError) const {
  return true;
}

bool GenericSpaceAttr::isValidAtomicXchg(
    Type type, ptr::AtomicOrdering successOrdering,
    ptr::AtomicOrdering failureOrdering, std::optional<int64_t> alignment,
    const ::mlir::DataLayout *dataLayout,
    function_ref<InFlightDiagnostic()> emitError) const {
  return true;
}

bool GenericSpaceAttr::isValidAddrSpaceCast(
    Type tgt, Type src, function_ref<InFlightDiagnostic()> emitError) const {
  // TODO: update this method once the `addrspace_cast` op is added to the
  // dialect.
  assert(false && "unimplemented, see TODO in the source.");
  return false;
}

bool GenericSpaceAttr::isValidPtrIntCast(
    Type intLikeTy, Type ptrLikeTy,
    function_ref<InFlightDiagnostic()> emitError) const {
  // TODO: update this method once the int-cast ops are added to the dialect.
  assert(false && "unimplemented, see TODO in the source.");
  return false;
}

//===----------------------------------------------------------------------===//
// SpecAttr
//===----------------------------------------------------------------------===//

LogicalResult SpecAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                               uint32_t size, uint32_t abi, uint32_t preferred,
                               uint32_t index) {
  if (size % kBitsInByte != 0)
    return emitError() << "size entry must be divisible by 8";
  if (abi % kBitsInByte != 0)
    return emitError() << "abi entry must be divisible by 8";
  if (preferred % kBitsInByte != 0)
    return emitError() << "preferred entry must be divisible by 8";
  if (index != kOptionalSpecValue && index % kBitsInByte != 0)
    return emitError() << "index entry must be divisible by 8";
  if (abi > preferred)
    return emitError() << "preferred alignment is expected to be at least "
                          "as large as ABI alignment";
  return success();
}
