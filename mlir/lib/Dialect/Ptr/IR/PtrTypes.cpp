//===- PtrTypes.cpp - Pointer dialect types ---------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Ptr dialect types.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Ptr/IR/PtrTypes.h"
#include "mlir/Dialect/Ptr/IR/PtrAttrs.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::ptr;

//===----------------------------------------------------------------------===//
// Pointer type
//===----------------------------------------------------------------------===//

constexpr const static unsigned kDefaultPointerSizeBits = 64;
constexpr const static unsigned kBitsInByte = 8;
constexpr const static unsigned kDefaultPointerAlignment = 8;

static Attribute getDefaultMemorySpace(PtrType ptr) { return nullptr; }

/// Searches the data layout for the pointer spec, returns nullptr if it is not
/// found.
static SpecAttr getPointerSpec(DataLayoutEntryListRef params, PtrType type) {
  for (DataLayoutEntryInterface entry : params) {
    if (!entry.isTypeEntry())
      continue;
    if (cast<PtrType>(entry.getKey().get<Type>()).getMemorySpace() ==
        type.getMemorySpace()) {
      if (auto spec = dyn_cast<SpecAttr>(entry.getValue()))
        return spec;
    }
  }
  // If not found, and this is the pointer to the default memory space, assume
  // 64-bit pointers.
  if (type.getMemorySpace() == getDefaultMemorySpace(type))
    return SpecAttr::get(type.getContext(), kDefaultPointerSizeBits,
                         kDefaultPointerAlignment, kDefaultPointerAlignment,
                         kDefaultPointerSizeBits);
  return nullptr;
}

bool PtrType::areCompatible(DataLayoutEntryListRef oldLayout,
                            DataLayoutEntryListRef newLayout) const {
  for (DataLayoutEntryInterface newEntry : newLayout) {
    if (!newEntry.isTypeEntry())
      continue;
    uint32_t size = kDefaultPointerSizeBits;
    uint32_t abi = kDefaultPointerAlignment;
    auto newType = llvm::cast<PtrType>(newEntry.getKey().get<Type>());
    const auto *it =
        llvm::find_if(oldLayout, [&](DataLayoutEntryInterface entry) {
          if (auto type = llvm::dyn_cast_if_present<Type>(entry.getKey())) {
            return llvm::cast<PtrType>(type).getMemorySpace() ==
                   newType.getMemorySpace();
          }
          return false;
        });
    if (it == oldLayout.end()) {
      it = llvm::find_if(oldLayout, [&](DataLayoutEntryInterface entry) {
        if (auto type = llvm::dyn_cast_if_present<Type>(entry.getKey())) {
          auto ptrTy = llvm::cast<PtrType>(type);
          return ptrTy.getMemorySpace() == getDefaultMemorySpace(ptrTy);
        }
        return false;
      });
    }
    if (it != oldLayout.end()) {
      auto spec = llvm::cast<SpecAttr>(*it);
      size = spec.getSize();
      abi = spec.getAbi();
    }

    auto newSpec = llvm::cast<SpecAttr>(newEntry.getValue());
    uint32_t newSize = newSpec.getSize();
    uint32_t newAbi = newSpec.getAbi();
    if (size != newSize || abi < newAbi || abi % newAbi != 0)
      return false;
  }
  return true;
}

uint64_t PtrType::getABIAlignment(const DataLayout &dataLayout,
                                  DataLayoutEntryListRef params) const {
  if (SpecAttr spec = getPointerSpec(params, *this))
    return spec.getAbi() / kBitsInByte;

  return dataLayout.getTypeABIAlignment(
      get(getContext(), getDefaultMemorySpace(*this)));
}

std::optional<uint64_t>
PtrType::getIndexBitwidth(const DataLayout &dataLayout,
                          DataLayoutEntryListRef params) const {
  if (SpecAttr spec = getPointerSpec(params, *this)) {
    return spec.getIndex() == SpecAttr::kOptionalSpecValue ? spec.getSize()
                                                           : spec.getIndex();
  }

  return dataLayout.getTypeIndexBitwidth(
      get(getContext(), getDefaultMemorySpace(*this)));
}

llvm::TypeSize PtrType::getTypeSizeInBits(const DataLayout &dataLayout,
                                          DataLayoutEntryListRef params) const {
  if (SpecAttr spec = getPointerSpec(params, *this))
    return llvm::TypeSize::getFixed(spec.getSize());

  // For other memory spaces, use the size of the pointer to the default memory
  // space.
  return dataLayout.getTypeSizeInBits(
      get(getContext(), getDefaultMemorySpace(*this)));
}

uint64_t PtrType::getPreferredAlignment(const DataLayout &dataLayout,
                                        DataLayoutEntryListRef params) const {
  if (SpecAttr spec = getPointerSpec(params, *this))
    return spec.getPreferred() / kBitsInByte;

  return dataLayout.getTypePreferredAlignment(
      get(getContext(), getDefaultMemorySpace(*this)));
}

LogicalResult PtrType::verifyEntries(DataLayoutEntryListRef entries,
                                     Location loc) const {
  for (DataLayoutEntryInterface entry : entries) {
    if (!entry.isTypeEntry())
      continue;
    auto key = entry.getKey().get<Type>();
    if (!llvm::isa<SpecAttr>(entry.getValue())) {
      return emitError(loc) << "expected layout attribute for " << key
                            << " to be a #ptr.spec attribute";
    }
  }
  return success();
}
