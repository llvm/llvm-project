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
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::ptr;

//===----------------------------------------------------------------------===//
// Pointer type
//===----------------------------------------------------------------------===//

constexpr const static unsigned kDefaultPointerSizeBits = 64;
constexpr const static unsigned kBitsInByte = 8;
constexpr const static unsigned kDefaultPointerAlignment = 8;

/// Returns the part of the data layout entry that corresponds to `pos` for the
/// given `type` by interpreting the list of entries `params`. For the pointer
/// type in the default address space, returns the default value if the entries
/// do not provide a custom one, for other address spaces returns std::nullopt.
static std::optional<uint64_t>
getPointerDataLayoutEntry(DataLayoutEntryListRef params, PtrType type,
                          PtrDLEntryPos pos) {
  // First, look for the entry for the pointer in the current address space.
  Attribute currentEntry;
  for (DataLayoutEntryInterface entry : params) {
    if (!entry.isTypeEntry())
      continue;
    if (cast<PtrType>(entry.getKey().get<Type>()).getAddressSpace() ==
        type.getAddressSpace()) {
      currentEntry = entry.getValue();
      break;
    }
  }
  if (currentEntry) {
    std::optional<uint64_t> value = extractPointerSpecValue(currentEntry, pos);
    // If the optional `PtrDLEntryPos::Index` entry is not available, use the
    // pointer size as the index bitwidth.
    if (!value && pos == PtrDLEntryPos::Index)
      value = extractPointerSpecValue(currentEntry, PtrDLEntryPos::Size);
    bool isSizeOrIndex =
        pos == PtrDLEntryPos::Size || pos == PtrDLEntryPos::Index;
    return *value / (isSizeOrIndex ? 1 : kBitsInByte);
  }

  // If not found, and this is the pointer to the default memory space, assume
  // 64-bit pointers.
  if (type.getAddressSpace() == 0) {
    bool isSizeOrIndex =
        pos == PtrDLEntryPos::Size || pos == PtrDLEntryPos::Index;
    return isSizeOrIndex ? kDefaultPointerSizeBits : kDefaultPointerAlignment;
  }

  return std::nullopt;
}

Dialect *PtrType::getAliasDialect() const {
  if (auto iface =
          mlir::dyn_cast_or_null<MemorySpaceAttrInterface>(getMemorySpace()))
    if (auto dialect = iface.getMemorySpaceDialect())
      return dialect;
  return &getDialect();
}

MemoryModel PtrType::getMemoryModel() const { return getMemorySpace(); }

int64_t PtrType::getAddressSpace() const {
  return getMemoryModel().getAddressSpace();
}

Attribute PtrType::getDefaultMemorySpace() const {
  return getMemoryModel().getDefaultMemorySpace();
}

bool PtrType::areCompatible(DataLayoutEntryListRef oldLayout,
                            DataLayoutEntryListRef newLayout) const {
  for (DataLayoutEntryInterface newEntry : newLayout) {
    if (!newEntry.isTypeEntry())
      continue;
    unsigned size = kDefaultPointerSizeBits;
    unsigned abi = kDefaultPointerAlignment;
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
      llvm::find_if(oldLayout, [&](DataLayoutEntryInterface entry) {
        if (auto type = llvm::dyn_cast_if_present<Type>(entry.getKey())) {
          return llvm::cast<PtrType>(type).getAddressSpace() == 0;
        }
        return false;
      });
    }
    if (it != oldLayout.end()) {
      size = *extractPointerSpecValue(*it, PtrDLEntryPos::Size);
      abi = *extractPointerSpecValue(*it, PtrDLEntryPos::Abi);
    }

    Attribute newSpec = llvm::cast<DenseIntElementsAttr>(newEntry.getValue());
    unsigned newSize = *extractPointerSpecValue(newSpec, PtrDLEntryPos::Size);
    unsigned newAbi = *extractPointerSpecValue(newSpec, PtrDLEntryPos::Abi);
    if (size != newSize || abi < newAbi || abi % newAbi != 0)
      return false;
  }
  return true;
}

uint64_t PtrType::getABIAlignment(const DataLayout &dataLayout,
                                  DataLayoutEntryListRef params) const {
  if (std::optional<uint64_t> alignment =
          getPointerDataLayoutEntry(params, *this, PtrDLEntryPos::Abi))
    return *alignment;

  return dataLayout.getTypeABIAlignment(
      get(getContext(), getDefaultMemorySpace()));
}

std::optional<uint64_t>
PtrType::getIndexBitwidth(const DataLayout &dataLayout,
                          DataLayoutEntryListRef params) const {
  if (std::optional<uint64_t> indexBitwidth =
          getPointerDataLayoutEntry(params, *this, PtrDLEntryPos::Index))
    return *indexBitwidth;

  return dataLayout.getTypeIndexBitwidth(
      get(getContext(), getDefaultMemorySpace()));
}

llvm::TypeSize PtrType::getTypeSizeInBits(const DataLayout &dataLayout,
                                          DataLayoutEntryListRef params) const {
  if (std::optional<uint64_t> size =
          getPointerDataLayoutEntry(params, *this, PtrDLEntryPos::Size))
    return llvm::TypeSize::getFixed(*size);

  // For other memory spaces, use the size of the pointer to the default memory
  // space.
  return dataLayout.getTypeSizeInBits(
      get(getContext(), getDefaultMemorySpace()));
}

uint64_t PtrType::getPreferredAlignment(const DataLayout &dataLayout,
                                        DataLayoutEntryListRef params) const {
  if (std::optional<uint64_t> alignment =
          getPointerDataLayoutEntry(params, *this, PtrDLEntryPos::Preferred))
    return *alignment;

  return dataLayout.getTypePreferredAlignment(
      get(getContext(), getDefaultMemorySpace()));
}

std::optional<uint64_t> mlir::ptr::extractPointerSpecValue(Attribute attr,
                                                           PtrDLEntryPos pos) {
  auto spec = cast<DenseIntElementsAttr>(attr);
  auto idx = static_cast<int64_t>(pos);
  if (idx >= spec.size())
    return std::nullopt;
  return spec.getValues<uint64_t>()[idx];
}

LogicalResult PtrType::verifyEntries(DataLayoutEntryListRef entries,
                                     Location loc) const {
  for (DataLayoutEntryInterface entry : entries) {
    if (!entry.isTypeEntry())
      continue;
    auto key = entry.getKey().get<Type>();
    auto values = llvm::dyn_cast<DenseIntElementsAttr>(entry.getValue());
    if (!values || (values.size() != 3 && values.size() != 4)) {
      return emitError(loc)
             << "expected layout attribute for " << key
             << " to be a dense integer elements attribute with 3 or 4 "
                "elements";
    }
    if (!values.getElementType().isInteger(64))
      return emitError(loc) << "expected i64 parameters for " << key;

    if (extractPointerSpecValue(values, PtrDLEntryPos::Abi) >
        extractPointerSpecValue(values, PtrDLEntryPos::Preferred)) {
      return emitError(loc) << "preferred alignment is expected to be at least "
                               "as large as ABI alignment";
    }
  }
  return success();
}
