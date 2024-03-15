//===- AttrDetail.h - Details of MLIR LLVM dialect attributes --------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains implementation details, such as storage structures, of
// MLIR LLVM dialect attributes.
//
//===----------------------------------------------------------------------===//
#ifndef DIALECT_LLVMIR_IR_ATTRDETAIL_H
#define DIALECT_LLVMIR_IR_ATTRDETAIL_H

#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace LLVM {
namespace detail {

//===----------------------------------------------------------------------===//
// DICompositeTypeAttrStorage
//===----------------------------------------------------------------------===//

struct DICompositeTypeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<unsigned, StringAttr, DIFileAttr, uint32_t,
                           DIScopeAttr, DITypeAttr, DIFlags, uint64_t, uint64_t,
                           ArrayRef<DINodeAttr>, StringAttr>;

  DICompositeTypeAttrStorage(unsigned tag, StringAttr name, DIFileAttr file,
                             uint32_t line, DIScopeAttr scope,
                             DITypeAttr baseType, DIFlags flags,
                             uint64_t sizeInBits, uint64_t alignInBits,
                             ArrayRef<DINodeAttr> elements,
                             StringAttr identifier = StringAttr())
      : tag(tag), name(name), file(file), line(line), scope(scope),
        baseType(baseType), flags(flags), sizeInBits(sizeInBits),
        alignInBits(alignInBits), elements(elements), identifier(identifier) {}

  unsigned getTag() const { return tag; }
  StringAttr getName() const { return name; }
  DIFileAttr getFile() const { return file; }
  uint32_t getLine() const { return line; }
  DIScopeAttr getScope() const { return scope; }
  DITypeAttr getBaseType() const { return baseType; }
  DIFlags getFlags() const { return flags; }
  uint64_t getSizeInBits() const { return sizeInBits; }
  uint64_t getAlignInBits() const { return alignInBits; }
  ArrayRef<DINodeAttr> getElements() const { return elements; }
  StringAttr getIdentifier() const { return identifier; }

  /// Returns true if this attribute is identified.
  bool isIdentified() const {
    return !(!identifier);
  }

  /// Returns the respective key for this attribute.
  KeyTy getAsKey() const {
    if (isIdentified())
      return KeyTy(tag, name, file, line, scope, baseType, flags, sizeInBits,
                   alignInBits, elements, identifier);

    return KeyTy(tag, name, file, line, scope, baseType, flags, sizeInBits,
                 alignInBits, elements, StringAttr());
  }

  /// Compares two keys.
  bool operator==(const KeyTy &other) const {
    if (isIdentified())
      // Just compare against the identifier.
      return identifier == std::get<10>(other);

    // Otherwise, compare the entire tuple.
    return other == getAsKey();
  }

  /// Returns the hash value of the key.
  static llvm::hash_code hashKey(const KeyTy &key) {
    const auto &[tag, name, file, line, scope, baseType, flags, sizeInBits,
                 alignInBits, elements, identifier] = key;

    if (identifier)
      // Only the identifier participates in the hash id.
      return hash_value(identifier);

    // Otherwise, everything else is included in the hash.
    return hash_combine(tag, name, file, line, scope, baseType, flags,
                              sizeInBits, alignInBits, elements);
  }

  /// Constructs new storage for an attribute.
  static DICompositeTypeAttrStorage *
  construct(AttributeStorageAllocator &allocator, const KeyTy &key) {
    auto [tag, name, file, line, scope, baseType, flags, sizeInBits,
           alignInBits, elements, identifier] = key;
    elements = allocator.copyInto(elements);
    if (identifier) {
      return new (allocator.allocate<DICompositeTypeAttrStorage>())
          DICompositeTypeAttrStorage(tag, name, file, line, scope, baseType,
                                     flags, sizeInBits, alignInBits, elements,
                                     identifier);
    }
    return new (allocator.allocate<DICompositeTypeAttrStorage>())
        DICompositeTypeAttrStorage(tag, name, file, line, scope, baseType,
                                   flags, sizeInBits, alignInBits, elements);
  }

  LogicalResult mutate(AttributeStorageAllocator &allocator,
                       const ArrayRef<DINodeAttr>& elements) {
    // Replace the elements.
    this->elements = allocator.copyInto(elements);
    return success();
  }

private:
  unsigned tag;
  StringAttr name;
  DIFileAttr file;
  uint32_t line;
  DIScopeAttr scope;
  DITypeAttr baseType;
  DIFlags flags;
  uint64_t sizeInBits;
  uint64_t alignInBits;
  ArrayRef<DINodeAttr> elements;
  StringAttr identifier;
};

} // namespace detail
} // namespace LLVM
} // namespace mlir

#endif // DIALECT_LLVMIR_IR_ATTRDETAIL_H
