//===- Target/DirectX/DXILPointerType.h - DXIL Typed Pointer Type ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_DIRECTX_DXILPOINTERTYPE_H
#define LLVM_TARGET_DIRECTX_DXILPOINTERTYPE_H

#include "llvm/IR/Type.h"

namespace llvm {
namespace dxil {

// DXIL has typed pointers, this pointer type abstraction is used for tracking
// in PointerTypeAnalysis and for the bitcode ValueEnumerator
class TypedPointerType : public Type {
  explicit TypedPointerType(Type *ElType, unsigned AddrSpace);

  Type *PointeeTy;

public:
  TypedPointerType(const TypedPointerType &) = delete;
  TypedPointerType &operator=(const TypedPointerType &) = delete;

  /// This constructs a pointer to an object of the specified type in a numbered
  /// address space.
  static TypedPointerType *get(Type *ElementType, unsigned AddressSpace);

  /// Return true if the specified type is valid as a element type.
  static bool isValidElementType(Type *ElemTy);

  /// Return the address space of the Pointer type.
  unsigned getAddressSpace() const { return getSubclassData(); }

  Type *getElementType() const { return PointeeTy; }

  /// Implement support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Type *T) {
    return T->getTypeID() == DXILPointerTyID;
  }
};

} // namespace dxil
} // namespace llvm

#endif // LLVM_TARGET_DIRECTX_DXILPOINTERTYPE_H
