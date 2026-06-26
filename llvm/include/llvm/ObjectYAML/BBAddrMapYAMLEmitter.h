//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Shared yaml2obj BBAddrMap writer, in a standalone header so type-only
/// includers don't pull in its dependencies.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECTYAML_BBADDRMAPYAMLEMITTER_H
#define LLVM_OBJECTYAML_BBADDRMAPYAMLEMITTER_H

#include "llvm/ADT/bit.h"
#include "llvm/ObjectYAML/BBAddrMapYAML.h"
#include <cassert>
#include <cstdint>

namespace llvm {
namespace BBAddrMapYAML {

/// CRTP interface for emitting the BBAddrMap payload to a target-specific
/// writer, so the encode logic can be shared across formats.
template <typename Derived> class Writer {
  llvm::endianness Endian;
  unsigned AddressSize;

  Derived &derived() { return static_cast<Derived &>(*this); }

public:
  Writer(llvm::endianness Endian, unsigned AddressSize)
      : Endian(Endian), AddressSize(AddressSize) {
    assert((AddressSize == 4 || AddressSize == 8) && "invalid address size");
  }

  template <typename T> void writeInt(T Val) {
    derived().template emitInt<T>(Val, Endian);
  }

  // Pointer-sized: 4 or 8 bytes per AddressSize.
  void writeAddress(uint64_t Val) {
    if (AddressSize == 8)
      writeInt<uint64_t>(Val);
    else
      writeInt<uint32_t>(static_cast<uint32_t>(Val));
  }

  void writeULEB128(uint64_t Val) { derived().emitULEB128(Val); }
};

} // end namespace BBAddrMapYAML
} // end namespace llvm

#endif // LLVM_OBJECTYAML_BBADDRMAPYAMLEMITTER_H
