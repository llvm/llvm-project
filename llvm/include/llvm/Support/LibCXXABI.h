//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file provides utility functions for interacting with the libc++abi
/// runtime.
///
/// This header defines helper functions used for handling libc++abi-specific
/// runtime operations, such as judging if a symbol name is the name of vtable,
/// type information or type name string.
///
/// The utilities provided here are useful for transformations that require
/// analyzing or modifying libc++abi-specific constructs.
//
//===----------------------------------------------------------------------===//
//
#ifndef LLVM_SUPPORT_LIBCXXABI_H
#define LLVM_SUPPORT_LIBCXXABI_H

#include "llvm/IR/DataLayout.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {


/// Abstract interface for handling C++ ABI (Application Binary Interface) specifics.
///
/// This class provides an abstraction for interacting with different C++ ABIs,
/// particularly in the context of name mangling and vtable layout.
/// This allows the CXX ABI-aware optimizations to correctly identify and transform
/// RTTI data and vtables.
///
/// Note this is an abstract class that should be subclassed to provide
/// implementation details for a specific C++ ABI such as Itanium or MSVC.
class CXXABI {

  virtual const char * getVTablePrefix() = 0;
  virtual const char * getTypeNamePrefix() = 0;
  virtual const char * getTypeInfoPrefix() = 0;

public:
  static std::unique_ptr<CXXABI> Create(Triple &TT);
  virtual ~CXXABI() {}

  /// Return the offset from the type info slot to its address point
  /// in the vtable.
  virtual int64_t
  getOffsetFromTypeInfoSlotToAddressPoint(const DataLayout &DT) = 0;

  bool isVTable(StringRef Name) { return Name.starts_with(getVTablePrefix()); }
  bool isTypeName(StringRef Name) {
    return Name.starts_with(getTypeNamePrefix());
  }
  bool isTypeInfo(StringRef Name) {
    return Name.starts_with(getTypeInfoPrefix());
  }

  /// Return the name of type name string from the given name of the
  /// type info.
  std::string getTypeNameFromTypeInfo(StringRef TypeInfo);

  /// Return the name of the type info from the given name of the vtable.
  std::string getTypeInfoFromVTable(StringRef VTable);
};

/// Implements C++ ABI support for the Itanium ABI.
///
/// This class provides functionality specific to the Itanium C++ ABI.
/// It extends the `CXXABI` interface to implement ABI-specific operations.
///
/// See https://itanium-cxx-abi.github.io/cxx-abi/abi.html#rtti
class Itanium final : public CXXABI {

  const char * getVTablePrefix() override { return "_ZTV"; }
  const char * getTypeNamePrefix() override { return "_ZTS"; }
  const char * getTypeInfoPrefix() override { return "_ZTI"; }

public:
  virtual ~Itanium() {}

  int64_t
  getOffsetFromTypeInfoSlotToAddressPoint(const DataLayout &DL) override {
    return -2 * static_cast<int64_t>(DL.getPointerSize());
  }
};

} // namespace llvm
#endif
