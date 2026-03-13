//===- OpenACCVariableInfo.h - OpenACC Variable Info -------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the VariableInfo classes used to carry language-specific
// variable metadata through the OpenACC type interfaces. The OpenACC dialect
// type interface methods (e.g., generatePrivateInit, generateCopy,
// generatePrivateDestroy) receive a VariableInfo that language implementations
// can dyn_cast to their own subclass to recover information not available from
// the type system alone (e.g., whether a Fortran variable is OPTIONAL).
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENACC_OPENACCVARIABLEINFO_H
#define MLIR_DIALECT_OPENACC_OPENACCVARIABLEINFO_H

#include <memory>

namespace mlir::acc {

/// Base class for language-specific variable information.
///
/// Languages should derive from this class and use LLVM-style RTTI
/// (via a `classof` static method and the `getLanguage()` discriminator) so
/// that consumers can dyn_cast to the concrete type.
///
/// See llvm/docs/HowToSetUpLLVMStyleRTTI.rst for details on the pattern.
class VariableInfoBase {
public:
  enum Language { Fortran, C, CPP };

  Language getLanguage() const { return lang; }
  virtual ~VariableInfoBase() = default;

protected:
  explicit VariableInfoBase(Language lang) : lang(lang) {}

private:
  Language lang;
};

/// A type-erased, move-only wrapper for language-specific variable
/// information. This is modeled after the PointerUnion discrimination pattern:
/// language implementations can use `dyn_cast<ConcreteType>()` to recover
/// their specific metadata.
///
/// A default-constructed VariableInfo is null and carries no information.
class VariableInfo {
public:
  VariableInfo() = default;
  VariableInfo(std::nullptr_t) {}
  explicit VariableInfo(std::unique_ptr<VariableInfoBase> impl)
      : impl(std::move(impl)) {}

  VariableInfo(VariableInfo &&) = default;
  VariableInfo &operator=(VariableInfo &&) = default;

  template <typename T>
  const T *dyn_cast() const {
    if (!impl || !T::classof(impl.get()))
      return nullptr;
    return static_cast<const T *>(impl.get());
  }

  explicit operator bool() const { return impl != nullptr; }

private:
  std::unique_ptr<VariableInfoBase> impl;
};

} // namespace mlir::acc

#endif // MLIR_DIALECT_OPENACC_OPENACCVARIABLEINFO_H
