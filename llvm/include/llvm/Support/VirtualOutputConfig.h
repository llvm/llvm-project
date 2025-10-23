//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declarations of the OutputConfig class.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_VIRTUALOUTPUTCONFIG_H
#define LLVM_SUPPORT_VIRTUALOUTPUTCONFIG_H

namespace llvm {

class raw_ostream;

namespace sys::fs {
enum OpenFlags : unsigned;
} // end namespace sys::fs

namespace vfs {

namespace detail {
/// Unused and empty base class to allow OutputConfig constructor to be
/// constexpr, with commas before every field's initializer.
struct EmptyBaseClass {};
} // namespace detail

/// Full configuration for an output for use by the \a OutputBackend. Each
/// configuration flag is either \c true or \c false.
struct OutputConfig : detail::EmptyBaseClass {
public:
  void print(raw_ostream &OS) const;
  void dump() const;

#define HANDLE_OUTPUT_CONFIG_FLAG(NAME, DEFAULT)                               \
  constexpr bool get##NAME() const { return NAME; }                            \
  constexpr bool getNo##NAME() const { return !NAME; }                         \
  constexpr OutputConfig &set##NAME(bool Value) {                              \
    NAME = Value;                                                              \
    return *this;                                                              \
  }                                                                            \
  constexpr OutputConfig &set##NAME() { return set##NAME(true); }              \
  constexpr OutputConfig &setNo##NAME() { return set##NAME(false); }
#include "llvm/Support/VirtualOutputConfig.def"

  constexpr OutputConfig &setBinary() { return setNoText().setNoCRLF(); }
  constexpr OutputConfig &setTextWithCRLF() { return setText().setCRLF(); }
  constexpr OutputConfig &setTextWithCRLF(bool Value) {
    return Value ? setText().setCRLF() : setBinary();
  }
  constexpr bool getTextWithCRLF() const { return getText() && getCRLF(); }
  constexpr bool getBinary() const { return !getText(); }

  /// Updates Text and CRLF flags based on \a sys::fs::OF_Text and \a
  /// sys::fs::OF_CRLF in \p Flags. Rejects CRLF without Text (calling
  /// \a setBinary()).
  OutputConfig &setOpenFlags(const sys::fs::OpenFlags &Flags);

  constexpr OutputConfig()
      : EmptyBaseClass()
#define HANDLE_OUTPUT_CONFIG_FLAG(NAME, DEFAULT) , NAME(DEFAULT)
#include "llvm/Support/VirtualOutputConfig.def"
  {
  }

  bool operator==(OutputConfig RHS) const {
#define HANDLE_OUTPUT_CONFIG_FLAG(NAME, DEFAULT)                               \
  if (NAME != RHS.NAME)                                                        \
    return false;
#include "llvm/Support/VirtualOutputConfig.def"
    return true;
  }
  bool operator!=(OutputConfig RHS) const { return !operator==(RHS); }

private:
#define HANDLE_OUTPUT_CONFIG_FLAG(NAME, DEFAULT) bool NAME : 1;
#include "llvm/Support/VirtualOutputConfig.def"
};

} // namespace vfs

raw_ostream &operator<<(raw_ostream &OS, vfs::OutputConfig Config);

} // namespace llvm

#endif // LLVM_SUPPORT_VIRTUALOUTPUTCONFIG_H
