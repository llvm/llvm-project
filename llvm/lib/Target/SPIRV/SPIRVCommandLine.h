//===--- SPIRVCommandLine.h ---- Command Line Options -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains classes and functions needed for processing, parsing, and
// using CLI options for the SPIR-V backend.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SPIRV_SPIRVCOMMANDLINE_H
#define LLVM_LIB_TARGET_SPIRV_SPIRVCOMMANDLINE_H

#include "MCTargetDesc/SPIRVBaseInfo.h"
#include <string>

namespace llvm {
template <typename T> class ArrayRef;
class Error;
class StringRef;
class Triple;

/// Parser for toggling SPIR-V extensions.
struct SPIRVExtensionsParser {
  /// Parses the value of -spirv-ext into \p Enabled and \p Disabled.
  static Error parse(StringRef ArgValue, ExtensionSet &Enabled,
                     ExtensionSet &Disabled);

  /// Validates and converts extension names into internal enum values.
  ///
  /// \return Returns a reference to the unknown SPIR-V extension name from the
  /// list if present, or an empty StringRef on success.
  static StringRef checkExtensions(ArrayRef<std::string> ExtNames,
                                   ExtensionSet &AllowedExtensions);

  /// Returns the list of extensions that are valid for a particular
  /// target environment (i.e., OpenCL or Vulkan), excluding \p Disabled.
  static ExtensionSet getValidExtensions(const Triple &TT,
                                         const ExtensionSet &Disabled);
};

} // namespace llvm
#endif // LLVM_LIB_TARGET_SPIRV_SPIRVCOMMANDLINE_H
