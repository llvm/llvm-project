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

#ifndef LLVM_LIB_TARGET_SPIRV_COMMANDLINE_H
#define LLVM_LIB_TARGET_SPIRV_COMMANDLINE_H

#include "MCTargetDesc/SPIRVBaseInfo.h"
#include "llvm/Support/CommandLine.h"
#include <string>

namespace llvm {
class StringRef;
class Triple;

/// Command line parser for toggling SPIR-V extensions.
struct SPIRVExtensionsParser : public cl::parser<ExtensionSet> {
public:
  SPIRVExtensionsParser(cl::Option &O) : cl::parser<ExtensionSet>(O) {}

  /// Parses SPIR-V extension name from CLI arguments.
  ///
  /// \return Returns true on error.
  bool parse(cl::Option &O, StringRef ArgName, StringRef ArgValue,
             ExtensionSet &Vals);

  /// Validates and converts extension names into internal enum values.
  ///
  /// \return Returns a reference to the unknown SPIR-V extension name from the
  /// list if present, or an empty StringRef on success.
  static StringRef checkExtensions(const std::vector<std::string> &ExtNames,
                                   ExtensionSet &AllowedExtensions);

  /// Returns the list of extensions that are valid for a particular
  /// target environment (i.e., OpenCL or Vulkan).
  static ExtensionSet getValidExtensions(const Triple &TT);

private:
  static ExtensionSet DisabledExtensions;
};

} // namespace llvm
#endif // LLVM_LIB_TARGET_SPIRV_COMMANDLINE_H
