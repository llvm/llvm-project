//===- RuntimeVerifiableOpInterface.cpp - Op Verification -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/RuntimeVerifiableOpInterface.h"

namespace mlir {
class Location;
class OpBuilder;

/// Generate an error message string for the given op and the specified error.
std::string
RuntimeVerifiableOpInterface::generateErrorMessage(Operation *op,
                                                   const std::string &msg) {
  std::string buffer;
  llvm::raw_string_ostream stream(buffer);
  OpPrintingFlags flags;
  // We may generate a lot of error messages and so we need to ensure the
  // printing is fast.
  flags.elideLargeElementsAttrs();
  flags.printGenericOpForm();
  flags.skipRegions();
  flags.useLocalScope();
  stream << "ERROR: Runtime op verification failed\n";
  op->print(stream, flags);
  stream << "\n^ " << msg;
  stream << "\nLocation: ";
  op->getLoc().print(stream);
  return stream.str();
}
} // namespace mlir

/// Include the definitions of the interface.
#include "mlir/Interfaces/RuntimeVerifiableOpInterface.cpp.inc"
