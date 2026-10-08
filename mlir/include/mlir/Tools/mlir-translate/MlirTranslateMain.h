//===- MlirTranslateMain.h - MLIR Translation Driver main -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-translate for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRTRANSLATE_MLIRTRANSLATEMAIN_H
#define MLIR_TOOLS_MLIRTRANSLATE_MLIRTRANSLATEMAIN_H

#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/ToolUtilities.h"
#include "llvm/ADT/StringRef.h"

#include <memory>
#include <string>
#include <utility>

namespace llvm {
class MemoryBuffer;
class raw_ostream;
} // namespace llvm

namespace mlir {
class Translation;

/// Configuration options for the mlir-translate driver.
class MlirTranslateMainConfig {
public:
  MlirTranslateMainConfig &allowUnregisteredDialects(bool allow) {
    allowUnregisteredDialectsFlag = allow;
    return *this;
  }
  bool shouldAllowUnregisteredDialects() const {
    return allowUnregisteredDialectsFlag;
  }

  MlirTranslateMainConfig &errorDiagnosticsOnly(bool errorOnly) {
    errorDiagnosticsOnlyFlag = errorOnly;
    return *this;
  }
  bool shouldEmitErrorDiagnosticsOnly() const {
    return errorDiagnosticsOnlyFlag;
  }

  MlirTranslateMainConfig &
  splitInputFile(std::string splitMarker = kDefaultSplitMarker) {
    inputSplitMarker = std::move(splitMarker);
    return *this;
  }
  StringRef getInputSplitMarker() const { return inputSplitMarker; }

  MlirTranslateMainConfig &outputSplitMarker(std::string splitMarker) {
    outputSplitMarkerFlag = std::move(splitMarker);
    return *this;
  }
  StringRef getOutputSplitMarker() const { return outputSplitMarkerFlag; }

  MlirTranslateMainConfig &
  verifyDiagnostics(SourceMgrDiagnosticVerifierHandler::Level level) {
    verifyDiagnosticsLevel = level;
    return *this;
  }
  SourceMgrDiagnosticVerifierHandler::Level getVerifyDiagnosticsLevel() const {
    return verifyDiagnosticsLevel;
  }

private:
  bool allowUnregisteredDialectsFlag = false;
  bool errorDiagnosticsOnlyFlag = false;
  std::string inputSplitMarker;
  std::string outputSplitMarkerFlag;
  SourceMgrDiagnosticVerifierHandler::Level verifyDiagnosticsLevel =
      SourceMgrDiagnosticVerifierHandler::Level::None;
};

/// Apply the requested translations to an input buffer and write the result to
/// the output stream.
LogicalResult mlirTranslateMain(std::unique_ptr<llvm::MemoryBuffer> input,
                                llvm::raw_ostream &output,
                                ArrayRef<const Translation *> translations,
                                const MlirTranslateMainConfig &config = {});

/// Translate to/from an MLIR module from/to an external representation (e.g.
/// LLVM IR, SPIRV binary, ...). This is the entry point for the implementation
/// of tools like `mlir-translate`. The translation to perform is parsed from
/// the command line. The `toolName` argument is used for the header displayed
/// by `--help`.
LogicalResult mlirTranslateMain(int argc, char **argv, StringRef toolName);
} // namespace mlir

#endif // MLIR_TOOLS_MLIRTRANSLATE_MLIRTRANSLATEMAIN_H
