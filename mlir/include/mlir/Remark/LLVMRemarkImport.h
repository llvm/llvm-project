//===- LLVMRemarkImport.h - Import LLVM remarks into MLIR -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to import LLVM optimization remarks into the MLIR remark engine,
// either as they are emitted by LLVM passes or from a serialized remark file.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_REMARK_LLVMREMARKIMPORT_H
#define MLIR_REMARK_LLVMREMARKIMPORT_H

#include "mlir/IR/Location.h"
#include "mlir/IR/Remarks.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/DiagnosticHandler.h"
#include "llvm/Remarks/RemarkFormat.h"

namespace mlir {
class Operation;

namespace remark {

/// Prefix of the category of remarks imported from LLVM. The category is the
/// prefix followed by the LLVM pass name, e.g. `llvm-inline`, so that remark
/// filters can select LLVM remarks (`llvm-.*`) or a single LLVM pass.
constexpr llvm::StringLiteral llvmRemarkCategoryPrefix = "llvm-";

/// Parses `buffer`, a serialized LLVM remark file, and reports its remarks into
/// the remark engine of the context of `anchor`, the operation the remarks
/// refer to. The category is the prefixed LLVM pass name, the LLVM remark name
/// becomes the remark name and the message is stored under the "Remark"
/// argument. A remark is attached to its debug location if it has one,
/// otherwise to the symbol named after its LLVM function inside `anchor`,
/// otherwise to `anchor`. Fails if the buffer cannot be parsed.
LogicalResult importLLVMRemarks(Operation *anchor, StringRef buffer,
                                llvm::remarks::Format format);

/// Diagnostic handler forwarding LLVM diagnostics to MLIR. Optimization remarks
/// enabled in the remark engine of the context of `anchor` are imported as
/// described for `importLLVMRemarks`, so the remark filters of the engine
/// select LLVM remarks by their prefixed pass name. Other remarks are left to
/// LLVM, which prints those enabled with its own `-pass-remarks` flags. All
/// other diagnostics are reported as MLIR diagnostics. Install the handler with
/// `RespectFilters` set, so that LLVM only forwards enabled remarks.
class LLVMToMLIRDiagnosticHandler : public llvm::DiagnosticHandler {
public:
  explicit LLVMToMLIRDiagnosticHandler(Operation *anchor);

  bool handleDiagnostics(const llvm::DiagnosticInfo &diag) override;
  bool isAnalysisRemarkEnabled(StringRef passName) const override;
  bool isMissedOptRemarkEnabled(StringRef passName) const override;
  bool isPassedOptRemarkEnabled(StringRef passName) const override;
  using llvm::DiagnosticHandler::isAnyRemarkEnabled;
  bool isAnyRemarkEnabled() const override;

private:
  Operation *anchor;
  detail::RemarkEngine *engine;
  llvm::StringMap<Location> functionLocations;
};

} // namespace remark
} // namespace mlir

#endif // MLIR_REMARK_LLVMREMARKIMPORT_H
