//===- LLVMRemarkImport.cpp - Import LLVM remarks into MLIR ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Remark/LLVMRemarkImport.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/Twine.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/Remarks/Remark.h"
#include "llvm/Remarks/RemarkParser.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Locations
//===----------------------------------------------------------------------===//

/// Returns the location of the symbol `functionName` inside `anchor`, or the
/// location of `anchor` if there is no such symbol.
static Location getFunctionLocation(Operation *anchor,
                                    llvm::StringMap<Location> &cache,
                                    StringRef functionName) {
  if (functionName.empty() || !anchor->hasTrait<OpTrait::SymbolTable>())
    return anchor->getLoc();
  auto it = cache.find(functionName);
  if (it != cache.end())
    return it->second;
  Location loc = anchor->getLoc();
  if (Operation *symbol = SymbolTable::lookupSymbolIn(anchor, functionName))
    loc = symbol->getLoc();
  cache.try_emplace(functionName, loc);
  return loc;
}

static Location resolveLocation(Operation *anchor,
                                llvm::StringMap<Location> &cache,
                                StringRef file, unsigned line, unsigned column,
                                StringRef functionName) {
  if (!file.empty() && line != 0)
    return FileLineColLoc::get(anchor->getContext(), file, line, column);
  return getFunctionLocation(anchor, cache, functionName);
}

static Location
resolveLocation(Operation *anchor, llvm::StringMap<Location> &cache,
                const llvm::DiagnosticInfoWithLocationBase &diag) {
  StringRef functionName = diag.getFunction().getName();
  if (!diag.isLocationAvailable())
    return resolveLocation(anchor, cache, "", 0, 0, functionName);
  llvm::DiagnosticLocation loc = diag.getLocation();
  return resolveLocation(anchor, cache, loc.getAbsolutePath(), loc.getLine(),
                         loc.getColumn(), functionName);
}

//===----------------------------------------------------------------------===//
// Remark conversion
//===----------------------------------------------------------------------===//

namespace {
/// The fields of an LLVM remark that are imported, shared by the live and the
/// serialized paths.
struct ImportedRemark {
  remark::RemarkKind kind = remark::RemarkKind::RemarkUnknown;
  StringRef passName;
  StringRef remarkName;
  StringRef functionName;
  std::string message;
  std::optional<uint64_t> hotness;
  SmallVector<std::pair<StringRef, StringRef>> args;
};
} // namespace

static remark::RemarkKind getRemarkKind(llvm::DiagnosticKind kind) {
  switch (kind) {
  case llvm::DK_OptimizationRemark:
  case llvm::DK_MachineOptimizationRemark:
    return remark::RemarkKind::RemarkPassed;
  case llvm::DK_OptimizationRemarkMissed:
  case llvm::DK_MachineOptimizationRemarkMissed:
    return remark::RemarkKind::RemarkMissed;
  case llvm::DK_OptimizationRemarkAnalysis:
  case llvm::DK_OptimizationRemarkAnalysisFPCommute:
  case llvm::DK_OptimizationRemarkAnalysisAliasing:
  case llvm::DK_MachineOptimizationRemarkAnalysis:
    return remark::RemarkKind::RemarkAnalysis;
  case llvm::DK_OptimizationFailure:
    return remark::RemarkKind::RemarkFailure;
  default:
    return remark::RemarkKind::RemarkUnknown;
  }
}

static remark::RemarkKind getRemarkKind(llvm::remarks::Type type) {
  switch (type) {
  case llvm::remarks::Type::Passed:
    return remark::RemarkKind::RemarkPassed;
  case llvm::remarks::Type::Missed:
    return remark::RemarkKind::RemarkMissed;
  case llvm::remarks::Type::Analysis:
  case llvm::remarks::Type::AnalysisFPCommute:
  case llvm::remarks::Type::AnalysisAliasing:
    return remark::RemarkKind::RemarkAnalysis;
  case llvm::remarks::Type::Failure:
    return remark::RemarkKind::RemarkFailure;
  case llvm::remarks::Type::Unknown:
    return remark::RemarkKind::RemarkUnknown;
  }
  llvm_unreachable("unknown remark type");
}

/// Returns the MLIR remark category of the LLVM pass `passName`.
static std::string getCategory(StringRef passName) {
  return (llvm::Twine(remark::llvmRemarkCategoryPrefix) + passName).str();
}

/// Renames the argument keys that the remark engine uses itself.
static std::string getArgKey(StringRef key) {
  if (key == "Remark" || key == "RemarkId" || key == "RelatedTo")
    return ("LLVM" + key).str();
  return key.str();
}

static void emitImportedRemark(remark::detail::RemarkEngine &engine,
                               Location loc, const ImportedRemark &imported) {
  std::string category = getCategory(imported.passName);
  remark::RemarkOpts opts = remark::RemarkOpts::name(imported.remarkName)
                                .category(category)
                                .function(imported.functionName);
  remark::detail::InFlightRemark inFlight;
  switch (imported.kind) {
  case remark::RemarkKind::RemarkPassed:
    inFlight = engine.emitOptimizationRemark(loc, opts);
    break;
  case remark::RemarkKind::RemarkMissed:
    inFlight = engine.emitOptimizationRemarkMiss(loc, opts);
    break;
  case remark::RemarkKind::RemarkFailure:
    inFlight = engine.emitOptimizationRemarkFailure(loc, opts);
    break;
  case remark::RemarkKind::RemarkAnalysis:
    inFlight = engine.emitOptimizationRemarkAnalysis(loc, opts);
    break;
  case remark::RemarkKind::RemarkUnknown:
    return;
  }
  if (!inFlight)
    return;

  inFlight << StringRef(imported.message);
  for (const auto &[key, value] : imported.args) {
    // Plain strings are already part of the message.
    if (key == "String")
      continue;
    inFlight << remark::detail::Remark::Arg(getArgKey(key), value);
  }
  if (imported.hotness)
    inFlight << remark::detail::Remark::Arg("Hotness", *imported.hotness);
}

static void importLLVMRemark(remark::detail::RemarkEngine &engine, Location loc,
                             const llvm::DiagnosticInfoOptimizationBase &diag) {
  ImportedRemark imported;
  imported.kind =
      getRemarkKind(static_cast<llvm::DiagnosticKind>(diag.getKind()));
  imported.passName = diag.getPassName();
  imported.remarkName = diag.getRemarkName();
  imported.functionName =
      llvm::GlobalValue::dropLLVMManglingEscape(diag.getFunction().getName());
  imported.message = diag.getMsg();
  imported.hotness = diag.getHotness();
  for (const llvm::DiagnosticInfoOptimizationBase::Argument &arg :
       diag.getArgs())
    imported.args.emplace_back(arg.Key, arg.Val);
  emitImportedRemark(engine, loc, imported);
}

static void importLLVMRemark(remark::detail::RemarkEngine &engine, Location loc,
                             const llvm::remarks::Remark &remark) {
  ImportedRemark imported;
  imported.kind = getRemarkKind(remark.RemarkType);
  imported.passName = remark.PassName;
  imported.remarkName = remark.RemarkName;
  imported.functionName = remark.FunctionName;
  imported.message = remark.getArgsAsMsg();
  imported.hotness = remark.Hotness;
  for (const llvm::remarks::Argument &arg : remark.Args)
    imported.args.emplace_back(arg.Key, arg.Val);
  emitImportedRemark(engine, loc, imported);
}

LogicalResult mlir::remark::importLLVMRemarks(Operation *anchor,
                                              StringRef buffer,
                                              llvm::remarks::Format format) {
  remark::detail::RemarkEngine *engine =
      anchor->getContext()->getRemarkEngine();
  if (!engine)
    return success();

  llvm::Expected<std::unique_ptr<llvm::remarks::RemarkParser>> parser =
      llvm::remarks::createRemarkParser(format, buffer);
  if (!parser) {
    llvm::consumeError(parser.takeError());
    return failure();
  }

  llvm::StringMap<Location> functionLocations;
  while (true) {
    llvm::Expected<std::unique_ptr<llvm::remarks::Remark>> next =
        (*parser)->next();
    if (!next) {
      llvm::Error error = next.takeError();
      bool endOfFile = error.isA<llvm::remarks::EndOfFileError>();
      llvm::consumeError(std::move(error));
      return success(endOfFile);
    }
    const llvm::remarks::Remark &remark = **next;
    Location loc = remark.Loc ? resolveLocation(anchor, functionLocations,
                                                remark.Loc->SourceFilePath,
                                                remark.Loc->SourceLine,
                                                remark.Loc->SourceColumn,
                                                remark.FunctionName)
                              : resolveLocation(anchor, functionLocations, "",
                                                0, 0, remark.FunctionName);
    importLLVMRemark(*engine, loc, remark);
  }
}

//===----------------------------------------------------------------------===//
// LLVMToMLIRDiagnosticHandler
//===----------------------------------------------------------------------===//

remark::LLVMToMLIRDiagnosticHandler::LLVMToMLIRDiagnosticHandler(
    Operation *anchor)
    : anchor(anchor), engine(anchor->getContext()->getRemarkEngine()) {}

bool remark::LLVMToMLIRDiagnosticHandler::isAnalysisRemarkEnabled(
    StringRef passName) const {
  return (engine &&
          engine->isAnalysisOptRemarkEnabled(getCategory(passName))) ||
         DiagnosticHandler::isAnalysisRemarkEnabled(passName);
}

bool remark::LLVMToMLIRDiagnosticHandler::isMissedOptRemarkEnabled(
    StringRef passName) const {
  return (engine && engine->isMissedOptRemarkEnabled(getCategory(passName))) ||
         DiagnosticHandler::isMissedOptRemarkEnabled(passName);
}

bool remark::LLVMToMLIRDiagnosticHandler::isPassedOptRemarkEnabled(
    StringRef passName) const {
  return (engine && engine->isPassedOptRemarkEnabled(getCategory(passName))) ||
         DiagnosticHandler::isPassedOptRemarkEnabled(passName);
}

bool remark::LLVMToMLIRDiagnosticHandler::isAnyRemarkEnabled() const {
  return (engine && engine->isAnyRemarkEnabled()) ||
         DiagnosticHandler::isAnyRemarkEnabled();
}

bool remark::LLVMToMLIRDiagnosticHandler::handleDiagnostics(
    const llvm::DiagnosticInfo &diag) {
  if (const auto *optRemark =
          dyn_cast<llvm::DiagnosticInfoOptimizationBase>(&diag)) {
    remark::RemarkKind kind =
        getRemarkKind(static_cast<llvm::DiagnosticKind>(optRemark->getKind()));
    if (!engine ||
        !engine->isRemarkEnabled(kind, getCategory(optRemark->getPassName())))
      return false;
    importLLVMRemark(*engine,
                     resolveLocation(anchor, functionLocations, *optRemark),
                     *optRemark);
    return true;
  }

  std::string message;
  llvm::raw_string_ostream os(message);
  llvm::DiagnosticPrinterRawOStream printer(os);
  diag.print(printer);
  StringRef text = StringRef(message).rtrim();

  Location loc = anchor->getLoc();
  if (const auto *withLoc = dyn_cast<llvm::DiagnosticInfoUnsupported>(&diag)) {
    loc = resolveLocation(anchor, functionLocations, *withLoc);
    // The location is carried by `loc` already.
    text.consume_front(withLoc->getLocationStr() + ": ");
  }

  switch (diag.getSeverity()) {
  case llvm::DS_Error:
    emitError(loc) << text;
    break;
  case llvm::DS_Warning:
    emitWarning(loc) << text;
    break;
  case llvm::DS_Remark:
  case llvm::DS_Note:
    emitRemark(loc) << text;
    break;
  }
  return true;
}
