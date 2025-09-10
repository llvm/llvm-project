//===- Remarks.cpp - MLIR Remarks -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Remarks.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir::remark::detail;

//------------------------------------------------------------------------------
// Remark
//------------------------------------------------------------------------------

Remark::Arg::Arg(llvm::StringRef k, Value v) : key(k) {
  llvm::raw_string_ostream os(val);
  os << v;
}

Remark::Arg::Arg(llvm::StringRef k, Type t) : key(k) {
  llvm::raw_string_ostream os(val);
  os << t;
}

void Remark::insert(llvm::StringRef s) { args.emplace_back(s); }
void Remark::insert(Arg a) { args.push_back(std::move(a)); }

// Simple helper to print key=val list (sorted).
static void printArgs(llvm::raw_ostream &os, llvm::ArrayRef<Remark::Arg> args) {
  if (args.empty())
    return;

  llvm::SmallVector<Remark::Arg, 8> sorted(args.begin(), args.end());
  llvm::sort(sorted, [](const Remark::Arg &a, const Remark::Arg &b) {
    return a.key < b.key;
  });

  for (size_t i = 0; i < sorted.size(); ++i) {
    const auto &a = sorted[i];
    os << a.key << "=";

    llvm::StringRef val(a.val);
    bool needsQuote = val.contains(' ') || val.contains(',') ||
                      val.contains('{') || val.contains('}');
    if (needsQuote)
      os << '"' << val << '"';
    else
      os << val;

    if (i + 1 < sorted.size())
      os << ", ";
  }
}

/// Print the remark to the given output stream.
/// Example output:
// clang-format off
/// [Missed] Category: Loop | Pass:Unroller |  Function=main | Reason="tripCount=4 < threshold=256"
/// [Failure] LoopOptimizer | Reason="failed due to unsupported pattern"
// clang-format on
void Remark::print(llvm::raw_ostream &os, bool printLocation) const {
  // Header: [Type] pass:remarkName
  StringRef type = getRemarkTypeString();
  StringRef categoryName = getFullCategoryName();
  StringRef name = remarkName;

  os << '[' << type << "] ";
  os << name << " | ";
  if (!categoryName.empty())
    os << "Category:" << categoryName << " | ";
  if (!functionName.empty())
    os << "Function=" << getFunction() << " | ";

  if (printLocation) {
    if (auto flc = mlir::dyn_cast<mlir::FileLineColLoc>(getLocation()))
      os << " @" << flc.getFilename() << ":" << flc.getLine() << ":"
         << flc.getColumn();
  }

  printArgs(os, getArgs());
}

std::string Remark::getMsg() const {
  std::string s;
  llvm::raw_string_ostream os(s);
  print(os);
  os.flush();
  return s;
}

llvm::StringRef Remark::getRemarkTypeString() const {
  switch (remarkKind) {
  case RemarkKind::RemarkUnknown:
    return "Unknown";
  case RemarkKind::RemarkPassed:
    return "Passed";
  case RemarkKind::RemarkMissed:
    return "Missed";
  case RemarkKind::RemarkFailure:
    return "Failure";
  case RemarkKind::RemarkAnalysis:
    return "Analysis";
  }
  llvm_unreachable("Unknown remark kind");
}

llvm::remarks::Type Remark::getRemarkType() const {
  switch (remarkKind) {
  case RemarkKind::RemarkUnknown:
    return llvm::remarks::Type::Unknown;
  case RemarkKind::RemarkPassed:
    return llvm::remarks::Type::Passed;
  case RemarkKind::RemarkMissed:
    return llvm::remarks::Type::Missed;
  case RemarkKind::RemarkFailure:
    return llvm::remarks::Type::Failure;
  case RemarkKind::RemarkAnalysis:
    return llvm::remarks::Type::Analysis;
  }
  llvm_unreachable("Unknown remark kind");
}

llvm::remarks::Remark Remark::generateRemark() const {
  auto locLambda = [&]() -> llvm::remarks::RemarkLocation {
    if (auto flc = dyn_cast<FileLineColLoc>(getLocation()))
      return {flc.getFilename(), flc.getLine(), flc.getColumn()};
    return {"<unknown file>", 0, 0};
  };

  llvm::remarks::Remark r; // The result.
  r.RemarkType = getRemarkType();
  r.RemarkName = getRemarkName();
  // MLIR does not use passes; instead, it has categories and sub-categories.
  r.PassName = getFullCategoryName();
  r.FunctionName = getFunction();
  r.Loc = locLambda();
  for (const Remark::Arg &arg : getArgs()) {
    r.Args.emplace_back();
    r.Args.back().Key = arg.key;
    r.Args.back().Val = arg.val;
  }
  return r;
}

//===----------------------------------------------------------------------===//
// InFlightRemark
//===----------------------------------------------------------------------===//

InFlightRemark::~InFlightRemark() {
  if (remark && owner)
    owner->report(std::move(*remark));
  owner = nullptr;
}

//===----------------------------------------------------------------------===//
// Remark Engine
//===----------------------------------------------------------------------===//

template <typename RemarkT, typename... Args>
InFlightRemark RemarkEngine::makeRemark(Args &&...args) {
  static_assert(std::is_base_of_v<Remark, RemarkT>,
                "RemarkT must derive from Remark");
  return InFlightRemark(*this,
                        std::make_unique<RemarkT>(std::forward<Args>(args)...));
}

template <typename RemarkT>
InFlightRemark
RemarkEngine::emitIfEnabled(Location loc, RemarkOpts opts,
                            bool (RemarkEngine::*isEnabled)(StringRef) const) {
  return (this->*isEnabled)(opts.categoryName) ? makeRemark<RemarkT>(loc, opts)
                                               : InFlightRemark{};
}

bool RemarkEngine::isMissedOptRemarkEnabled(StringRef categoryName) const {
  return missFilter && missFilter->match(categoryName);
}

bool RemarkEngine::isPassedOptRemarkEnabled(StringRef categoryName) const {
  return passedFilter && passedFilter->match(categoryName);
}

bool RemarkEngine::isAnalysisOptRemarkEnabled(StringRef categoryName) const {
  return analysisFilter && analysisFilter->match(categoryName);
}

bool RemarkEngine::isFailedOptRemarkEnabled(StringRef categoryName) const {
  return failedFilter && failedFilter->match(categoryName);
}

InFlightRemark RemarkEngine::emitOptimizationRemark(Location loc,
                                                    RemarkOpts opts) {
  return emitIfEnabled<OptRemarkPass>(loc, opts,
                                      &RemarkEngine::isPassedOptRemarkEnabled);
}

InFlightRemark RemarkEngine::emitOptimizationRemarkMiss(Location loc,
                                                        RemarkOpts opts) {
  return emitIfEnabled<OptRemarkMissed>(
      loc, opts, &RemarkEngine::isMissedOptRemarkEnabled);
}

InFlightRemark RemarkEngine::emitOptimizationRemarkFailure(Location loc,
                                                           RemarkOpts opts) {
  return emitIfEnabled<OptRemarkFailure>(
      loc, opts, &RemarkEngine::isFailedOptRemarkEnabled);
}

InFlightRemark RemarkEngine::emitOptimizationRemarkAnalysis(Location loc,
                                                            RemarkOpts opts) {
  return emitIfEnabled<OptRemarkAnalysis>(
      loc, opts, &RemarkEngine::isAnalysisOptRemarkEnabled);
}

//===----------------------------------------------------------------------===//
// RemarkEngine
//===----------------------------------------------------------------------===//

void RemarkEngine::report(const Remark &&remark) {
  // Stream the remark
  if (remarkStreamer)
    remarkStreamer->streamOptimizationRemark(remark);

  // Print using MLIR's diagnostic
  if (printAsEmitRemarks)
    emitRemark(remark.getLocation(), remark.getMsg());
}

RemarkEngine::~RemarkEngine() {
  if (remarkStreamer)
    remarkStreamer->finalize();
}

llvm::LogicalResult
RemarkEngine::initialize(std::unique_ptr<MLIRRemarkStreamerBase> streamer,
                         std::string *errMsg) {
  // If you need to validate categories/filters, do so here and set errMsg.
  remarkStreamer = std::move(streamer);
  return success();
}

RemarkEngine::RemarkEngine(bool printAsEmitRemarks,
                           const RemarkCategories &cats)
    : printAsEmitRemarks(printAsEmitRemarks) {
  if (cats.passed)
    passedFilter = llvm::Regex(cats.passed.value());
  if (cats.missed)
    missFilter = llvm::Regex(cats.missed.value());
  if (cats.analysis)
    analysisFilter = llvm::Regex(cats.analysis.value());
  if (cats.failed)
    failedFilter = llvm::Regex(cats.failed.value());
}

llvm::LogicalResult mlir::remark::enableOptimizationRemarks(
    MLIRContext &ctx,
    std::unique_ptr<remark::detail::MLIRRemarkStreamerBase> streamer,
    const remark::RemarkCategories &cats, bool printAsEmitRemarks) {
  auto engine =
      std::make_unique<remark::detail::RemarkEngine>(printAsEmitRemarks, cats);

  std::string errMsg;
  if (failed(engine->initialize(std::move(streamer), &errMsg))) {
    llvm::report_fatal_error(
        llvm::Twine("Failed to initialize remark engine. Error: ") + errMsg);
  }
  ctx.setRemarkEngine(std::move(engine));

  return success();
}
