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

using namespace mlir::remark;

//------------------------------------------------------------------------------
// Remark
//------------------------------------------------------------------------------

Remark::RemarkKeyValue::RemarkKeyValue(StringRef key, Value value)
    : key(std::string(key)) {

  llvm::raw_string_ostream rss(val);
  rss << value;
}

Remark::RemarkKeyValue::RemarkKeyValue(StringRef key, Type type)
    : key(std::string(key)) {
  llvm::raw_string_ostream os(val);
  os << type;
}

Remark::RemarkKeyValue::RemarkKeyValue(StringRef key, StringRef s)
    : key(std::string(key)), val(s.str()) {}

Remark::RemarkKeyValue::RemarkKeyValue(StringRef key, int n)
    : key(std::string(key)), val(llvm::itostr(n)) {}

Remark::RemarkKeyValue::RemarkKeyValue(StringRef key, float n)
    : key(std::string(key)), val(std::to_string(n)) {}

Remark::RemarkKeyValue::RemarkKeyValue(StringRef key, long n)
    : key(std::string(key)), val(llvm::itostr(n)) {}

Remark::RemarkKeyValue::RemarkKeyValue(StringRef key, long long n)
    : key(std::string(key)), val(llvm::itostr(n)) {}

Remark::RemarkKeyValue::RemarkKeyValue(StringRef key, unsigned n)
    : key(std::string(key)), val(llvm::utostr(n)) {}

Remark::RemarkKeyValue::RemarkKeyValue(StringRef key, unsigned long n)
    : key(std::string(key)), val(llvm::utostr(n)) {}

Remark::RemarkKeyValue::RemarkKeyValue(StringRef key, unsigned long long n)
    : key(std::string(key)), val(llvm::utostr(n)) {}

void Remark::insert(StringRef s) { args.emplace_back(s); }

void Remark::insert(RemarkKeyValue a) { args.push_back(std::move(a)); }

// Simple helper to print key=val list.
static void printArgs(llvm::raw_ostream &os,
                      llvm::ArrayRef<Remark::RemarkKeyValue> args) {
  if (args.empty())
    return;
  os << " {";
  for (size_t i = 0; i < args.size(); ++i) {
    const auto &a = args[i];
    os << a.key << "=" << a.val;
    if (i + 1 < args.size())
      os << ", ";
  }
  os << "}";
}

/// Print the remark to the given output stream.
/// Example output:
/// [Missed] LoopUnroll:UnrolledLoop func=myFunction @file.cpp:42:7
/// {tripCount=128, reason=too_small}
void Remark::print(llvm::raw_ostream &os, bool printLocation) const {
  os << '[' << getRemarkTypeString() << "] ";
  os << getPassName() << ':' << getRemarkName();
  if (functionName)
    os << " func=" << getFunction() << " ";

  if (printLocation)
    if (auto flc = mlir::dyn_cast<mlir::FileLineColLoc>(getLocation()))
      os << " @" << flc.getFilename() << ":" << flc.getLine() << ":"
         << flc.getColumn();

  // Key/Value args
  printArgs(os, getArgs());
}

std::string Remark::getMsg() const {
  std::string s;
  llvm::raw_string_ostream os(s);
  print(os);
  os.flush();
  return s;
}

std::string Remark::getRemarkTypeString() const {
  switch (remarkKind) {
  case RemarkKind::OptimizationRemarkUnknown:
    return "Unknown";
  case RemarkKind::OptimizationRemarkPassed:
    return "Passed";
  case RemarkKind::OptimizationRemarkMissed:
    return "Missed";
  case RemarkKind::OptimizationRemarkFailure:
    return "Failure";
  case RemarkKind::OptimizationRemarkAnalysis:
    return "Analysis";
  }
  llvm_unreachable("Unknown remark kind");
}

llvm::remarks::Type Remark::getRemarkType() const {
  switch (remarkKind) {
  case RemarkKind::OptimizationRemarkUnknown:
    return llvm::remarks::Type::Unknown;
  case RemarkKind::OptimizationRemarkPassed:
    return llvm::remarks::Type::Passed;
  case RemarkKind::OptimizationRemarkMissed:
    return llvm::remarks::Type::Missed;
  case RemarkKind::OptimizationRemarkFailure:
    return llvm::remarks::Type::Failure;
  case RemarkKind::OptimizationRemarkAnalysis:
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
  r.PassName = getPassName();
  r.RemarkName = getRemarkName();
  r.FunctionName = getFunction();
  r.Loc = locLambda();
  for (const Remark::RemarkKeyValue &arg : getArgs()) {
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
RemarkEngine::emitIfEnabled(Location loc, StringRef passName,
                            StringRef categoryName,
                            bool (RemarkEngine::*isEnabled)(StringRef) const) {
  return (this->*isEnabled)(categoryName)
             ? makeRemark<RemarkT>(loc, categoryName, passName)
             : InFlightRemark{};
}

bool RemarkEngine::isMissedOptRemarkEnabled(StringRef categoryName) const {
  return missFilter && missFilter->match(categoryName);
}

bool RemarkEngine::isPassedOptRemarkEnabled(StringRef categoryName) const {
  return passFilter && passFilter->match(categoryName);
}

bool RemarkEngine::isAnalysisOptRemarkEnabled(StringRef categoryName) const {
  return analysisFilter && analysisFilter->match(categoryName);
}

bool RemarkEngine::isFailedOptRemarkEnabled(StringRef categoryName) const {
  return failedFilter && failedFilter->match(categoryName);
}

InFlightRemark RemarkEngine::emitOptimizationRemark(Location loc,
                                                    StringRef passName,
                                                    StringRef categoryName) {
  return emitIfEnabled<OptRemarkPass>(loc, passName, categoryName,
                                      &RemarkEngine::isPassedOptRemarkEnabled);
}

InFlightRemark
RemarkEngine::emitOptimizationRemarkMiss(Location loc, StringRef passName,
                                         StringRef categoryName) {
  return emitIfEnabled<OptRemarkMissed>(
      loc, passName, categoryName, &RemarkEngine::isMissedOptRemarkEnabled);
}

InFlightRemark
RemarkEngine::emitOptimizationRemarkFailure(Location loc, StringRef passName,
                                            StringRef categoryName) {
  return emitIfEnabled<OptRemarkFailure>(
      loc, passName, categoryName, &RemarkEngine::isFailedOptRemarkEnabled);
}

InFlightRemark
RemarkEngine::emitOptimizationRemarkAnalysis(Location loc, StringRef passName,
                                             StringRef categoryName) {
  return emitIfEnabled<OptRemarkAnalysis>(
      loc, passName, categoryName, &RemarkEngine::isAnalysisOptRemarkEnabled);
}

//===----------------------------------------------------------------------===//
// Remarkengine
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
                           const MLIRContext::RemarkCategories &cats)
    : printAsEmitRemarks(printAsEmitRemarks) {
  if (cats.passed)
    passFilter = llvm::Regex(cats.passed.value());
  if (cats.missed)
    missFilter = llvm::Regex(cats.missed.value());
  if (cats.analysis)
    analysisFilter = llvm::Regex(cats.analysis.value());
  if (cats.failed)
    failedFilter = llvm::Regex(cats.failed.value());
}
