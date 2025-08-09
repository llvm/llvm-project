//===- Remarks.cpp - MLIR Remarks ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Remarks.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMRemarkStreamer.h"
#include "llvm/Remarks/RemarkFormat.h"
#include "llvm/Support/FileSystem.h"

using namespace mlir;

//------------------------------------------------------------------------------
// RemarkBase
//------------------------------------------------------------------------------

RemarkBase::RemarkKeyValue::RemarkKeyValue(StringRef key, Value value)
    : key(std::string(key)) {

  llvm::raw_string_ostream rss(val);
  rss << value;
}

RemarkBase::RemarkKeyValue::RemarkKeyValue(StringRef key, Type type)
    : key(std::string(key)) {
  llvm::raw_string_ostream os(val);
  os << type;
}

RemarkBase::RemarkKeyValue::RemarkKeyValue(StringRef key, StringRef s)
    : key(std::string(key)), val(s.str()) {}

RemarkBase::RemarkKeyValue::RemarkKeyValue(StringRef key, int n)
    : key(std::string(key)), val(llvm::itostr(n)) {}

RemarkBase::RemarkKeyValue::RemarkKeyValue(StringRef key, float n)
    : key(std::string(key)), val(std::to_string(n)) {}

RemarkBase::RemarkKeyValue::RemarkKeyValue(StringRef key, long n)
    : key(std::string(key)), val(llvm::itostr(n)) {}

RemarkBase::RemarkKeyValue::RemarkKeyValue(StringRef key, long long n)
    : key(std::string(key)), val(llvm::itostr(n)) {}

RemarkBase::RemarkKeyValue::RemarkKeyValue(StringRef key, unsigned n)
    : key(std::string(key)), val(llvm::utostr(n)) {}

RemarkBase::RemarkKeyValue::RemarkKeyValue(StringRef key, unsigned long n)
    : key(std::string(key)), val(llvm::utostr(n)) {}

RemarkBase::RemarkKeyValue::RemarkKeyValue(StringRef key, unsigned long long n)
    : key(std::string(key)), val(llvm::utostr(n)) {}

void RemarkBase::print(llvm::DiagnosticPrinter &dp) const {
  std::string str;
  llvm::raw_string_ostream os(str);
  getLocation()->print(os);
  os.flush();
  dp << str << ": " << getMsg();
}

void RemarkBase::print() const { emitError(getLocation(), getMsg()); }

void RemarkBase::insert(StringRef s) { args.emplace_back(s); }

void RemarkBase::insert(RemarkKeyValue a) { args.push_back(std::move(a)); }

void RemarkBase::insert(SetIsVerbose v) { isVerboseRemark = true; }

std::string RemarkBase::getMsg() const {
  std::string str;
  llvm::raw_string_ostream rss(str);
  for (const RemarkBase::RemarkKeyValue &arg :
       llvm::make_range(args.begin(), args.end()))
    rss << arg.val;
  return rss.str();
}

llvm::remarks::Type RemarkBase::getRemarkType() const {
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

llvm::remarks::Remark RemarkBase::generateRemark() const {
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
  for (const RemarkBase::RemarkKeyValue &arg : getArgs()) {
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
  static_assert(std::is_base_of_v<RemarkBase, RemarkT>,
                "RemarkT must derive from RemarkBase");
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
// RemarkStreamer
//===----------------------------------------------------------------------===//
void MLIRRemarkStreamer::streamOptimizationRemark(const RemarkBase &remark) {
  if (!remarkStreamer.matchesFilter(remark.getPassName()))
    return;

  // First, convert the diagnostic to a remark.
  llvm::remarks::Remark r = remark.generateRemark();
  // Then, emit the remark through the serializer.
  remarkStreamer.getSerializer().emit(r);
}
//===----------------------------------------------------------------------===//
// Remarkengine
//===----------------------------------------------------------------------===//

void RemarkEngine::report(const RemarkBase &&diag) {
  // Stream the remark
  if (getLLVMRemarkStreamer() && remarksFile)
    getLLVMRemarkStreamer()->streamOptimizationRemark(diag);

  // Print using MLIR's diagnostic
  if (printAsEmitRemarks)
    emitRemark(diag.getLocation(), diag.getMsg());
}

RemarkEngine::~RemarkEngine() {
  if (remarksFile) {
    remarksFile->keep();
    remarksFile.reset();
  }
  setMainRemarkStreamer(nullptr);
  setRemarkStreamer(nullptr);
}

LogicalResult RemarkEngine::initialize(StringRef outputPath,
                                       llvm::remarks::Format fmt,
                                       std::string *errMsg) {
  auto fail = [&](llvm::StringRef msg) {
    if (errMsg)
      *errMsg = msg.str();
    return failure();
  };

  if (remarksFile)
    return fail("RemarkEngine is already initialized with an output file.");

  llvm::sys::fs::OpenFlags flags = (fmt == llvm::remarks::Format::YAML)
                                       ? llvm::sys::fs::OF_Text
                                       : llvm::sys::fs::OF_None;
  std::error_code ec;
  remarksFile = std::make_unique<llvm::ToolOutputFile>(outputPath, ec, flags);

  if (ec) {
    remarksFile.reset();
    return fail(
        ("Failed to open remarks file '" + outputPath + "': " + ec.message())
            .str());
  }

  auto serializer = llvm::remarks::createRemarkSerializer(
      fmt, llvm::remarks::SerializerMode::Separate, remarksFile->os());
  if (!serializer) {
    remarksFile.reset();
    return fail(llvm::toString(serializer.takeError()));
  }

  setMainRemarkStreamer(std::make_unique<llvm::remarks::RemarkStreamer>(
      std::move(*serializer), outputPath));
  setRemarkStreamer(
      std::make_unique<MLIRRemarkStreamer>(*getMainRemarkStreamer()));
  return success();
}

RemarkEngine::RemarkEngine(bool printAsEmitRemarks,
                           std::optional<std::string> categoryPassName,
                           std::optional<std::string> categoryMissName,
                           std::optional<std::string> categoryAnalysisName,
                           std::optional<std::string> categoryFailedName)
    : printAsEmitRemarks(printAsEmitRemarks) {
  if (categoryPassName)
    passFilter = llvm::Regex(categoryPassName.value());
  if (categoryMissName)
    missFilter = llvm::Regex(categoryMissName.value());
  if (categoryAnalysisName)
    analysisFilter = llvm::Regex(categoryAnalysisName.value());
  if (categoryFailedName)
    failedFilter = llvm::Regex(categoryFailedName.value());
}
