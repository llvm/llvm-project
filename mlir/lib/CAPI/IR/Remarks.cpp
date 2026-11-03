//===- Remarks.cpp - C Interface for the MLIR Remark Engine ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Remarks.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Remarks.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Remarks.h"
#include "mlir/Remark/RemarkStreamer.h"
#include "llvm/Remarks/RemarkFormat.h"
#include "llvm/Support/ErrorHandling.h"

#include <cassert>
#include <memory>

using namespace mlir;
using mlir::remark::detail::Remark;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

namespace {
/// A streamer forwarding every reported remark to a C callback.
class CallbackRemarkStreamer : public remark::detail::MLIRRemarkStreamerBase {
public:
  CallbackRemarkStreamer(MlirRemarkCallback callback, void *userData,
                         void (*deleteUserData)(void *))
      : callback(callback), userData(userData), deleteUserData(deleteUserData) {
  }

  ~CallbackRemarkStreamer() override {
    if (deleteUserData)
      deleteUserData(userData);
  }

  void streamOptimizationRemark(const Remark &remark) override {
    callback(wrap(&remark), userData);
  }

private:
  MlirRemarkCallback callback;
  void *userData;
  void (*deleteUserData)(void *);
};
} // namespace

/// Every filter reaches the engine as a (possibly empty) string, as with
/// mlir-opt's options: a kind is reported when its own filter or `all` matches.
static remark::RemarkCategories unwrap(MlirRemarkCategories categories) {
  return remark::RemarkCategories{
      unwrap(categories.all).str(), unwrap(categories.passed).str(),
      unwrap(categories.missed).str(), unwrap(categories.analysis).str(),
      unwrap(categories.failed).str()};
}

static std::unique_ptr<remark::detail::RemarkEmittingPolicyBase>
createPolicy(MlirRemarkPolicy policy) {
  switch (policy) {
  case MlirRemarkPolicyAll:
    return std::make_unique<remark::RemarkEmittingPolicyAll>();
  case MlirRemarkPolicyFinal:
    return std::make_unique<remark::RemarkEmittingPolicyFinal>();
  }
  llvm_unreachable("unknown remark policy");
}

static llvm::remarks::Format unwrap(MlirRemarkFileFormat format) {
  switch (format) {
  case MlirRemarkFileFormatYAML:
    return llvm::remarks::Format::YAML;
  case MlirRemarkFileFormatBitstream:
    return llvm::remarks::Format::Bitstream;
  }
  llvm_unreachable("unknown remark file format");
}

static MlirRemarkKind wrap(remark::RemarkKind kind) {
  switch (kind) {
  case remark::RemarkKind::RemarkUnknown:
    return MlirRemarkKindUnknown;
  case remark::RemarkKind::RemarkPassed:
    return MlirRemarkKindPassed;
  case remark::RemarkKind::RemarkMissed:
    return MlirRemarkKindMissed;
  case remark::RemarkKind::RemarkFailure:
    return MlirRemarkKindFailure;
  case remark::RemarkKind::RemarkAnalysis:
    return MlirRemarkKindAnalysis;
  }
  llvm_unreachable("unknown remark kind");
}

//===----------------------------------------------------------------------===//
// Enabling and finalizing
//===----------------------------------------------------------------------===//

MlirLogicalResult mlirContextEnableOptimizationRemarks(
    MlirContext context, MlirRemarkCategories categories,
    MlirRemarkPolicy policy, bool printAsEmitRemarks) {
  MLIRContext *ctx = unwrap(context);
  if (ctx->getRemarkEngine())
    return mlirLogicalResultFailure();
  return wrap(remark::enableOptimizationRemarks(
      *ctx, /*streamer=*/nullptr, createPolicy(policy), unwrap(categories),
      printAsEmitRemarks));
}

MlirLogicalResult mlirContextEnableOptimizationRemarksToFile(
    MlirContext context, MlirRemarkCategories categories,
    MlirRemarkPolicy policy, MlirRemarkFileFormat format, MlirStringRef path,
    bool printAsEmitRemarks) {
  MLIRContext *ctx = unwrap(context);
  if (ctx->getRemarkEngine())
    return mlirLogicalResultFailure();
  return wrap(remark::enableOptimizationRemarksWithLLVMStreamer(
      *ctx, unwrap(path), unwrap(format), createPolicy(policy),
      unwrap(categories), printAsEmitRemarks));
}

MlirLogicalResult mlirContextEnableOptimizationRemarksWithCallback(
    MlirContext context, MlirRemarkCategories categories,
    MlirRemarkPolicy policy, MlirRemarkCallback callback, void *userData,
    void (*deleteUserData)(void *), bool printAsEmitRemarks) {
  assert(callback && "unexpected null remark callback");
  MLIRContext *ctx = unwrap(context);
  if (ctx->getRemarkEngine()) {
    // The caller handed over `userData`; release it as on success.
    if (deleteUserData)
      deleteUserData(userData);
    return mlirLogicalResultFailure();
  }
  return wrap(remark::enableOptimizationRemarks(
      *ctx,
      std::make_unique<CallbackRemarkStreamer>(callback, userData,
                                               deleteUserData),
      createPolicy(policy), unwrap(categories), printAsEmitRemarks));
}

void mlirContextFinalizeOptimizationRemarks(MlirContext context) {
  // The engine destructor reports the postponed remarks, then destroys the
  // streamer.
  unwrap(context)->setRemarkEngine(nullptr);
}

bool mlirContextHasRemarkEngine(MlirContext context) {
  return unwrap(context)->getRemarkEngine() != nullptr;
}

//===----------------------------------------------------------------------===//
// Remark accessors
//===----------------------------------------------------------------------===//

MlirRemarkKind mlirRemarkGetKind(MlirRemark remark) {
  return wrap(unwrap(remark)->getRemarkKind());
}

MlirStringRef mlirRemarkGetRemarkName(MlirRemark remark) {
  return wrap(unwrap(remark)->getRemarkName());
}

MlirStringRef mlirRemarkGetCategoryName(MlirRemark remark) {
  return wrap(unwrap(remark)->getCategoryName());
}

MlirStringRef mlirRemarkGetFullCategoryName(MlirRemark remark) {
  return wrap(unwrap(remark)->getCombinedCategoryName());
}

MlirStringRef mlirRemarkGetFunctionName(MlirRemark remark) {
  return wrap(unwrap(remark)->getFunction());
}

MlirLocation mlirRemarkGetLocation(MlirRemark remark) {
  return wrap(unwrap(remark)->getLocation());
}

uint64_t mlirRemarkGetId(MlirRemark remark) {
  return unwrap(remark)->getId().getValue();
}

intptr_t mlirRemarkGetNumArgs(MlirRemark remark) {
  return static_cast<intptr_t>(unwrap(remark)->getArgs().size());
}

MlirStringRef mlirRemarkGetArgKey(MlirRemark remark, intptr_t pos) {
  ArrayRef<Remark::Arg> args = unwrap(remark)->getArgs();
  assert(pos >= 0 && static_cast<size_t>(pos) < args.size() &&
         "remark argument index out of range");
  return wrap(llvm::StringRef(args[pos].key));
}

MlirStringRef mlirRemarkGetArgValue(MlirRemark remark, intptr_t pos) {
  ArrayRef<Remark::Arg> args = unwrap(remark)->getArgs();
  assert(pos >= 0 && static_cast<size_t>(pos) < args.size() &&
         "remark argument index out of range");
  return wrap(llvm::StringRef(args[pos].val));
}

void mlirRemarkPrint(MlirRemark remark, MlirStringCallback callback,
                     void *userData) {
  detail::CallbackOstream stream(callback, userData);
  unwrap(remark)->print(stream);
}

//===----------------------------------------------------------------------===//
// Emission
//===----------------------------------------------------------------------===//

bool mlirEmitOptimizationRemark(
    MlirLocation location, MlirRemarkKind kind, MlirStringRef remarkName,
    MlirStringRef categoryName, MlirStringRef subCategoryName,
    MlirStringRef functionName, MlirStringRef message, intptr_t numArgs,
    const MlirStringRef *argKeys, const MlirStringRef *argValues) {
  assert((numArgs == 0 || (argKeys && argValues)) &&
         "remark arguments require both keys and values");
  Location loc = unwrap(location);
  remark::RemarkOpts opts = remark::RemarkOpts::name(unwrap(remarkName))
                                .category(unwrap(categoryName))
                                .subCategory(unwrap(subCategoryName))
                                .function(unwrap(functionName));
  remark::detail::InFlightRemark inFlight;
  switch (kind) {
  case MlirRemarkKindUnknown:
    return false;
  case MlirRemarkKindPassed:
    inFlight = remark::passed(loc, opts);
    break;
  case MlirRemarkKindMissed:
    inFlight = remark::missed(loc, opts);
    break;
  case MlirRemarkKindFailure:
    inFlight = remark::failed(loc, opts);
    break;
  case MlirRemarkKindAnalysis:
    inFlight = remark::analysis(loc, opts);
    break;
  }
  if (!inFlight)
    return false;
  if (message.length != 0)
    inFlight << unwrap(message);
  for (intptr_t i = 0; i < numArgs; ++i)
    inFlight << Remark::Arg(unwrap(argKeys[i]), unwrap(argValues[i]));
  // The remark is reported when `inFlight` goes out of scope.
  return true;
}
