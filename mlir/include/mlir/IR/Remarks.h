//===- Remarks.h - MLIR Optimization Remark ----------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines utilities for emitting optimization remarks.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_REMARKS_H
#define MLIR_IR_REMARKS_H

#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/Remarks/Remark.h"
#include "llvm/Remarks/RemarkStreamer.h"
#include "llvm/Support/ToolOutputFile.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir {

/// Defines different remark kinds that can be used to categorize remarks.
enum class RemarkKind {
  OptimizationRemarkUnknown = 0,
  OptimizationRemarkPassed,
  OptimizationRemarkMissed,
  OptimizationRemarkFailure,
  OptimizationRemarkAnalysis,
};

//===----------------------------------------------------------------------===//
// Remark Base Class
//===----------------------------------------------------------------------===//
class RemarkBase : public llvm::DiagnosticInfo {

public:
  RemarkBase(RemarkKind remarkKind, DiagnosticSeverity severity,
             const char *passName, StringRef remarkName, Location loc,
             std::optional<StringRef> functionName = std::nullopt)
      : llvm::DiagnosticInfo(makeLLVMKind(remarkKind),
                             makeLLVMSeverity(severity)),
        remarkKind(remarkKind), functionName(functionName), loc(loc),
        passName(passName), remarkName(remarkName) {}

  struct SetIsVerbose {};

  struct SetExtraArgs {};

  struct RemarkKeyValue {
    std::string key;
    std::string val;

    explicit RemarkKeyValue(StringRef str = "") : key("String"), val(str) {}
    RemarkKeyValue(StringRef key, Value value);
    RemarkKeyValue(StringRef key, Type type);
    RemarkKeyValue(StringRef key, StringRef s);
    RemarkKeyValue(StringRef key, const char *s)
        : RemarkKeyValue(key, StringRef(s)) {};
    RemarkKeyValue(StringRef key, int n);
    RemarkKeyValue(StringRef key, float n);
    RemarkKeyValue(StringRef key, long n);
    RemarkKeyValue(StringRef key, long long n);
    RemarkKeyValue(StringRef key, unsigned n);
    RemarkKeyValue(StringRef key, unsigned long n);
    RemarkKeyValue(StringRef key, unsigned long long n);
    RemarkKeyValue(StringRef key, bool b)
        : key(key), val(b ? "true" : "false") {}
  };

  void insert(StringRef s);
  void insert(RemarkKeyValue a);
  void insert(SetIsVerbose v);
  void insert(SetExtraArgs ea);

  void print(llvm::DiagnosticPrinter &dp) const override;
  void print() const;

  virtual bool isEnabled() const = 0;
  Location getLocation() const { return loc; }
  /// Diagnostic -> Remark
  llvm::remarks::Remark generateRemark() const;

  StringRef getFunction() const {
    if (functionName)
      return *functionName;
    return "<unknown function>";
  }
  StringRef getPassName() const { return passName; }
  StringRef getRemarkName() const { return remarkName; }
  std::string getMsg() const;

  bool isVerbose() const { return isVerboseRemark; }

  ArrayRef<RemarkKeyValue> getArgs() const { return args; }

  llvm::remarks::Type getRemarkType() const;

protected:
  /// Keeps the MLIR diagnostic kind, which is used to determine the
  /// diagnostic kind in the LLVM remark streamer.
  RemarkKind remarkKind;
  /// Name of the convering function like interface
  std::optional<std::string> functionName;

  Location loc;
  /// Name of the pass that triggers this report.
  const char *passName;

  /// Textual identifier for the remark (single-word, CamelCase). Can be used
  /// by external tools reading the output file for optimization remarks to
  /// identify the remark.
  StringRef remarkName;

  /// RemarkKeyValues collected via the streaming interface.
  SmallVector<RemarkKeyValue, 4> args;

  /// The remark is expected to be noisy.
  bool isVerboseRemark = false;

private:
  /// Convert the MLIR diagnostic severity to LLVM diagnostic severity.
  static llvm::DiagnosticSeverity
  makeLLVMSeverity(DiagnosticSeverity severity) {
    switch (severity) {
    case DiagnosticSeverity::Note:
      return llvm::DiagnosticSeverity::DS_Note;
    case DiagnosticSeverity::Warning:
      return llvm::DiagnosticSeverity::DS_Warning;
    case DiagnosticSeverity::Error:
      return llvm::DiagnosticSeverity::DS_Error;
    case DiagnosticSeverity::Remark:
      return llvm::DiagnosticSeverity::DS_Remark;
    }
    llvm_unreachable("Unknown diagnostic severity");
  }
  /// Convert the MLIR remark kind to LLVM diagnostic kind.
  static llvm::DiagnosticKind makeLLVMKind(RemarkKind remarkKind) {
    switch (remarkKind) {
    case RemarkKind::OptimizationRemarkUnknown:
      return llvm::DiagnosticKind::DK_Generic;
    case RemarkKind::OptimizationRemarkPassed:
      return llvm::DiagnosticKind::DK_OptimizationRemark;
    case RemarkKind::OptimizationRemarkMissed:
      return llvm::DiagnosticKind::DK_OptimizationRemarkMissed;
    case RemarkKind::OptimizationRemarkFailure:
      return llvm::DiagnosticKind::DK_OptimizationFailure;
    case RemarkKind::OptimizationRemarkAnalysis:
      return llvm::DiagnosticKind::DK_OptimizationRemarkAnalysis;
    }
    llvm_unreachable("Unknown diagnostic kind");
  }
};

inline RemarkBase &operator<<(RemarkBase &r, StringRef s) {
  r.insert(s);
  return r;
}
inline RemarkBase &&operator<<(RemarkBase &&r, StringRef s) {
  r.insert(s);
  return std::move(r);
}
inline RemarkBase &operator<<(RemarkBase &r,
                              const RemarkBase::RemarkKeyValue &kv) {
  r.insert(kv);
  return r;
}

//===----------------------------------------------------------------------===//
// Shorthand aliases for different kinds of remarks.
//===----------------------------------------------------------------------===//

template <RemarkKind K, DiagnosticSeverity S>
class OptRemarkBase final : public RemarkBase {
public:
  explicit OptRemarkBase(Location loc, StringRef passName,
                         StringRef categoryName)
      : RemarkBase(K, S, passName.data(), categoryName, loc) {}

  bool isEnabled() const override { return true; }
};

using OptRemarkAnalysis = OptRemarkBase<RemarkKind::OptimizationRemarkAnalysis,
                                        DiagnosticSeverity::Remark>;

using OptRemarkPass = OptRemarkBase<RemarkKind::OptimizationRemarkPassed,
                                    DiagnosticSeverity::Remark>;

using OptRemarkMissed = OptRemarkBase<RemarkKind::OptimizationRemarkMissed,
                                      DiagnosticSeverity::Remark>;

using OptRemarkFailure = OptRemarkBase<RemarkKind::OptimizationRemarkFailure,
                                       DiagnosticSeverity::Remark>;

class RemarkEngine;

//===----------------------------------------------------------------------===//
// InFlightRemark
//===----------------------------------------------------------------------===//

/// InFlightRemark is a RAII class that holds a reference to a RemarkBase
/// instance and allows to build the remark using the << operator. The remark
/// is emitted when the InFlightRemark instance is destroyed, which happens
/// when the scope ends or when the InFlightRemark instance is moved.
/// Similar to InFlightDiagnostic, but for remarks.
class InFlightRemark {
public:
  explicit InFlightRemark(RemarkBase *diag) : remark(diag) {}
  InFlightRemark(RemarkEngine &eng, std::unique_ptr<RemarkBase> diag)
      : owner(&eng), remark(std::move(diag)) {}

  InFlightRemark() = default; // empty ctor

  template <typename T>
  InFlightRemark &operator<<(T &&arg) {
    if (remark)
      *remark << std::forward<T>(arg);
    return *this;
  }

  explicit operator bool() const { return remark != nullptr; }

  ~InFlightRemark();

  InFlightRemark(const InFlightRemark &) = delete;
  InFlightRemark &operator=(const InFlightRemark &) = delete;
  InFlightRemark(InFlightRemark &&) = default;
  InFlightRemark &operator=(InFlightRemark &&) = default;

private:
  RemarkEngine *owner{nullptr};
  std::unique_ptr<RemarkBase> remark;
};

//===----------------------------------------------------------------------===//
// MLIR Remark Streamer
//===----------------------------------------------------------------------===//

class MLIRRemarkStreamer {
  llvm::remarks::RemarkStreamer &remarkStreamer;

public:
  explicit MLIRRemarkStreamer(llvm::remarks::RemarkStreamer &remarkStreamer)
      : remarkStreamer(remarkStreamer) {}

  void streamOptimizationRemark(const RemarkBase &remark);
};

//===----------------------------------------------------------------------===//
// Remark Engine (MLIR Context will own this class)
//===----------------------------------------------------------------------===//

class RemarkEngine {
private:
  /// Regex that filters missed optimization remarks: only matching one are
  /// reported.
  std::optional<llvm::Regex> missFilter;
  /// The category for passed optimization remarks.
  std::optional<llvm::Regex> passFilter;
  /// The category for analysis remarks.
  std::optional<llvm::Regex> analysisFilter;
  /// The category for failed optimization remarks.
  std::optional<llvm::Regex> failedFilter;
  /// The output file for the remarks.
  std::unique_ptr<llvm::ToolOutputFile> remarksFile;
  /// The MLIR remark streamer that will be used to emit the remarks.
  std::unique_ptr<MLIRRemarkStreamer> remarkStreamer;
  /// The LLVM remark streamer that will be used to emit the remarks.
  std::unique_ptr<llvm::remarks::RemarkStreamer> llvmRemarkStreamer;
  /// When is enabled, engine also prints remarks as mlir::emitRemarks.
  bool printAsEmitRemarks = false;

  /// The main MLIR remark streamer that will be used to emit the remarks.
  MLIRRemarkStreamer *getLLVMRemarkStreamer() { return remarkStreamer.get(); }
  const MLIRRemarkStreamer *getLLVMRemarkStreamer() const {
    return remarkStreamer.get();
  }
  void setRemarkStreamer(std::unique_ptr<MLIRRemarkStreamer> remarkStreamer) {
    this->remarkStreamer = std::move(remarkStreamer);
  }

  /// Get the main MLIR remark streamer that will be used to emit the remarks.
  llvm::remarks::RemarkStreamer *getMainRemarkStreamer() {
    return llvmRemarkStreamer.get();
  }
  const llvm::remarks::RemarkStreamer *getMainRemarkStreamer() const {
    return llvmRemarkStreamer.get();
  }
  /// Set the main remark streamer to be used by the engine.
  void setMainRemarkStreamer(
      std::unique_ptr<llvm::remarks::RemarkStreamer> mainRemarkStreamer) {
    llvmRemarkStreamer = std::move(mainRemarkStreamer);
  }

  /// Return true if missed optimization remarks are enabled, override
  /// to provide different implementation.
  bool isMissedOptRemarkEnabled(StringRef categoryName) const;

  /// Return true if passed optimization remarks are enabled, override
  /// to provide different implementation.
  bool isPassedOptRemarkEnabled(StringRef categoryName) const;

  /// Return true if analysis optimization remarks are enabled, override
  /// to provide different implementation.
  bool isAnalysisOptRemarkEnabled(StringRef categoryName) const;

  /// Return true if analysis optimization remarks are enabled, override
  /// to provide different implementation.
  bool isFailedOptRemarkEnabled(StringRef categoryName) const;

  /// Return true if any type of remarks are enabled for this pass.
  bool isAnyRemarkEnabled(StringRef categoryName) const {
    return (isMissedOptRemarkEnabled(categoryName) ||
            isPassedOptRemarkEnabled(categoryName) ||
            isFailedOptRemarkEnabled(categoryName) ||
            isAnalysisOptRemarkEnabled(categoryName));
  }

  /// Emit a remark using the given maker function, which should return
  /// a RemarkBase instance. The remark will be emitted using the main
  /// remark streamer.
  template <typename RemarkT, typename... Args>
  InFlightRemark makeRemark(Args &&...args);

  template <typename RemarkT>
  InFlightRemark
  emitIfEnabled(Location loc, StringRef passName, StringRef category,
                bool (RemarkEngine::*isEnabled)(StringRef) const);

public:
  /// Default constructor is deleted, use the other constructor.
  RemarkEngine() = delete;

  /// Constructs Remark engine with optional category names. If a category
  /// name is not provided, it is not enabled. The category names are used to
  /// filter the remarks that are emitted.
  RemarkEngine(bool printAsEmitRemarks,
               std::optional<std::string> categoryPassName = std::nullopt,
               std::optional<std::string> categoryMissName = std::nullopt,
               std::optional<std::string> categoryAnalysisName = std::nullopt,
               std::optional<std::string> categoryFailedName = std::nullopt);

  /// Destructor that will close the output file and reset the
  /// main remark streamer.
  ~RemarkEngine();

  /// Setup the remark engine with the given output path and format.
  LogicalResult initialize(StringRef outputPath, llvm::remarks::Format fmt,
                           std::string *errMsg);

  /// Report a diagnostic remark.
  void report(const RemarkBase &&diag);

  /// Report a successful remark, this will create an InFlightRemark
  /// that can be used to build the remark using the << operator.
  InFlightRemark emitOptimizationRemark(Location loc, StringRef passName,
                                        StringRef category);
  /// Report a missed optimization remark
  /// that can be used to build the remark using the << operator.
  InFlightRemark emitOptimizationRemarkMiss(Location loc, StringRef passName,
                                            StringRef category);
  /// Report a failed optimization remark, this will create an InFlightRemark
  /// that can be used to build the remark using the << operator.
  InFlightRemark emitOptimizationRemarkFailure(Location loc, StringRef passName,
                                               StringRef category);
  /// Report an analysis remark, this will create an InFlightRemark
  /// that can be used to build the remark using the << operator.
  InFlightRemark emitOptimizationRemarkAnalysis(Location loc,
                                                StringRef passName,
                                                StringRef category);
};

//===----------------------------------------------------------------------===//
// Emitters
//===----------------------------------------------------------------------===//

using Suggestion = RemarkBase::RemarkKeyValue;
inline Suggestion suggest(StringRef txt) { return {"Suggestion", txt}; }

template <typename Fn, typename... Args>
inline InFlightRemark withEngine(Fn fn, Location loc, Args &&...args) {
  MLIRContext *ctx = loc->getContext();

  RemarkEngine *enginePtr = ctx->getRemarkEngine();

  if (enginePtr)
    return (enginePtr->*fn)(loc, std::forward<Args>(args)...);

  return {};
}

/// Report an optimization remark that was passed.
inline InFlightRemark reportOptimizationPass(Location loc, StringRef cat,
                                             StringRef passName) {
  return withEngine(&RemarkEngine::emitOptimizationRemark, loc, passName, cat);
}

/// Report an optimization remark that was missed.
inline InFlightRemark reportOptimizationMiss(Location loc, StringRef cat,
                                             StringRef passName,
                                             StringRef suggestion) {
  auto r =
      withEngine(&RemarkEngine::emitOptimizationRemarkMiss, loc, passName, cat);
  if (r)
    r << suggest(suggestion);
  return r;
}
/// Report an optimization failure remark.
inline InFlightRemark reportOptimizationFail(Location loc, StringRef cat,
                                             StringRef passName) {
  return withEngine(&RemarkEngine::emitOptimizationRemarkFailure, loc, passName,
                    cat);
}

/// Report an optimization analysis remark.
inline InFlightRemark reportOptimizationAnalysis(Location loc, StringRef cat,
                                                 StringRef passName) {
  return withEngine(&RemarkEngine::emitOptimizationRemarkAnalysis, loc,
                    passName, cat);
}

} // namespace mlir

#endif // MLIR_IR_REMARKS_H