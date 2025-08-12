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

#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/Remarks/Remark.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Regex.h"
#include <optional>

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"

namespace mlir::remark {

/// Defines different remark kinds that can be used to categorize remarks.
enum class RemarkKind {
  OptimizationRemarkUnknown = 0,
  OptimizationRemarkPassed,   // Optimization remark that was passed.
  OptimizationRemarkMissed,   // Optimization remark that was missed.
  OptimizationRemarkFailure,  // Optimization failure remark.
  OptimizationRemarkAnalysis, // Analysis remark that does not indicate a pass.
};
} // namespace mlir::remark

namespace mlir::remark::detail {
//===----------------------------------------------------------------------===//
// Remark Base Class
//===----------------------------------------------------------------------===//
class Remark {

public:
  Remark(RemarkKind remarkKind, DiagnosticSeverity severity,
         const char *passName, StringRef remarkName, Location loc,
         std::optional<StringRef> functionName = std::nullopt)
      : remarkKind(remarkKind), functionName(functionName), loc(loc),
        passName(passName), remarkName(remarkName) {}

  // Remark argument that is a key-value pair that can be printed as machine
  // parsable args.
  struct Arg {
    std::string key;
    std::string val;
    Arg(llvm::StringRef m) : key("Remark"), val(m) {}
    Arg(llvm::StringRef k, llvm::StringRef v) : key(k), val(v) {}
    Arg(llvm::StringRef k, std::string v) : key(k), val(std::move(v)) {}
    Arg(llvm::StringRef k, const char *v) : Arg(k, llvm::StringRef(v)) {}
    Arg(llvm::StringRef k, Value v);
    Arg(llvm::StringRef k, Type t);
    Arg(llvm::StringRef k, bool b) : key(k), val(b ? "true" : "false") {}

    // One constructor for all arithmetic types except bool.
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T> &&
                                                      !std::is_same_v<T, bool>>>
    Arg(llvm::StringRef k, T v) : key(k) {
      if constexpr (std::is_floating_point_v<T>) {
        llvm::raw_string_ostream os(val);
        os << v;
      } else if constexpr (std::is_signed_v<T>) {
        val = llvm::itostr(static_cast<long long>(v));
      } else {
        val = llvm::utostr(static_cast<unsigned long long>(v));
      }
    }
  };

  void insert(llvm::StringRef s);
  void insert(Arg a);

  void print(llvm::raw_ostream &os, bool printLocation = false) const;

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

  ArrayRef<Arg> getArgs() const { return args; }

  llvm::remarks::Type getRemarkType() const;

  std::string getRemarkTypeString() const;

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

  /// Args collected via the streaming interface.
  SmallVector<Arg, 4> args;

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

inline Remark &operator<<(Remark &r, StringRef s) {
  r.insert(s);
  return r;
}
inline Remark &&operator<<(Remark &&r, StringRef s) {
  r.insert(s);
  return std::move(r);
}
inline Remark &operator<<(Remark &r, const Remark::Arg &kv) {
  r.insert(kv);
  return r;
}

//===----------------------------------------------------------------------===//
// Shorthand aliases for different kinds of remarks.
//===----------------------------------------------------------------------===//

template <RemarkKind K, DiagnosticSeverity S>
class OptRemarkBase final : public Remark {
public:
  explicit OptRemarkBase(Location loc, StringRef passName,
                         StringRef categoryName)
      : Remark(K, S, passName.data(), categoryName, loc) {}
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

/// Lazy text building for zero cost  string formatting.
struct LazyTextBuild {
  llvm::StringRef key;
  std::function<std::string()> thunk;
};

/// InFlightRemark is a RAII class that holds a reference to a Remark
/// instance and allows to build the remark using the << operator. The remark
/// is emitted when the InFlightRemark instance is destroyed, which happens
/// when the scope ends or when the InFlightRemark instance is moved.
/// Similar to InFlightDiagnostic, but for remarks.
class InFlightRemark {
public:
  explicit InFlightRemark(Remark *diag) : remark(diag) {}
  InFlightRemark(RemarkEngine &eng, std::unique_ptr<Remark> diag)
      : owner(&eng), remark(std::move(diag)) {}

  InFlightRemark() = default; // empty ctor

  InFlightRemark &operator<<(const LazyTextBuild &l) {
    if (remark)
      *remark << Remark::Arg(l.key, l.thunk());
    return *this;
  }

  // Generic path, but *not* for Lazy
  template <typename T, typename = std::enable_if_t<
                            !std::is_same_v<std::decay_t<T>, LazyTextBuild>>>
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
  std::unique_ptr<Remark> remark;
};

//===----------------------------------------------------------------------===//
// MLIR Remark Streamer
//===----------------------------------------------------------------------===//

/// Base class for MLIR remark streamers that is used to stream
/// optimization remarks to the underlying remark streamer. The derived classes
/// should implement the `streamOptimizationRemark` method to provide the
/// actual streaming implementation.
class MLIRRemarkStreamerBase {
public:
  virtual ~MLIRRemarkStreamerBase() = default;
  /// Stream an optimization remark to the underlying remark streamer. It is
  /// called by the RemarkEngine to stream the optimization remarks.
  ///
  /// It must be overridden by the derived classes to provide
  /// the actual streaming implementation.
  virtual void streamOptimizationRemark(const Remark &remark) = 0;

  virtual void finalize() {} // optional
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
  /// The MLIR remark streamer that will be used to emit the remarks.
  std::unique_ptr<MLIRRemarkStreamerBase> remarkStreamer;
  /// When is enabled, engine also prints remarks as mlir::emitRemarks.
  bool printAsEmitRemarks = false;

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
  /// a Remark instance. The remark will be emitted using the main
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
               const MLIRContext::RemarkCategories &cats);

  /// Destructor that will close the output file and reset the
  /// main remark streamer.
  ~RemarkEngine();

  /// Setup the remark engine with the given output path and format.
  LogicalResult initialize(std::unique_ptr<MLIRRemarkStreamerBase> streamer,
                           std::string *errMsg);

  /// Report a remark.
  void report(const Remark &&remark);

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

template <typename Fn, typename... Args>
inline InFlightRemark withEngine(Fn fn, Location loc, Args &&...args) {
  MLIRContext *ctx = loc->getContext();

  RemarkEngine *enginePtr = ctx->getRemarkEngine();

  if (enginePtr)
    return (enginePtr->*fn)(loc, std::forward<Args>(args)...);

  return {};
}

} // namespace mlir::remark::detail

namespace mlir::remark {

template <class... Ts>
inline detail::LazyTextBuild reason(const char *fmt, Ts &&...ts) {
  return {"Reason", [=] { return llvm::formatv(fmt, ts...).str(); }};
}

template <class... Ts>
inline detail::LazyTextBuild suggest(const char *fmt, Ts &&...ts) {
  return {"Suggestion", [=] { return llvm::formatv(fmt, ts...).str(); }};
}

template <class V>
inline detail::LazyTextBuild metric(StringRef key, V &&v) {
  using DV = std::decay_t<V>;
  return {key, [key, vv = DV(std::forward<V>(v))]() mutable {
            // Reuse Arg's formatting logic and return just the value string.
            return detail::Remark::Arg(key, std::move(vv)).val;
          }};
}
//===----------------------------------------------------------------------===//
// Emitters
//===----------------------------------------------------------------------===//

/// Report an optimization remark that was passed.
inline detail::InFlightRemark passed(Location loc, StringRef cat,
                                     StringRef passName = {}) {
  return withEngine(&detail::RemarkEngine::emitOptimizationRemark, loc,
                    passName, cat);
}

/// Report an optimization remark that was missed.
inline detail::InFlightRemark missed(Location loc, llvm::StringRef cat,
                                     llvm::StringRef passName = {}) {
  return withEngine(&detail::RemarkEngine::emitOptimizationRemarkMiss, loc,
                    passName, cat);
}

/// Report an optimization remark that failed.
inline detail::InFlightRemark failed(Location loc, llvm::StringRef cat,
                                     llvm::StringRef passName = {}) {
  return withEngine(&detail::RemarkEngine::emitOptimizationRemarkFailure, loc,
                    passName, cat);
}

/// Report an optimization analysis remark.
inline detail::InFlightRemark analysis(Location loc, StringRef cat,
                                       StringRef passName = {}) {
  return withEngine(&detail::RemarkEngine::emitOptimizationRemarkAnalysis, loc,
                    passName, cat);
}

} // namespace mlir::remark

#endif // MLIR_IR_REMARKS_H
