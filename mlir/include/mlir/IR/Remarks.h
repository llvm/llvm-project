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

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"

#include <atomic>

namespace mlir::remark {

//===----------------------------------------------------------------------===//
// RemarkId - Unique identifier for linking related remarks
//===----------------------------------------------------------------------===//

/// A unique identifier for a remark, used to link related remarks together.
/// An invalid/empty ID has value 0.
class RemarkId {
public:
  RemarkId() : id(0) {}
  explicit RemarkId(uint64_t id) : id(id) {}

  /// Check if this is a valid (non-zero) ID.
  explicit operator bool() const { return id != 0; }

  /// Get the raw ID value.
  uint64_t getValue() const { return id; }

  bool operator==(const RemarkId &other) const { return id == other.id; }
  bool operator!=(const RemarkId &other) const { return id != other.id; }

  /// Print the ID.
  void print(llvm::raw_ostream &os) const { os << "remark-" << id; }

private:
  uint64_t id;
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, RemarkId id) {
  id.print(os);
  return os;
}

/// Define an the set of categories to accept. By default none are, the provided
/// regex matches against the category names for each kind of remark.
struct RemarkCategories {
  std::optional<std::string> all, passed, missed, analysis, failed;
};

/// Categories describe the outcome of an transformation, not the mechanics of
/// emitting/serializing remarks.
enum class RemarkKind {
  RemarkUnknown = 0,

  /// An optimization was applied.
  RemarkPassed,

  /// A profitable optimization opportunity was found but not applied.
  RemarkMissed,

  /// The compiler attempted the optimization but failed (e.g., legality
  /// checks, or better opportunites).
  RemarkFailure,

  /// Informational context (e.g., analysis numbers) without a pass/fail
  /// outcome.
  RemarkAnalysis,
};

using namespace llvm;

namespace detail {
class InFlightRemark; // forward declaration
} // namespace detail

/// Options to create a Remark
struct RemarkOpts {
  StringRef remarkName;      // Identifiable name
  StringRef categoryName;    // Category name (subject to regex filtering)
  StringRef subCategoryName; // Subcategory name
  StringRef functionName;    // Function name if available

  /// Link to a related remark by explicit ID.
  RemarkId relatedId;

  // Construct RemarkOpts from a remark name.
  static RemarkOpts name(StringRef n) {
    RemarkOpts o;
    o.remarkName = n;
    return o;
  }

  /// Return a copy with the category set.
  RemarkOpts category(StringRef v) const {
    auto copy = *this;
    copy.categoryName = v;
    return copy;
  }
  /// Return a copy with the subcategory set.
  RemarkOpts subCategory(StringRef v) const {
    auto copy = *this;
    copy.subCategoryName = v;
    return copy;
  }
  /// Return a copy with the function name set.
  RemarkOpts function(StringRef v) const {
    auto copy = *this;
    copy.functionName = v;
    return copy;
  }

  /// Link this remark to a previously emitted remark by explicit ID.
  RemarkOpts relatedTo(RemarkId id) const {
    auto copy = *this;
    copy.relatedId = id;
    return copy;
  }

  /// Link this remark to a related remark that is still in flight.
  /// Extracts the ID from the InFlightRemark.
  inline RemarkOpts relatedTo(const detail::InFlightRemark &r) const;
};

} // namespace mlir::remark

namespace mlir::remark::detail {
//===----------------------------------------------------------------------===//
// Remark Base Class
//===----------------------------------------------------------------------===//
class Remark {

public:
  Remark(RemarkKind remarkKind, DiagnosticSeverity severity, Location loc,
         RemarkOpts opts)
      : remarkKind(remarkKind), functionName(opts.functionName.str()), loc(loc),
        categoryName(opts.categoryName.str()),
        subCategoryName(opts.subCategoryName.str()),
        remarkName(opts.remarkName.str()) {
    if (!categoryName.empty() && !subCategoryName.empty()) {
      (llvm::Twine(categoryName) + ":" + subCategoryName)
          .toVector(fullCategoryName);
    }
    // Explicit ID linking from opts.
    if (opts.relatedId)
      addRelatedRemark(opts.relatedId);
  }

  // Remark argument that is a key-value pair that can be printed as machine
  // parsable args. For Attribute arguments, the original attribute is also
  // stored to allow custom streamers to handle them specially.
  struct Arg {
    std::string key;
    std::string val;
    /// Optional attribute storage for Attribute-based args. Allows streamers
    /// to access the original attribute for custom handling.
    std::optional<Attribute> attr;

    Arg(llvm::StringRef m) : key("Remark"), val(m) {}
    Arg(llvm::StringRef k, llvm::StringRef v) : key(k), val(v) {}
    Arg(llvm::StringRef k, std::string v) : key(k), val(std::move(v)) {}
    Arg(llvm::StringRef k, const char *v) : Arg(k, llvm::StringRef(v)) {}
    Arg(llvm::StringRef k, Value v);
    Arg(llvm::StringRef k, Type t);
    Arg(llvm::StringRef k, Attribute a);
    Arg(llvm::StringRef k, bool b) : key(k), val(b ? "true" : "false") {}

    /// Check if this arg has an associated attribute.
    bool hasAttribute() const { return attr.has_value(); }

    /// Get the attribute if present.
    Attribute getAttribute() const { return attr.value_or(Attribute()); }

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
    if (!functionName.empty())
      return functionName;
    return "<unknown function>";
  }

  llvm::StringRef getCategoryName() const { return categoryName; }

  llvm::StringRef getCombinedCategoryName() const {
    if (categoryName.empty() && subCategoryName.empty())
      return {};
    if (subCategoryName.empty())
      return categoryName;
    if (categoryName.empty())
      return subCategoryName;
    return fullCategoryName;
  }

  StringRef getRemarkName() const {
    if (remarkName.empty())
      return "<unknown remark name>";
    return remarkName;
  }

  std::string getMsg() const;

  ArrayRef<Arg> getArgs() const { return args; }

  llvm::remarks::Type getRemarkType() const;

  //===--------------------------------------------------------------------===//
  // Remark Linking Support
  //===--------------------------------------------------------------------===//

  /// Get this remark's unique ID (0 if not assigned).
  RemarkId getId() const { return id; }

  /// Set this remark's unique ID. Also adds it as an Arg for serialization.
  void setId(RemarkId newId) {
    id = newId;
    if (id)
      args.emplace_back("RemarkId", llvm::utostr(id.getValue()));
  }

  /// Get the list of related remark IDs.
  ArrayRef<RemarkId> getRelatedRemarkIds() const { return relatedRemarks; }

  /// Add a reference to a related remark. Also adds it as an Arg for
  /// serialization.
  void addRelatedRemark(RemarkId relatedId) {
    if (relatedId) {
      relatedRemarks.push_back(relatedId);
      args.emplace_back("RelatedTo", llvm::utostr(relatedId.getValue()));
    }
  }

  /// Check if this remark has any related remarks.
  bool hasRelatedRemarks() const { return !relatedRemarks.empty(); }

  /// Get the remark kind.
  RemarkKind getRemarkKind() const { return remarkKind; }

  StringRef getRemarkTypeString() const;

protected:
  /// Keeps the MLIR diagnostic kind, which is used to determine the
  /// diagnostic kind in the LLVM remark streamer.
  RemarkKind remarkKind;
  /// Name of the covering function like interface.
  /// Stored as std::string to ensure the Remark owns its data.
  std::string functionName;

  Location loc;
  /// Category name e.g., "Unroll" or "UnrollAndJam".
  /// Stored as std::string to ensure the Remark owns its data.
  std::string categoryName;

  /// Sub category name e.g., "Loop Optimizer".
  /// Stored as std::string to ensure the Remark owns its data.
  std::string subCategoryName;

  /// Combined name for category and sub-category
  SmallString<64> fullCategoryName;

  /// Remark identifier.
  /// Stored as std::string to ensure the Remark owns its data.
  std::string remarkName;

  /// Args collected via the streaming interface.
  SmallVector<Arg, 4> args;

  /// Unique ID for this remark (assigned by RemarkEngine).
  RemarkId id;

  /// IDs of related remarks (e.g., parent analysis that enabled this opt).
  SmallVector<RemarkId> relatedRemarks;

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
    case RemarkKind::RemarkUnknown:
      return llvm::DiagnosticKind::DK_Generic;
    case RemarkKind::RemarkPassed:
      return llvm::DiagnosticKind::DK_OptimizationRemark;
    case RemarkKind::RemarkMissed:
      return llvm::DiagnosticKind::DK_OptimizationRemarkMissed;
    case RemarkKind::RemarkFailure:
      return llvm::DiagnosticKind::DK_OptimizationFailure;
    case RemarkKind::RemarkAnalysis:
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
  explicit OptRemarkBase(Location loc, RemarkOpts opts)
      : Remark(K, S, loc, opts) {}
};

using OptRemarkAnalysis =
    OptRemarkBase<RemarkKind::RemarkAnalysis, DiagnosticSeverity::Remark>;

using OptRemarkPass =
    OptRemarkBase<RemarkKind::RemarkPassed, DiagnosticSeverity::Remark>;

using OptRemarkMissed =
    OptRemarkBase<RemarkKind::RemarkMissed, DiagnosticSeverity::Remark>;

using OptRemarkFailure =
    OptRemarkBase<RemarkKind::RemarkFailure, DiagnosticSeverity::Remark>;

class RemarkEngine;

//===----------------------------------------------------------------------===//
// InFlightRemark
//===----------------------------------------------------------------------===//

/// Lazy text building for zero cost string formatting.
struct LazyTextBuild {
  llvm::StringRef key;
  std::function<std::string()> thunk;
};

/// A wrapper for linking remarks by query - searches the engine's registry
/// at stream time and links to all matching remarks.

/// InFlightRemark is a RAII class that holds a reference to a Remark
/// instance and allows to build the remark using the << operator. The remark
/// is emitted when the InFlightRemark instance is destroyed, which happens
/// when the scope ends or when the InFlightRemark instance is moved.
/// Similar to InFlightDiagnostic, but for remarks.
class InFlightRemark {
public:
  explicit InFlightRemark(std::unique_ptr<Remark> diag)
      : remark(std::move(diag)) {}

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

  /// Get this remark's unique ID (for linking from other remarks).
  RemarkId getId() const { return remark ? remark->getId() : RemarkId(); }

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
// Pluggable Remark Utilities
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

using ReportFn = llvm::unique_function<void(const Remark &)>;

/// Base class for MLIR remark emitting policies that is used to emit
/// optimization remarks to the underlying remark streamer. The derived classes
/// should implement the `reportRemark` method to provide the actual emitting
/// implementation.
class RemarkEmittingPolicyBase {
protected:
  ReportFn reportImpl;

public:
  RemarkEmittingPolicyBase() = default;
  virtual ~RemarkEmittingPolicyBase() = default;

  void initialize(ReportFn fn) { reportImpl = std::move(fn); }

  virtual void reportRemark(const Remark &remark) = 0;
  virtual void finalize() = 0;

  /// Find previously reported remarks matching the given criteria.
  /// Default returns empty -- only policies that store remarks (like
  /// PolicyFinal) override this to enable query-based linking.
  virtual SmallVector<RemarkId>
  findRemarks(const RemarkOpts &opts,
              std::optional<RemarkKind> kind = std::nullopt) const {
    return {};
  }
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
  std::optional<llvm::Regex> passedFilter;
  /// The category for analysis remarks.
  std::optional<llvm::Regex> analysisFilter;
  /// The category for failed optimization remarks.
  std::optional<llvm::Regex> failedFilter;
  /// The MLIR remark streamer that will be used to emit the remarks.
  std::unique_ptr<MLIRRemarkStreamerBase> remarkStreamer;
  /// The MLIR remark policy that will be used to emit the remarks.
  std::unique_ptr<RemarkEmittingPolicyBase> remarkEmittingPolicy;
  /// When is enabled, engine also prints remarks as mlir::emitRemarks.
  bool printAsEmitRemarks = false;
  /// Atomic counter for generating unique remark IDs.
  std::atomic<uint64_t> nextRemarkId{1};

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
    return isMissedOptRemarkEnabled(categoryName) ||
           isPassedOptRemarkEnabled(categoryName) ||
           isFailedOptRemarkEnabled(categoryName) ||
           isAnalysisOptRemarkEnabled(categoryName);
  }

  /// Emit a remark using the given maker function, which should return
  /// a Remark instance. The remark will be emitted using the main
  /// remark streamer.
  template <typename RemarkT>
  InFlightRemark makeRemark(Location loc, RemarkOpts opts);

  template <typename RemarkT>
  InFlightRemark emitIfEnabled(Location loc, RemarkOpts opts,
                               bool (RemarkEngine::*isEnabled)(StringRef)
                                   const);
  /// Report a remark.
  void reportImpl(const Remark &remark);

public:
  /// Default constructor is deleted, use the other constructor.
  RemarkEngine() = delete;

  /// Constructs Remark engine with optional category names. If a category
  /// name is not provided, it is not enabled. The category names are used to
  /// filter the remarks that are emitted.
  RemarkEngine(bool printAsEmitRemarks, const RemarkCategories &cats);

  /// Destructor that will close the output file and reset the
  /// main remark streamer.
  ~RemarkEngine();

  /// Setup the remark engine with the given output path and format.
  LogicalResult
  initialize(std::unique_ptr<MLIRRemarkStreamerBase> streamer,
             std::unique_ptr<RemarkEmittingPolicyBase> remarkEmittingPolicy,
             std::string *errMsg);

  /// Get the remark emitting policy.
  RemarkEmittingPolicyBase *getRemarkEmittingPolicy() const {
    return remarkEmittingPolicy.get();
  }

  /// Generate a unique ID for a new remark.
  RemarkId generateRemarkId() {
    return RemarkId(nextRemarkId.fetch_add(1, std::memory_order_relaxed));
  }

  //===--------------------------------------------------------------------===//
  // Remark Linking - query previously emitted remarks
  //===--------------------------------------------------------------------===//

  /// Find all remarks matching the given criteria. Delegates to the
  /// emitting policy. Only works with policies that store remarks
  /// (e.g. RemarkEmittingPolicyFinal); returns empty for PolicyAll.
  SmallVector<RemarkId>
  findRemarks(const RemarkOpts &opts,
              std::optional<RemarkKind> kind = std::nullopt) const;

  /// Report a remark.
  void report(const Remark &&remark);

  /// Report a successful remark, this will create an InFlightRemark
  /// that can be used to build the remark using the << operator.
  InFlightRemark emitOptimizationRemark(Location loc, RemarkOpts opts);

  /// Report a missed optimization remark
  /// that can be used to build the remark using the << operator.
  InFlightRemark emitOptimizationRemarkMiss(Location loc, RemarkOpts opts);

  /// Report a failed optimization remark, this will create an InFlightRemark
  /// that can be used to build the remark using the << operator.
  InFlightRemark emitOptimizationRemarkFailure(Location loc, RemarkOpts opts);

  /// Report an analysis remark, this will create an InFlightRemark
  /// that can be used to build the remark using the << operator.
  InFlightRemark emitOptimizationRemarkAnalysis(Location loc, RemarkOpts opts);
};

template <typename Fn, typename... Args>
inline InFlightRemark withEngine(Fn fn, Location loc, Args &&...args) {
  MLIRContext *ctx = loc->getContext();

  RemarkEngine *enginePtr = ctx->getRemarkEngine();

  if (LLVM_UNLIKELY(enginePtr))
    return (enginePtr->*fn)(loc, std::forward<Args>(args)...);

  return {};
}

} // namespace mlir::remark::detail

// Deferred definition: needs InFlightRemark to be complete.
inline mlir::remark::RemarkOpts
mlir::remark::RemarkOpts::relatedTo(const detail::InFlightRemark &r) const {
  return relatedTo(r.getId());
}

namespace mlir::remark {

//===----------------------------------------------------------------------===//
// Remark Emitting Policies
//===----------------------------------------------------------------------===//

/// Policy that emits all remarks.
class RemarkEmittingPolicyAll : public detail::RemarkEmittingPolicyBase {
public:
  RemarkEmittingPolicyAll();

  void reportRemark(const detail::Remark &remark) override {
    assert(reportImpl && "reportImpl is not set");
    reportImpl(remark);
  }
  void finalize() override {}
};

/// Policy that emits final remarks. Stores all remarks until finalize(),
/// which enables query-based linking via findRemarks().
class RemarkEmittingPolicyFinal : public detail::RemarkEmittingPolicyBase {
private:
  /// user can intercept them for custom processing via a registered callback,
  /// otherwise they will be reported on engine destruction.
  llvm::DenseSet<detail::Remark> postponedRemarks;

public:
  RemarkEmittingPolicyFinal();

  void reportRemark(const detail::Remark &remark) override {
    postponedRemarks.erase(remark);
    postponedRemarks.insert(remark);
  }

  /// Emits all stored remarks. Related remarks are printed as nested notes
  /// under the remark that references them.
  void finalize() override;
};

/// Create a Reason with llvm::formatv formatting.
template <class... Ts>
inline detail::LazyTextBuild reason(const char *fmt, Ts &&...ts) {
  return {"Reason", [=] { return llvm::formatv(fmt, ts...).str(); }};
}

/// Create a Suggestion with llvm::formatv formatting.
template <class... Ts>
inline detail::LazyTextBuild suggest(const char *fmt, Ts &&...ts) {
  return {"Suggestion", [=] { return llvm::formatv(fmt, ts...).str(); }};
}

/// Create a Remark with llvm::formatv formatting.
template <class... Ts>
inline detail::LazyTextBuild add(const char *fmt, Ts &&...ts) {
  return {"Remark", [=] { return llvm::formatv(fmt, ts...).str(); }};
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
inline detail::InFlightRemark passed(Location loc, RemarkOpts opts) {
  return withEngine(&detail::RemarkEngine::emitOptimizationRemark, loc, opts);
}

/// Report an optimization remark that was missed.
inline detail::InFlightRemark missed(Location loc, RemarkOpts opts) {
  return withEngine(&detail::RemarkEngine::emitOptimizationRemarkMiss, loc,
                    opts);
}

/// Report an optimization remark that failed.
inline detail::InFlightRemark failed(Location loc, RemarkOpts opts) {
  return withEngine(&detail::RemarkEngine::emitOptimizationRemarkFailure, loc,
                    opts);
}

/// Report an optimization analysis remark.
inline detail::InFlightRemark analysis(Location loc, RemarkOpts opts) {
  return withEngine(&detail::RemarkEngine::emitOptimizationRemarkAnalysis, loc,
                    opts);
}

//===----------------------------------------------------------------------===//
// Setup
//===----------------------------------------------------------------------===//

/// Setup remarks for the context. This function will enable the remark engine
/// and set the streamer to be used for optimization remarks. The remark
/// categories are used to filter the remarks that will be emitted by the
/// remark engine. If a category is not specified, it will not be emitted. If
/// `printAsEmitRemarks` is true, the remarks will be printed as
/// mlir::emitRemarks. 'streamer' must inherit from MLIRRemarkStreamerBase and
/// will be used to stream the remarks.
LogicalResult enableOptimizationRemarks(
    MLIRContext &ctx,
    std::unique_ptr<remark::detail::MLIRRemarkStreamerBase> streamer,
    std::unique_ptr<remark::detail::RemarkEmittingPolicyBase>
        remarkEmittingPolicy,
    const remark::RemarkCategories &cats, bool printAsEmitRemarks = false);

} // namespace mlir::remark

// DenseMapInfo specialization for Remark
namespace llvm {
template <>
struct DenseMapInfo<mlir::remark::detail::Remark> {
  static constexpr StringRef kEmptyKey = "<EMPTY_KEY>";
  static constexpr StringRef kTombstoneKey = "<TOMBSTONE_KEY>";

  /// Helper to provide a static dummy context for sentinel keys.
  static mlir::MLIRContext *getStaticDummyContext() {
    static mlir::MLIRContext dummyContext;
    return &dummyContext;
  }

  /// Create an empty remark
  static inline mlir::remark::detail::Remark getEmptyKey() {
    return mlir::remark::detail::Remark(
        mlir::remark::RemarkKind::RemarkUnknown, mlir::DiagnosticSeverity::Note,
        mlir::UnknownLoc::get(getStaticDummyContext()),
        mlir::remark::RemarkOpts::name(kEmptyKey));
  }

  /// Create a dead remark
  static inline mlir::remark::detail::Remark getTombstoneKey() {
    return mlir::remark::detail::Remark(
        mlir::remark::RemarkKind::RemarkUnknown, mlir::DiagnosticSeverity::Note,
        mlir::UnknownLoc::get(getStaticDummyContext()),
        mlir::remark::RemarkOpts::name(kTombstoneKey));
  }

  /// Compute the hash value of the remark
  static unsigned getHashValue(const mlir::remark::detail::Remark &remark) {
    return llvm::hash_combine(
        remark.getLocation().getAsOpaquePointer(),
        llvm::hash_value(remark.getRemarkName()),
        llvm::hash_value(remark.getCombinedCategoryName()),
        static_cast<unsigned>(remark.getRemarkKind()));
  }

  static bool isEqual(const mlir::remark::detail::Remark &lhs,
                      const mlir::remark::detail::Remark &rhs) {
    // Check for empty/tombstone keys first
    if (lhs.getRemarkName() == kEmptyKey ||
        lhs.getRemarkName() == kTombstoneKey ||
        rhs.getRemarkName() == kEmptyKey ||
        rhs.getRemarkName() == kTombstoneKey) {
      return lhs.getRemarkName() == rhs.getRemarkName();
    }

    // For regular remarks, compare key identifying fields
    return lhs.getLocation() == rhs.getLocation() &&
           lhs.getRemarkName() == rhs.getRemarkName() &&
           lhs.getCombinedCategoryName() == rhs.getCombinedCategoryName() &&
           lhs.getRemarkKind() == rhs.getRemarkKind();
  }
};
} // namespace llvm
#endif // MLIR_IR_REMARKS_H
