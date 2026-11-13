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

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/Remarks/Remark.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Regex.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"

#include <atomic>
#include <mutex>
#include <optional>
#include <utility>

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
class RemarkEngine;
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

  /// Link to an in-flight remark's ID. Final removes the link if the target is
  /// replaced in the same batch.
  inline RemarkOpts relatedTo(const detail::InFlightRemark &r) const;
};

/// Exact-match criteria for searching emitted or pending remarks. Unspecified
/// fields match any value; a specified empty string matches an empty field.
/// Strings are owned by the query. Locations belong to their MLIRContext and
/// must not outlive it. A location match does not imply operation identity.
class RemarkQuery {
public:
  /// Construct a query matching the given remark name.
  [[nodiscard]] static RemarkQuery named(StringRef value) {
    return RemarkQuery().name(value);
  }

  [[nodiscard]] RemarkQuery name(StringRef value) const & {
    return RemarkQuery(*this).name(value);
  }
  [[nodiscard]] RemarkQuery name(StringRef value) && {
    remarkName = value.str();
    return std::move(*this);
  }
  /// Match the base category, without the subcategory.
  [[nodiscard]] RemarkQuery category(StringRef value) const & {
    return RemarkQuery(*this).category(value);
  }
  [[nodiscard]] RemarkQuery category(StringRef value) && {
    categoryName = value.str();
    return std::move(*this);
  }
  [[nodiscard]] RemarkQuery subCategory(StringRef value) const & {
    return RemarkQuery(*this).subCategory(value);
  }
  [[nodiscard]] RemarkQuery subCategory(StringRef value) && {
    subCategoryName = value.str();
    return std::move(*this);
  }
  [[nodiscard]] RemarkQuery function(StringRef value) const & {
    return RemarkQuery(*this).function(value);
  }
  [[nodiscard]] RemarkQuery function(StringRef value) && {
    functionName = value.str();
    return std::move(*this);
  }
  [[nodiscard]] RemarkQuery kind(RemarkKind value) const & {
    return RemarkQuery(*this).kind(value);
  }
  [[nodiscard]] RemarkQuery kind(RemarkKind value) && {
    remarkKind = value;
    return std::move(*this);
  }
  /// Match the complete location, including any wrapping locations.
  [[nodiscard]] RemarkQuery at(Location value) const & {
    return RemarkQuery(*this).at(value);
  }
  [[nodiscard]] RemarkQuery at(Location value) && {
    location = value;
    return std::move(*this);
  }

private:
  friend class detail::RemarkEngine;
  std::optional<std::string> remarkName, categoryName, subCategoryName,
      functionName;
  std::optional<RemarkKind> remarkKind;
  std::optional<Location> location;
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
    /// Generated RelatedTo argument, distinct from a user-supplied argument.
    bool isLink = false;

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

  /// Return the stored function name, which may be empty.
  StringRef getRawFunction() const { return functionName; }

  llvm::StringRef getCategoryName() const { return categoryName; }

  llvm::StringRef getSubCategoryName() const { return subCategoryName; }

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

  /// Return the stored remark name, which may be empty.
  StringRef getRawRemarkName() const { return remarkName; }

  std::string getMsg() const;

  /// Return arguments, owned until mutation or destruction.
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

  /// Replace the list of related remark IDs.
  void setRelatedRemarkIds(ArrayRef<RemarkId> ids);

  /// Add a related ID and its generated argument.
  void addRelatedRemark(RemarkId relatedId) {
    if (!relatedId)
      return;
    relatedRemarks.push_back(relatedId);
    Arg &arg =
        args.emplace_back("RelatedTo", llvm::utostr(relatedId.getValue()));
    arg.isLink = true;
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

/// Resolves a query when inserted into an in-flight remark.
struct RelatedRemarkQuery {
  RemarkQuery query;
};

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

  InFlightRemark &operator<<(const RelatedRemarkQuery &related);

  // Generic path, excluding wrappers handled by dedicated overloads.
  template <typename T,
            typename = std::enable_if_t<
                !std::is_same_v<std::decay_t<T>, LazyTextBuild> &&
                !std::is_same_v<std::decay_t<T>, RelatedRemarkQuery>>>
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

  /// Visit remarks retained for future emission. findRemarks calls this; do not
  /// call findRemarks while holding the lock taken here. The callback must not
  /// reenter the policy or retain references. The default visits nothing.
  virtual void forEachPendingRemark(
      llvm::function_ref<void(const Remark &)> callback) const {}
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
  /// History records metadata for remarks selected for emission by the policy.
  /// Its mutex has a stable lifetime; retained storage is allocated on demand.
  /// Declared before the streamer and the policy so that a policy destructor
  /// that still reports finds them alive.
  struct RemarkHistory;
  std::unique_ptr<RemarkHistory> history;
  std::atomic<bool> historyEnabled{false};
  /// The MLIR remark streamer that will be used to emit the remarks.
  std::unique_ptr<MLIRRemarkStreamerBase> remarkStreamer;
  /// The MLIR remark policy that will be used to emit the remarks.
  std::unique_ptr<RemarkEmittingPolicyBase> remarkEmittingPolicy;
  /// When is enabled, engine also prints remarks as mlir::emitRemarks.
  bool printAsEmitRemarks = false;
  /// Atomic counter for generating unique remark IDs.
  std::atomic<uint64_t> nextRemarkId{1};

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
  // Remark Filtering - query which remarks are enabled
  //===--------------------------------------------------------------------===//

  /// Return true if missed optimization remarks are enabled for the given
  /// category.
  bool isMissedOptRemarkEnabled(StringRef categoryName) const;

  /// Return true if passed optimization remarks are enabled for the given
  /// category.
  bool isPassedOptRemarkEnabled(StringRef categoryName) const;

  /// Return true if analysis remarks are enabled for the given category.
  bool isAnalysisOptRemarkEnabled(StringRef categoryName) const;

  /// Return true if failed optimization remarks are enabled for the given
  /// category.
  bool isFailedOptRemarkEnabled(StringRef categoryName) const;

  /// Return true if remarks of the given kind are enabled for the given
  /// category. Always returns false for `RemarkKind::RemarkUnknown`.
  bool isRemarkEnabled(RemarkKind kind, StringRef categoryName) const;

  /// Return true if any kind of remark is enabled for the given category.
  bool isAnyRemarkEnabled(StringRef categoryName) const {
    return isMissedOptRemarkEnabled(categoryName) ||
           isPassedOptRemarkEnabled(categoryName) ||
           isFailedOptRemarkEnabled(categoryName) ||
           isAnalysisOptRemarkEnabled(categoryName);
  }

  /// Return true if any kind of remark is enabled for any category, i.e. if at
  /// least one category filter is active. This is a cheap check that external
  /// remark producers can use before doing any work to build a remark.
  bool isAnyRemarkEnabled() const {
    return missFilter.has_value() || passedFilter.has_value() ||
           analysisFilter.has_value() || failedFilter.has_value();
  }

  //===--------------------------------------------------------------------===//
  // Remark Linking - query emitted and pending remarks
  //===--------------------------------------------------------------------===//

  /// Enable or disable history. Enabling records subsequent emissions and makes
  /// pending policy remarks searchable; disabling discards the recorded
  /// metadata and suppresses queries.
  void enableHistory(bool enable = true);

  /// Discard recorded metadata without changing whether history is enabled.
  void clearHistory();

  /// Find matching emitted remarks and pending candidates in ID order, each
  /// once. Returns empty without history; otherwise scans all pending and
  /// emitted remarks. Candidates may be replaced before emission. Queries may
  /// overlap reports, but not Final::finalize(). Cross-function matches depend
  /// on thread timing; scope queries with function().
  SmallVector<RemarkId> findRemarks(const RemarkQuery &query) const;

  /// Search using nonempty option fields as exact-match criteria. The related
  /// ID in opts is ignored. Use RemarkQuery to match empty fields or a
  /// location.
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
/// Link to all remarks matching the query, each once, when inserted into an
/// in-flight remark. Requires history; no match adds no links. Final drops
/// links to targets replaced in the same finalization batch.
inline detail::RelatedRemarkQuery relatedTo(RemarkQuery query) {
  return {std::move(query)};
}

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

/// Policy that emits only the last remark reported for each identity, see
/// DenseMapInfo<Remark>. Remarks are stored until finalize().
class RemarkEmittingPolicyFinal : public detail::RemarkEmittingPolicyBase {
public:
  RemarkEmittingPolicyFinal();

  void reportRemark(const detail::Remark &remark) override;

  /// Visit pending candidates. Must not overlap finalize().
  void forEachPendingRemark(
      llvm::function_ref<void(const detail::Remark &)> callback) const override;

  /// Emit each survivor once, grouping related remarks after their parents.
  /// Visit roots with IDs in ID order, then ID-less roots in unspecified order.
  /// Finally, start traversals from any unvisited remarks in ID order.
  /// Links to IDs replaced in this batch are removed; IDs outside this batch
  /// are preserved without validation. Call after reports and queries stop.
  /// Callbacks must not reenter the policy. Later calls emit only new remarks.
  void finalize() override;

private:
  mutable std::mutex mutex;
  llvm::DenseSet<detail::Remark> pendingRemarks;
  /// IDs replaced in the current batch.
  llvm::DenseSet<uint64_t> discardedIds;
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

/// Two remarks are the same for RemarkEmittingPolicyFinal when they have the
/// same location, remark name, combined category name and kind.
namespace llvm {
template <>
struct DenseMapInfo<mlir::remark::detail::Remark> {
  static unsigned getHashValue(const mlir::remark::detail::Remark &remark) {
    return llvm::hash_combine(
        remark.getLocation().getAsOpaquePointer(),
        llvm::hash_value(remark.getRemarkName()),
        llvm::hash_value(remark.getCombinedCategoryName()),
        static_cast<unsigned>(remark.getRemarkKind()));
  }

  static bool isEqual(const mlir::remark::detail::Remark &lhs,
                      const mlir::remark::detail::Remark &rhs) {
    return lhs.getLocation() == rhs.getLocation() &&
           lhs.getRemarkName() == rhs.getRemarkName() &&
           lhs.getCombinedCategoryName() == rhs.getCombinedCategoryName() &&
           lhs.getRemarkKind() == rhs.getRemarkKind();
  }
};
} // namespace llvm
#endif // MLIR_IR_REMARKS_H
