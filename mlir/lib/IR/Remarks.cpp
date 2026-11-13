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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/StringSaver.h"

#include <mutex>
#include <vector>

using namespace mlir::remark::detail;
using namespace mlir::remark;
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

Remark::Arg::Arg(llvm::StringRef k, Attribute a) : key(k), attr(a) {
  llvm::raw_string_ostream os(val);
  os << a;
}

void Remark::insert(llvm::StringRef s) { args.emplace_back(s); }
void Remark::insert(Arg a) { args.push_back(std::move(a)); }

void Remark::setRelatedRemarkIds(ArrayRef<RemarkId> ids) {
  SmallVector<RemarkId> copy(ids);
  llvm::erase_if(args, [](const Arg &arg) { return arg.isLink; });
  relatedRemarks.clear();
  for (RemarkId id : copy)
    addRelatedRemark(id);
}

// Simple helper to print key=val list (sorted).
static void printArgs(llvm::raw_ostream &os, llvm::ArrayRef<Remark::Arg> args) {
  if (args.empty())
    return;

  // Preserve the relative order of repeated keys, including related IDs.
  llvm::SmallVector<Remark::Arg, 8> sorted(args.begin(), args.end());
  llvm::stable_sort(sorted, [](const Remark::Arg &a, const Remark::Arg &b) {
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
  StringRef categoryName = getCombinedCategoryName();
  StringRef name = remarkName;

  os << '[' << type << "] ";
  os << name << " | ";
  if (!categoryName.empty())
    os << "Category:" << categoryName << " | ";
  if (!functionName.empty())
    os << "Function=" << getFunction() << " | ";

  if (printLocation) {
    if (auto flc = getLocation()->findInstanceOf<FileLineColLoc>()) {
      os << " @" << flc.getFilename() << ":" << flc.getLine() << ":"
         << flc.getColumn();
    }
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
    // Remarks are often emitted at name, fused or call-site locations wrapping
    // a file location; use the first file location found inside.
    if (auto flc = getLocation()->findInstanceOf<FileLineColLoc>())
      return {flc.getFilename(), flc.getLine(), flc.getColumn()};
    return {"<unknown file>", 0, 0};
  };

  llvm::remarks::Remark r; // The result.
  r.RemarkType = getRemarkType();
  r.RemarkName = getRemarkName();
  // MLIR does not use passes; instead, it has categories and sub-categories.
  r.PassName = getCombinedCategoryName();
  r.FunctionName = getFunction();
  r.Loc = locLambda();
  // Keep serialized strings owned by the source remark.
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

InFlightRemark &InFlightRemark::operator<<(const RelatedRemarkQuery &related) {
  if (!remark || !owner)
    return *this;
  // Link each target once, even if an explicit link or an earlier query
  // already added it.
  llvm::SmallDenseSet<uint64_t> linked;
  for (RemarkId id : remark->getRelatedRemarkIds())
    linked.insert(id.getValue());
  for (RemarkId id : owner->findRemarks(related.query))
    if (linked.insert(id.getValue()).second)
      remark->addRelatedRemark(id);
  return *this;
}

InFlightRemark::~InFlightRemark() {
  if (remark && owner)
    owner->report(std::move(*remark));
  owner = nullptr;
}

//===----------------------------------------------------------------------===//
// Remark Engine
//===----------------------------------------------------------------------===//

struct RemarkEngine::RemarkHistory {
  struct Entry {
    RemarkId id;
    RemarkKind kind;
    Location location;
    StringRef name, category, subCategory, function;

    static Entry from(const Remark &remark) {
      return {remark.getId(),           remark.getRemarkKind(),
              remark.getLocation(),     remark.getRawRemarkName(),
              remark.getCategoryName(), remark.getSubCategoryName(),
              remark.getRawFunction()};
    }

    static Entry from(const Remark &remark, llvm::UniqueStringSaver &strings) {
      Entry entry = from(remark);
      entry.name = strings.save(entry.name);
      entry.category = strings.save(entry.category);
      entry.subCategory = strings.save(entry.subCategory);
      entry.function = strings.save(entry.function);
      return entry;
    }
  };

  struct Storage {
    llvm::BumpPtrAllocator allocator;
    llvm::UniqueStringSaver strings{allocator};
    std::vector<Entry> entries;
    // IDs already recorded.
    llvm::DenseSet<uint64_t> recordedIds;
  };

  // Keep the mutex alive for the engine's lifetime so disabling history cannot
  // destroy it while another thread is about to lock it. Storage is opt-in.
  std::mutex mutex;
  std::unique_ptr<Storage> storage;
};

void RemarkEngine::enableHistory(bool enable) {
  std::scoped_lock<std::mutex> lock(history->mutex);
  if (enable) {
    if (!history->storage)
      history->storage = std::make_unique<RemarkHistory::Storage>();
  } else {
    history->storage.reset();
  }
  historyEnabled.store(enable, std::memory_order_release);
}

void RemarkEngine::clearHistory() {
  if (!historyEnabled.load(std::memory_order_acquire))
    return;
  std::scoped_lock<std::mutex> lock(history->mutex);
  if (history->storage)
    history->storage = std::make_unique<RemarkHistory::Storage>();
}

SmallVector<RemarkId>
RemarkEngine::findRemarks(const RemarkQuery &query) const {
  SmallVector<RemarkId> result;
  if (!historyEnabled.load(std::memory_order_acquire))
    return result;

  auto matches = [](const auto &filter, const auto &value) {
    return !filter || *filter == value;
  };
  auto matchesQuery = [&](const RemarkHistory::Entry &entry) {
    return matches(query.location, entry.location) &&
           matches(query.remarkKind, entry.kind) &&
           matches(query.remarkName, entry.name) &&
           matches(query.categoryName, entry.category) &&
           matches(query.subCategoryName, entry.subCategory) &&
           matches(query.functionName, entry.function);
  };
  // Call policies without holding the history lock.
  if (remarkEmittingPolicy)
    remarkEmittingPolicy->forEachPendingRemark([&](const Remark &remark) {
      if (remark.getId() && matchesQuery(RemarkHistory::Entry::from(remark)))
        result.push_back(remark.getId());
    });

  {
    std::scoped_lock<std::mutex> lock(history->mutex);
    if (!history->storage)
      return {};
    for (const RemarkHistory::Entry &entry : history->storage->entries)
      if (matchesQuery(entry))
        result.push_back(entry.id);
  }
  llvm::sort(result, [](RemarkId lhs, RemarkId rhs) {
    return lhs.getValue() < rhs.getValue();
  });
  result.erase(llvm::unique(result), result.end());
  return result;
}

SmallVector<RemarkId>
RemarkEngine::findRemarks(const RemarkOpts &opts,
                          std::optional<RemarkKind> kind) const {
  RemarkQuery query;
  if (!opts.remarkName.empty())
    query.remarkName = opts.remarkName.str();
  if (!opts.categoryName.empty())
    query.categoryName = opts.categoryName.str();
  if (!opts.subCategoryName.empty())
    query.subCategoryName = opts.subCategoryName.str();
  if (!opts.functionName.empty())
    query.functionName = opts.functionName.str();
  query.remarkKind = kind;
  return findRemarks(query);
}

template <typename RemarkT>
InFlightRemark RemarkEngine::makeRemark(Location loc, RemarkOpts opts) {
  static_assert(std::is_base_of_v<Remark, RemarkT>,
                "RemarkT must derive from Remark");
  auto remark = std::make_unique<RemarkT>(loc, opts);
  remark->setId(generateRemarkId());
  return InFlightRemark(*this, std::move(remark));
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

bool RemarkEngine::isRemarkEnabled(RemarkKind kind,
                                   StringRef categoryName) const {
  switch (kind) {
  case RemarkKind::RemarkUnknown:
    return false;
  case RemarkKind::RemarkPassed:
    return isPassedOptRemarkEnabled(categoryName);
  case RemarkKind::RemarkMissed:
    return isMissedOptRemarkEnabled(categoryName);
  case RemarkKind::RemarkFailure:
    return isFailedOptRemarkEnabled(categoryName);
  case RemarkKind::RemarkAnalysis:
    return isAnalysisOptRemarkEnabled(categoryName);
  }
  llvm_unreachable("Unknown remark kind");
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

void RemarkEngine::reportImpl(const Remark &remark) {
  if (historyEnabled.load(std::memory_order_acquire)) {
    std::scoped_lock<std::mutex> lock(history->mutex);
    if (history->storage && remark.getId()) {
      auto &storage = *history->storage;
      // A policy may deliver the same event repeatedly.
      if (storage.recordedIds.insert(remark.getId().getValue()).second)
        storage.entries.push_back(
            RemarkHistory::Entry::from(remark, storage.strings));
    }
  }
  // Stream the remark
  if (remarkStreamer) {
    remarkStreamer->streamOptimizationRemark(remark);
  }

  // Print using MLIR's diagnostic
  if (printAsEmitRemarks)
    emitRemark(remark.getLocation(), remark.getMsg());
}

void RemarkEngine::report(const Remark &&remark) {
  if (remarkEmittingPolicy)
    remarkEmittingPolicy->reportRemark(remark);
}

RemarkEngine::~RemarkEngine() {
  if (remarkEmittingPolicy)
    remarkEmittingPolicy->finalize();

  if (remarkStreamer)
    remarkStreamer->finalize();
}

llvm::LogicalResult RemarkEngine::initialize(
    std::unique_ptr<MLIRRemarkStreamerBase> streamer,
    std::unique_ptr<RemarkEmittingPolicyBase> remarkEmittingPolicy,
    std::string *errMsg) {
  remarkStreamer = std::move(streamer);

  auto reportFunc = llvm::bind_front<&RemarkEngine::reportImpl>(this);
  remarkEmittingPolicy->initialize(ReportFn(std::move(reportFunc)));

  this->remarkEmittingPolicy = std::move(remarkEmittingPolicy);
  return success();
}

/// Returns true if filter is already anchored like ^...$
static bool isAnchored(llvm::StringRef s) {
  s = s.trim();
  return s.starts_with("^") && s.ends_with("$"); // note: startswith/endswith
}

/// Anchor the entire pattern so it matches the whole string.
static std::string anchorWhole(llvm::StringRef filter) {
  if (isAnchored(filter))
    return filter.str();
  return (llvm::Twine("^(") + filter + ")$").str();
}

/// Build a combined filter from cats.all and a category-specific pattern.
/// If neither is present, return std::nullopt. Otherwise "(all|specific)"
/// and anchor once. Also validate before returning.
static std::optional<llvm::Regex>
buildFilter(const mlir::remark::RemarkCategories &cats,
            const std::optional<std::string> &specific) {
  llvm::SmallVector<llvm::StringRef, 2> parts;
  if (cats.all && !cats.all->empty())
    parts.emplace_back(*cats.all);
  if (specific && !specific->empty())
    parts.emplace_back(*specific);

  if (parts.empty())
    return std::nullopt;

  std::string joined = llvm::join(parts, "|");
  std::string anchored = anchorWhole(joined);

  llvm::Regex rx(anchored);
  std::string err;
  if (!rx.isValid(err))
    return std::nullopt;

  return std::make_optional<llvm::Regex>(std::move(rx));
}

RemarkEngine::RemarkEngine(bool printAsEmitRemarks,
                           const RemarkCategories &cats)
    : history(std::make_unique<RemarkHistory>()),
      printAsEmitRemarks(printAsEmitRemarks) {
  if (cats.passed)
    passedFilter = buildFilter(cats, cats.passed);
  if (cats.missed)
    missFilter = buildFilter(cats, cats.missed);
  if (cats.analysis)
    analysisFilter = buildFilter(cats, cats.analysis);
  if (cats.failed)
    failedFilter = buildFilter(cats, cats.failed);
}

llvm::LogicalResult mlir::remark::enableOptimizationRemarks(
    MLIRContext &ctx, std::unique_ptr<detail::MLIRRemarkStreamerBase> streamer,
    std::unique_ptr<detail::RemarkEmittingPolicyBase> remarkEmittingPolicy,
    const RemarkCategories &cats, bool printAsEmitRemarks) {
  auto engine =
      std::make_unique<detail::RemarkEngine>(printAsEmitRemarks, cats);

  std::string errMsg;
  if (failed(engine->initialize(std::move(streamer),
                                std::move(remarkEmittingPolicy), &errMsg))) {
    llvm::report_fatal_error(
        llvm::Twine("Failed to initialize remark engine. Error: ") + errMsg);
  }
  ctx.setRemarkEngine(std::move(engine));

  return success();
}

//===----------------------------------------------------------------------===//
// Remark emitting policies
//===----------------------------------------------------------------------===//

namespace mlir::remark {
RemarkEmittingPolicyAll::RemarkEmittingPolicyAll() = default;

RemarkEmittingPolicyFinal::RemarkEmittingPolicyFinal() = default;

static void dropDiscardedLinks(detail::Remark &remark,
                               const llvm::DenseSet<uint64_t> &discardedIds) {
  if (discardedIds.empty())
    return;
  auto ids = llvm::to_vector(remark.getRelatedRemarkIds());
  llvm::erase_if(
      ids, [&](RemarkId id) { return discardedIds.contains(id.getValue()); });
  if (ids.size() != remark.getRelatedRemarkIds().size())
    remark.setRelatedRemarkIds(ids);
}

void RemarkEmittingPolicyFinal::reportRemark(const detail::Remark &remark) {
  detail::Remark copy = remark;
  std::scoped_lock<std::mutex> lock(mutex);
  auto previous = pendingRemarks.find(copy);
  if (previous != pendingRemarks.end()) {
    if (previous->getId() && previous->getId() != copy.getId())
      discardedIds.insert(previous->getId().getValue());
    pendingRemarks.erase(previous);
  }
  pendingRemarks.insert(std::move(copy));
}

void RemarkEmittingPolicyFinal::forEachPendingRemark(
    llvm::function_ref<void(const detail::Remark &)> callback) const {
  std::scoped_lock<std::mutex> lock(mutex);
  for (const auto &remark : pendingRemarks)
    callback(remark);
}

void RemarkEmittingPolicyFinal::finalize() {
  assert(reportImpl && "reportImpl is not set");

  SmallVector<detail::Remark, 0> remarks;
  llvm::DenseSet<uint64_t> discarded;
  {
    std::scoped_lock<std::mutex> lock(mutex);
    remarks = llvm::to_vector<0>(pendingRemarks);
    pendingRemarks.clear();
    discarded.swap(discardedIds);
  }

  // A re-reported ID that survives is still a valid target.
  for (const auto &remark : remarks)
    if (remark.getId())
      discarded.erase(remark.getId().getValue());
  for (auto &remark : remarks)
    dropDiscardedLinks(remark, discarded);

  llvm::sort(remarks, [](const auto &lhs, const auto &rhs) {
    if (!lhs.getId())
      return false;
    if (!rhs.getId())
      return true;
    return lhs.getId().getValue() < rhs.getId().getValue();
  });

  // Build ID -> Remark* lookup for resolving related remark references.
  llvm::DenseMap<uint64_t, const detail::Remark *> idMap;
  llvm::DenseSet<uint64_t> childIds; // IDs referenced as children

  for (const auto &remark : remarks) {
    if (remark.getId())
      idMap[remark.getId().getValue()] = &remark;
    for (auto relId : remark.getRelatedRemarkIds())
      childIds.insert(relId.getValue());
  }

  // Visit by address so ID-less remarks remain distinct.
  llvm::DenseSet<const detail::Remark *> emitted;
  SmallVector<const detail::Remark *> worklist;
  auto emit = [&](const detail::Remark &root) {
    worklist.push_back(&root);
    while (!worklist.empty()) {
      const auto *remark = worklist.pop_back_val();
      if (!emitted.insert(remark).second)
        continue;
      reportImpl(*remark);
      for (RemarkId id : llvm::reverse(remark->getRelatedRemarkIds()))
        if (const auto *related = idMap.lookup(id.getValue()))
          worklist.push_back(related);
    }
  };

  for (const auto &remark : remarks) {
    if (remark.getId() && childIds.contains(remark.getId().getValue()))
      continue;
    emit(remark);
  }
  // Cycles have no root.
  for (const auto &remark : remarks)
    if (!emitted.contains(&remark))
      emit(remark);
}

} // namespace mlir::remark
