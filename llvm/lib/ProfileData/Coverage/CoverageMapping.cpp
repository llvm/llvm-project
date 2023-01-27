//===- CoverageMapping.cpp - Code coverage mapping support ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains support for clang's and llvm's instrumentation based
// code coverage.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/Coverage/CoverageMapping.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/BuildID.h"
#include "llvm/ProfileData/Coverage/CoverageMappingReader.h"
#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

using namespace llvm;
using namespace coverage;

#define DEBUG_TYPE "coverage-mapping"

Counter CounterExpressionBuilder::get(const CounterExpression &E) {
  auto It = ExpressionIndices.find(E);
  if (It != ExpressionIndices.end())
    return Counter::getExpression(It->second);
  unsigned I = Expressions.size();
  Expressions.push_back(E);
  ExpressionIndices[E] = I;
  return Counter::getExpression(I);
}

void CounterExpressionBuilder::extractTerms(Counter C, int Factor,
                                            SmallVectorImpl<Term> &Terms) {
  switch (C.getKind()) {
  case Counter::Zero:
    break;
  case Counter::CounterValueReference:
    Terms.emplace_back(C.getCounterID(), Factor);
    break;
  case Counter::Expression:
    const auto &E = Expressions[C.getExpressionID()];
    extractTerms(E.LHS, Factor, Terms);
    extractTerms(
        E.RHS, E.Kind == CounterExpression::Subtract ? -Factor : Factor, Terms);
    break;
  }
}

Counter CounterExpressionBuilder::simplify(Counter ExpressionTree) {
  // Gather constant terms.
  SmallVector<Term, 32> Terms;
  extractTerms(ExpressionTree, +1, Terms);

  // If there are no terms, this is just a zero. The algorithm below assumes at
  // least one term.
  if (Terms.size() == 0)
    return Counter::getZero();

  // Group the terms by counter ID.
  llvm::sort(Terms, [](const Term &LHS, const Term &RHS) {
    return LHS.CounterID < RHS.CounterID;
  });

  // Combine terms by counter ID to eliminate counters that sum to zero.
  auto Prev = Terms.begin();
  for (auto I = Prev + 1, E = Terms.end(); I != E; ++I) {
    if (I->CounterID == Prev->CounterID) {
      Prev->Factor += I->Factor;
      continue;
    }
    ++Prev;
    *Prev = *I;
  }
  Terms.erase(++Prev, Terms.end());

  Counter C;
  // Create additions. We do this before subtractions to avoid constructs like
  // ((0 - X) + Y), as opposed to (Y - X).
  for (auto T : Terms) {
    if (T.Factor <= 0)
      continue;
    for (int I = 0; I < T.Factor; ++I)
      if (C.isZero())
        C = Counter::getCounter(T.CounterID);
      else
        C = get(CounterExpression(CounterExpression::Add, C,
                                  Counter::getCounter(T.CounterID)));
  }

  // Create subtractions.
  for (auto T : Terms) {
    if (T.Factor >= 0)
      continue;
    for (int I = 0; I < -T.Factor; ++I)
      C = get(CounterExpression(CounterExpression::Subtract, C,
                                Counter::getCounter(T.CounterID)));
  }
  return C;
}

Counter CounterExpressionBuilder::add(Counter LHS, Counter RHS, bool Simplify) {
  auto Cnt = get(CounterExpression(CounterExpression::Add, LHS, RHS));
  return Simplify ? simplify(Cnt) : Cnt;
}

Counter CounterExpressionBuilder::subtract(Counter LHS, Counter RHS,
                                           bool Simplify) {
  auto Cnt = get(CounterExpression(CounterExpression::Subtract, LHS, RHS));
  return Simplify ? simplify(Cnt) : Cnt;
}

void CounterMappingContext::dump(const Counter &C, raw_ostream &OS) const {
  switch (C.getKind()) {
  case Counter::Zero:
    OS << '0';
    return;
  case Counter::CounterValueReference:
    OS << '#' << C.getCounterID();
    break;
  case Counter::Expression: {
    if (C.getExpressionID() >= Expressions.size())
      return;
    const auto &E = Expressions[C.getExpressionID()];
    OS << '(';
    dump(E.LHS, OS);
    OS << (E.Kind == CounterExpression::Subtract ? " - " : " + ");
    dump(E.RHS, OS);
    OS << ')';
    break;
  }
  }
  if (CounterValues.empty())
    return;
  Expected<int64_t> Value = evaluate(C);
  if (auto E = Value.takeError()) {
    consumeError(std::move(E));
    return;
  }
  OS << '[' << *Value << ']';
}

Expected<int64_t> CounterMappingContext::evaluate(const Counter &C) const {
  switch (C.getKind()) {
  case Counter::Zero:
    return 0;
  case Counter::CounterValueReference:
    if (C.getCounterID() >= CounterValues.size())
      return errorCodeToError(errc::argument_out_of_domain);
    return CounterValues[C.getCounterID()];
  case Counter::Expression: {
    if (C.getExpressionID() >= Expressions.size())
      return errorCodeToError(errc::argument_out_of_domain);
    const auto &E = Expressions[C.getExpressionID()];
    Expected<int64_t> LHS = evaluate(E.LHS);
    if (!LHS)
      return LHS;
    Expected<int64_t> RHS = evaluate(E.RHS);
    if (!RHS)
      return RHS;
    return E.Kind == CounterExpression::Subtract ? *LHS - *RHS : *LHS + *RHS;
  }
  }
  llvm_unreachable("Unhandled CounterKind");
}

unsigned CounterMappingContext::getMaxCounterID(const Counter &C) const {
  switch (C.getKind()) {
  case Counter::Zero:
    return 0;
  case Counter::CounterValueReference:
    return C.getCounterID();
  case Counter::Expression: {
    if (C.getExpressionID() >= Expressions.size())
      return 0;
    const auto &E = Expressions[C.getExpressionID()];
    return std::max(getMaxCounterID(E.LHS), getMaxCounterID(E.RHS));
  }
  }
  llvm_unreachable("Unhandled CounterKind");
}

void FunctionRecordIterator::skipOtherFiles() {
  while (Current != Records.end() && !Filename.empty() &&
         Filename != Current->Filenames[0])
    ++Current;
  if (Current == Records.end())
    *this = FunctionRecordIterator();
}

ArrayRef<unsigned> CoverageMapping::getImpreciseRecordIndicesForFilename(
    StringRef Filename) const {
  size_t FilenameHash = hash_value(Filename);
  auto RecordIt = FilenameHash2RecordIndices.find(FilenameHash);
  if (RecordIt == FilenameHash2RecordIndices.end())
    return {};
  return RecordIt->second;
}

static unsigned getMaxCounterID(const CounterMappingContext &Ctx,
                                const CoverageMappingRecord &Record) {
  unsigned MaxCounterID = 0;
  for (const auto &Region : Record.MappingRegions) {
    MaxCounterID = std::max(MaxCounterID, Ctx.getMaxCounterID(Region.Count));
  }
  return MaxCounterID;
}

Error CoverageMapping::loadFunctionRecord(
    const CoverageMappingRecord &Record,
    IndexedInstrProfReader &ProfileReader) {
  StringRef OrigFuncName = Record.FunctionName;
  if (OrigFuncName.empty())
    return make_error<CoverageMapError>(coveragemap_error::malformed);

  if (Record.Filenames.empty())
    OrigFuncName = getFuncNameWithoutPrefix(OrigFuncName);
  else
    OrigFuncName = getFuncNameWithoutPrefix(OrigFuncName, Record.Filenames[0]);

  CounterMappingContext Ctx(Record.Expressions);

  std::vector<uint64_t> Counts;
  if (Error E = ProfileReader.getFunctionCounts(Record.FunctionName,
                                                Record.FunctionHash, Counts)) {
    instrprof_error IPE = InstrProfError::take(std::move(E));
    if (IPE == instrprof_error::hash_mismatch) {
      FuncHashMismatches.emplace_back(std::string(Record.FunctionName),
                                      Record.FunctionHash);
      return Error::success();
    } else if (IPE != instrprof_error::unknown_function)
      return make_error<InstrProfError>(IPE);
    Counts.assign(getMaxCounterID(Ctx, Record) + 1, 0);
  }
  Ctx.setCounts(Counts);

  assert(!Record.MappingRegions.empty() && "Function has no regions");

  // This coverage record is a zero region for a function that's unused in
  // some TU, but used in a different TU. Ignore it. The coverage maps from the
  // the other TU will either be loaded (providing full region counts) or they
  // won't (in which case we don't unintuitively report functions as uncovered
  // when they have non-zero counts in the profile).
  if (Record.MappingRegions.size() == 1 &&
      Record.MappingRegions[0].Count.isZero() && Counts[0] > 0)
    return Error::success();

  FunctionRecord Function(OrigFuncName, Record.Filenames);
  for (const auto &Region : Record.MappingRegions) {
    Expected<int64_t> ExecutionCount = Ctx.evaluate(Region.Count);
    if (auto E = ExecutionCount.takeError()) {
      consumeError(std::move(E));
      return Error::success();
    }
    Expected<int64_t> AltExecutionCount = Ctx.evaluate(Region.FalseCount);
    if (auto E = AltExecutionCount.takeError()) {
      consumeError(std::move(E));
      return Error::success();
    }
    Function.pushRegion(Region, *ExecutionCount, *AltExecutionCount);
  }

  // Don't create records for (filenames, function) pairs we've already seen.
  auto FilenamesHash = hash_combine_range(Record.Filenames.begin(),
                                          Record.Filenames.end());
  if (!RecordProvenance[FilenamesHash].insert(hash_value(OrigFuncName)).second)
    return Error::success();

  Functions.push_back(std::move(Function));

  // Performance optimization: keep track of the indices of the function records
  // which correspond to each filename. This can be used to substantially speed
  // up queries for coverage info in a file.
  unsigned RecordIndex = Functions.size() - 1;
  for (StringRef Filename : Record.Filenames) {
    auto &RecordIndices = FilenameHash2RecordIndices[hash_value(Filename)];
    // Note that there may be duplicates in the filename set for a function
    // record, because of e.g. macro expansions in the function in which both
    // the macro and the function are defined in the same file.
    if (RecordIndices.empty() || RecordIndices.back() != RecordIndex)
      RecordIndices.push_back(RecordIndex);
  }

  return Error::success();
}

// This function is for memory optimization by shortening the lifetimes
// of CoverageMappingReader instances.
Error CoverageMapping::loadFromReaders(
    ArrayRef<std::unique_ptr<CoverageMappingReader>> CoverageReaders,
    IndexedInstrProfReader &ProfileReader, CoverageMapping &Coverage) {
  for (const auto &CoverageReader : CoverageReaders) {
    for (auto RecordOrErr : *CoverageReader) {
      if (Error E = RecordOrErr.takeError())
        return E;
      const auto &Record = *RecordOrErr;
      if (Error E = Coverage.loadFunctionRecord(Record, ProfileReader))
        return E;
    }
  }
  return Error::success();
}

Expected<std::unique_ptr<CoverageMapping>> CoverageMapping::load(
    ArrayRef<std::unique_ptr<CoverageMappingReader>> CoverageReaders,
    IndexedInstrProfReader &ProfileReader) {
  auto Coverage = std::unique_ptr<CoverageMapping>(new CoverageMapping());
  if (Error E = loadFromReaders(CoverageReaders, ProfileReader, *Coverage))
    return std::move(E);
  return std::move(Coverage);
}

// If E is a no_data_found error, returns success. Otherwise returns E.
static Error handleMaybeNoDataFoundError(Error E) {
  return handleErrors(
      std::move(E), [](const CoverageMapError &CME) {
        if (CME.get() == coveragemap_error::no_data_found)
          return static_cast<Error>(Error::success());
        return make_error<CoverageMapError>(CME.get());
      });
}

Error CoverageMapping::loadFromFile(
    StringRef Filename, StringRef Arch, StringRef CompilationDir,
    IndexedInstrProfReader &ProfileReader, CoverageMapping &Coverage,
    bool &DataFound, SmallVectorImpl<object::BuildID> *FoundBinaryIDs) {
  auto CovMappingBufOrErr = MemoryBuffer::getFileOrSTDIN(
      Filename, /*IsText=*/false, /*RequiresNullTerminator=*/false);
  if (std::error_code EC = CovMappingBufOrErr.getError())
    return createFileError(Filename, errorCodeToError(EC));
  MemoryBufferRef CovMappingBufRef =
      CovMappingBufOrErr.get()->getMemBufferRef();
  SmallVector<std::unique_ptr<MemoryBuffer>, 4> Buffers;

  SmallVector<object::BuildIDRef> BinaryIDs;
  auto CoverageReadersOrErr = BinaryCoverageReader::create(
      CovMappingBufRef, Arch, Buffers, CompilationDir,
      FoundBinaryIDs ? &BinaryIDs : nullptr);
  if (Error E = CoverageReadersOrErr.takeError()) {
    E = handleMaybeNoDataFoundError(std::move(E));
    if (E)
      return createFileError(Filename, std::move(E));
    return E;
  }

  SmallVector<std::unique_ptr<CoverageMappingReader>, 4> Readers;
  for (auto &Reader : CoverageReadersOrErr.get())
    Readers.push_back(std::move(Reader));
  if (FoundBinaryIDs && !Readers.empty()) {
    llvm::append_range(*FoundBinaryIDs,
                       llvm::map_range(BinaryIDs, [](object::BuildIDRef BID) {
                         return object::BuildID(BID);
                       }));
  }
  DataFound |= !Readers.empty();
  if (Error E = loadFromReaders(Readers, ProfileReader, Coverage))
    return createFileError(Filename, std::move(E));
  return Error::success();
}

Expected<std::unique_ptr<CoverageMapping>>
CoverageMapping::load(ArrayRef<StringRef> ObjectFilenames,
                      StringRef ProfileFilename, ArrayRef<StringRef> Arches,
                      StringRef CompilationDir,
                      const object::BuildIDFetcher *BIDFetcher) {
  auto ProfileReaderOrErr = IndexedInstrProfReader::create(ProfileFilename);
  if (Error E = ProfileReaderOrErr.takeError())
    return createFileError(ProfileFilename, std::move(E));
  auto ProfileReader = std::move(ProfileReaderOrErr.get());
  auto Coverage = std::unique_ptr<CoverageMapping>(new CoverageMapping());
  bool DataFound = false;

  auto GetArch = [&](size_t Idx) {
    if (Arches.empty())
      return StringRef();
    if (Arches.size() == 1)
      return Arches.front();
    return Arches[Idx];
  };

  SmallVector<object::BuildID> FoundBinaryIDs;
  for (const auto &File : llvm::enumerate(ObjectFilenames)) {
    if (Error E =
            loadFromFile(File.value(), GetArch(File.index()), CompilationDir,
                         *ProfileReader, *Coverage, DataFound, &FoundBinaryIDs))
      return std::move(E);
  }

  if (BIDFetcher) {
    const auto &CompareLT = [](object::BuildIDRef A, object::BuildIDRef B) {
      return StringRef(reinterpret_cast<const char *>(A.data()), A.size()) <
             StringRef(reinterpret_cast<const char *>(B.data()), B.size());
    };
    const auto &CompareEQ = [](object::BuildIDRef A, object::BuildIDRef B) {
      return StringRef(reinterpret_cast<const char *>(A.data()), A.size()) ==
             StringRef(reinterpret_cast<const char *>(B.data()), B.size());
    };
    std::vector<object::BuildID> ProfileBinaryIDs;
    if (Error E = ProfileReader->readBinaryIds(ProfileBinaryIDs))
      return createFileError(ProfileFilename, std::move(E));
    llvm::sort(ProfileBinaryIDs, CompareLT);
    ProfileBinaryIDs.erase(llvm::unique(ProfileBinaryIDs, CompareEQ),
                           ProfileBinaryIDs.end());

    SmallVector<object::BuildIDRef> BinaryIDsToFetch;
    if (!ProfileBinaryIDs.empty()) {
      llvm::sort(FoundBinaryIDs, CompareLT);
      FoundBinaryIDs.erase(llvm::unique(FoundBinaryIDs, CompareEQ),
                           FoundBinaryIDs.end());
      std::set_difference(
          ProfileBinaryIDs.begin(), ProfileBinaryIDs.end(),
          FoundBinaryIDs.begin(), FoundBinaryIDs.end(),
          std::inserter(BinaryIDsToFetch, BinaryIDsToFetch.end()), CompareLT);
    }

    for (object::BuildIDRef BinaryID : BinaryIDsToFetch) {
      std::optional<std::string> PathOpt = BIDFetcher->fetch(BinaryID);
      if (!PathOpt)
        continue;
      std::string Path = std::move(*PathOpt);
      StringRef Arch = Arches.size() == 1 ? Arches.front() : StringRef();
      if (Error E = loadFromFile(Path, Arch, CompilationDir, *ProfileReader,
                                 *Coverage, DataFound))
        return std::move(E);
    }
  }

  if (!DataFound)
    return createFileError(
        join(ObjectFilenames.begin(), ObjectFilenames.end(), ", "),
        make_error<CoverageMapError>(coveragemap_error::no_data_found));
  return std::move(Coverage);
}

namespace {

/// Distributes functions into instantiation sets.
///
/// An instantiation set is a collection of functions that have the same source
/// code, ie, template functions specializations.
class FunctionInstantiationSetCollector {
  using MapT = std::map<LineColPair, std::vector<const FunctionRecord *>>;
  MapT InstantiatedFunctions;

public:
  void insert(const FunctionRecord &Function, unsigned FileID) {
    auto I = Function.CountedRegions.begin(), E = Function.CountedRegions.end();
    while (I != E && I->FileID != FileID)
      ++I;
    assert(I != E && "function does not cover the given file");
    auto &Functions = InstantiatedFunctions[I->startLoc()];
    Functions.push_back(&Function);
  }

  MapT::iterator begin() { return InstantiatedFunctions.begin(); }
  MapT::iterator end() { return InstantiatedFunctions.end(); }
};

class SegmentBuilder {
  std::vector<CoverageSegment> &Segments;
  SmallVector<const CountedRegion *, 8> ActiveRegions;

  SegmentBuilder(std::vector<CoverageSegment> &Segments) : Segments(Segments) {}

  /// Emit a segment with the count from \p Region starting at \p StartLoc.
  //
  /// \p IsRegionEntry: The segment is at the start of a new non-gap region.
  /// \p EmitSkippedRegion: The segment must be emitted as a skipped region.
  void startSegment(const CountedRegion &Region, LineColPair StartLoc,
                    bool IsRegionEntry, bool EmitSkippedRegion = false) {
    bool HasCount = !EmitSkippedRegion &&
                    (Region.Kind != CounterMappingRegion::SkippedRegion);

    // If the new segment wouldn't affect coverage rendering, skip it.
    if (!Segments.empty() && !IsRegionEntry && !EmitSkippedRegion) {
      const auto &Last = Segments.back();
      if (Last.HasCount == HasCount && Last.Count == Region.ExecutionCount &&
          !Last.IsRegionEntry)
        return;
    }

    if (HasCount)
      Segments.emplace_back(StartLoc.first, StartLoc.second,
                            Region.ExecutionCount, IsRegionEntry,
                            Region.Kind == CounterMappingRegion::GapRegion);
    else
      Segments.emplace_back(StartLoc.first, StartLoc.second, IsRegionEntry);

    LLVM_DEBUG({
      const auto &Last = Segments.back();
      dbgs() << "Segment at " << Last.Line << ":" << Last.Col
             << " (count = " << Last.Count << ")"
             << (Last.IsRegionEntry ? ", RegionEntry" : "")
             << (!Last.HasCount ? ", Skipped" : "")
             << (Last.IsGapRegion ? ", Gap" : "") << "\n";
    });
  }

  /// Emit segments for active regions which end before \p Loc.
  ///
  /// \p Loc: The start location of the next region. If std::nullopt, all active
  /// regions are completed.
  /// \p FirstCompletedRegion: Index of the first completed region.
  void completeRegionsUntil(std::optional<LineColPair> Loc,
                            unsigned FirstCompletedRegion) {
    // Sort the completed regions by end location. This makes it simple to
    // emit closing segments in sorted order.
    auto CompletedRegionsIt = ActiveRegions.begin() + FirstCompletedRegion;
    std::stable_sort(CompletedRegionsIt, ActiveRegions.end(),
                      [](const CountedRegion *L, const CountedRegion *R) {
                        return L->endLoc() < R->endLoc();
                      });

    // Emit segments for all completed regions.
    for (unsigned I = FirstCompletedRegion + 1, E = ActiveRegions.size(); I < E;
         ++I) {
      const auto *CompletedRegion = ActiveRegions[I];
      assert((!Loc || CompletedRegion->endLoc() <= *Loc) &&
             "Completed region ends after start of new region");

      const auto *PrevCompletedRegion = ActiveRegions[I - 1];
      auto CompletedSegmentLoc = PrevCompletedRegion->endLoc();

      // Don't emit any more segments if they start where the new region begins.
      if (Loc && CompletedSegmentLoc == *Loc)
        break;

      // Don't emit a segment if the next completed region ends at the same
      // location as this one.
      if (CompletedSegmentLoc == CompletedRegion->endLoc())
        continue;

      // Use the count from the last completed region which ends at this loc.
      for (unsigned J = I + 1; J < E; ++J)
        if (CompletedRegion->endLoc() == ActiveRegions[J]->endLoc())
          CompletedRegion = ActiveRegions[J];

      startSegment(*CompletedRegion, CompletedSegmentLoc, false);
    }

    auto Last = ActiveRegions.back();
    if (FirstCompletedRegion && Last->endLoc() != *Loc) {
      // If there's a gap after the end of the last completed region and the
      // start of the new region, use the last active region to fill the gap.
      startSegment(*ActiveRegions[FirstCompletedRegion - 1], Last->endLoc(),
                   false);
    } else if (!FirstCompletedRegion && (!Loc || *Loc != Last->endLoc())) {
      // Emit a skipped segment if there are no more active regions. This
      // ensures that gaps between functions are marked correctly.
      startSegment(*Last, Last->endLoc(), false, true);
    }

    // Pop the completed regions.
    ActiveRegions.erase(CompletedRegionsIt, ActiveRegions.end());
  }

  void buildSegmentsImpl(ArrayRef<CountedRegion> Regions) {
    for (const auto &CR : enumerate(Regions)) {
      auto CurStartLoc = CR.value().startLoc();

      // Active regions which end before the current region need to be popped.
      auto CompletedRegions =
          std::stable_partition(ActiveRegions.begin(), ActiveRegions.end(),
                                [&](const CountedRegion *Region) {
                                  return !(Region->endLoc() <= CurStartLoc);
                                });
      if (CompletedRegions != ActiveRegions.end()) {
        unsigned FirstCompletedRegion =
            std::distance(ActiveRegions.begin(), CompletedRegions);
        completeRegionsUntil(CurStartLoc, FirstCompletedRegion);
      }

      bool GapRegion = CR.value().Kind == CounterMappingRegion::GapRegion;

      // Try to emit a segment for the current region.
      if (CurStartLoc == CR.value().endLoc()) {
        // Avoid making zero-length regions active. If it's the last region,
        // emit a skipped segment. Otherwise use its predecessor's count.
        const bool Skipped =
            (CR.index() + 1) == Regions.size() ||
            CR.value().Kind == CounterMappingRegion::SkippedRegion;
        startSegment(ActiveRegions.empty() ? CR.value() : *ActiveRegions.back(),
                     CurStartLoc, !GapRegion, Skipped);
        // If it is skipped segment, create a segment with last pushed
        // regions's count at CurStartLoc.
        if (Skipped && !ActiveRegions.empty())
          startSegment(*ActiveRegions.back(), CurStartLoc, false);
        continue;
      }
      if (CR.index() + 1 == Regions.size() ||
          CurStartLoc != Regions[CR.index() + 1].startLoc()) {
        // Emit a segment if the next region doesn't start at the same location
        // as this one.
        startSegment(CR.value(), CurStartLoc, !GapRegion);
      }

      // This region is active (i.e not completed).
      ActiveRegions.push_back(&CR.value());
    }

    // Complete any remaining active regions.
    if (!ActiveRegions.empty())
      completeRegionsUntil(std::nullopt, 0);
  }

  /// Sort a nested sequence of regions from a single file.
  static void sortNestedRegions(MutableArrayRef<CountedRegion> Regions) {
    llvm::sort(Regions, [](const CountedRegion &LHS, const CountedRegion &RHS) {
      if (LHS.startLoc() != RHS.startLoc())
        return LHS.startLoc() < RHS.startLoc();
      if (LHS.endLoc() != RHS.endLoc())
        // When LHS completely contains RHS, we sort LHS first.
        return RHS.endLoc() < LHS.endLoc();
      // If LHS and RHS cover the same area, we need to sort them according
      // to their kinds so that the most suitable region will become "active"
      // in combineRegions(). Because we accumulate counter values only from
      // regions of the same kind as the first region of the area, prefer
      // CodeRegion to ExpansionRegion and ExpansionRegion to SkippedRegion.
      static_assert(CounterMappingRegion::CodeRegion <
                            CounterMappingRegion::ExpansionRegion &&
                        CounterMappingRegion::ExpansionRegion <
                            CounterMappingRegion::SkippedRegion,
                    "Unexpected order of region kind values");
      return LHS.Kind < RHS.Kind;
    });
  }

  /// Combine counts of regions which cover the same area.
  static ArrayRef<CountedRegion>
  combineRegions(MutableArrayRef<CountedRegion> Regions) {
    if (Regions.empty())
      return Regions;
    auto Active = Regions.begin();
    auto End = Regions.end();
    for (auto I = Regions.begin() + 1; I != End; ++I) {
      if (Active->startLoc() != I->startLoc() ||
          Active->endLoc() != I->endLoc()) {
        // Shift to the next region.
        ++Active;
        if (Active != I)
          *Active = *I;
        continue;
      }
      // Merge duplicate region.
      // If CodeRegions and ExpansionRegions cover the same area, it's probably
      // a macro which is fully expanded to another macro. In that case, we need
      // to accumulate counts only from CodeRegions, or else the area will be
      // counted twice.
      // On the other hand, a macro may have a nested macro in its body. If the
      // outer macro is used several times, the ExpansionRegion for the nested
      // macro will also be added several times. These ExpansionRegions cover
      // the same source locations and have to be combined to reach the correct
      // value for that area.
      // We add counts of the regions of the same kind as the active region
      // to handle the both situations.
      if (I->Kind == Active->Kind)
        Active->ExecutionCount += I->ExecutionCount;
    }
    return Regions.drop_back(std::distance(++Active, End));
  }

public:
  /// Build a sorted list of CoverageSegments from a list of Regions.
  static std::vector<CoverageSegment>
  buildSegments(MutableArrayRef<CountedRegion> Regions) {
    std::vector<CoverageSegment> Segments;
    SegmentBuilder Builder(Segments);

    sortNestedRegions(Regions);
    ArrayRef<CountedRegion> CombinedRegions = combineRegions(Regions);

    LLVM_DEBUG({
      dbgs() << "Combined regions:\n";
      for (const auto &CR : CombinedRegions)
        dbgs() << "  " << CR.LineStart << ":" << CR.ColumnStart << " -> "
               << CR.LineEnd << ":" << CR.ColumnEnd
               << " (count=" << CR.ExecutionCount << ")\n";
    });

    Builder.buildSegmentsImpl(CombinedRegions);

#ifndef NDEBUG
    for (unsigned I = 1, E = Segments.size(); I < E; ++I) {
      const auto &L = Segments[I - 1];
      const auto &R = Segments[I];
      if (!(L.Line < R.Line) && !(L.Line == R.Line && L.Col < R.Col)) {
        if (L.Line == R.Line && L.Col == R.Col && !L.HasCount)
          continue;
        LLVM_DEBUG(dbgs() << " ! Segment " << L.Line << ":" << L.Col
                          << " followed by " << R.Line << ":" << R.Col << "\n");
        assert(false && "Coverage segments not unique or sorted");
      }
    }
#endif

    return Segments;
  }
};

} // end anonymous namespace

std::vector<StringRef> CoverageMapping::getUniqueSourceFiles() const {
  std::vector<StringRef> Filenames;
  for (const auto &Function : getCoveredFunctions())
    llvm::append_range(Filenames, Function.Filenames);
  llvm::sort(Filenames);
  auto Last = std::unique(Filenames.begin(), Filenames.end());
  Filenames.erase(Last, Filenames.end());
  return Filenames;
}

static SmallBitVector gatherFileIDs(StringRef SourceFile,
                                    const FunctionRecord &Function) {
  SmallBitVector FilenameEquivalence(Function.Filenames.size(), false);
  for (unsigned I = 0, E = Function.Filenames.size(); I < E; ++I)
    if (SourceFile == Function.Filenames[I])
      FilenameEquivalence[I] = true;
  return FilenameEquivalence;
}

/// Return the ID of the file where the definition of the function is located.
static std::optional<unsigned>
findMainViewFileID(const FunctionRecord &Function) {
  SmallBitVector IsNotExpandedFile(Function.Filenames.size(), true);
  for (const auto &CR : Function.CountedRegions)
    if (CR.Kind == CounterMappingRegion::ExpansionRegion)
      IsNotExpandedFile[CR.ExpandedFileID] = false;
  int I = IsNotExpandedFile.find_first();
  if (I == -1)
    return std::nullopt;
  return I;
}

/// Check if SourceFile is the file that contains the definition of
/// the Function. Return the ID of the file in that case or std::nullopt
/// otherwise.
static std::optional<unsigned>
findMainViewFileID(StringRef SourceFile, const FunctionRecord &Function) {
  std::optional<unsigned> I = findMainViewFileID(Function);
  if (I && SourceFile == Function.Filenames[*I])
    return I;
  return std::nullopt;
}

static bool isExpansion(const CountedRegion &R, unsigned FileID) {
  return R.Kind == CounterMappingRegion::ExpansionRegion && R.FileID == FileID;
}

CoverageData CoverageMapping::getCoverageForFile(StringRef Filename) const {
  CoverageData FileCoverage(Filename);
  std::vector<CountedRegion> Regions;

  // Look up the function records in the given file. Due to hash collisions on
  // the filename, we may get back some records that are not in the file.
  ArrayRef<unsigned> RecordIndices =
      getImpreciseRecordIndicesForFilename(Filename);
  for (unsigned RecordIndex : RecordIndices) {
    const FunctionRecord &Function = Functions[RecordIndex];
    auto MainFileID = findMainViewFileID(Filename, Function);
    auto FileIDs = gatherFileIDs(Filename, Function);
    for (const auto &CR : Function.CountedRegions)
      if (FileIDs.test(CR.FileID)) {
        Regions.push_back(CR);
        if (MainFileID && isExpansion(CR, *MainFileID))
          FileCoverage.Expansions.emplace_back(CR, Function);
      }
    // Capture branch regions specific to the function (excluding expansions).
    for (const auto &CR : Function.CountedBranchRegions)
      if (FileIDs.test(CR.FileID) && (CR.FileID == CR.ExpandedFileID))
        FileCoverage.BranchRegions.push_back(CR);
  }

  LLVM_DEBUG(dbgs() << "Emitting segments for file: " << Filename << "\n");
  FileCoverage.Segments = SegmentBuilder::buildSegments(Regions);

  return FileCoverage;
}

std::vector<InstantiationGroup>
CoverageMapping::getInstantiationGroups(StringRef Filename) const {
  FunctionInstantiationSetCollector InstantiationSetCollector;
  // Look up the function records in the given file. Due to hash collisions on
  // the filename, we may get back some records that are not in the file.
  ArrayRef<unsigned> RecordIndices =
      getImpreciseRecordIndicesForFilename(Filename);
  for (unsigned RecordIndex : RecordIndices) {
    const FunctionRecord &Function = Functions[RecordIndex];
    auto MainFileID = findMainViewFileID(Filename, Function);
    if (!MainFileID)
      continue;
    InstantiationSetCollector.insert(Function, *MainFileID);
  }

  std::vector<InstantiationGroup> Result;
  for (auto &InstantiationSet : InstantiationSetCollector) {
    InstantiationGroup IG{InstantiationSet.first.first,
                          InstantiationSet.first.second,
                          std::move(InstantiationSet.second)};
    Result.emplace_back(std::move(IG));
  }
  return Result;
}

CoverageData
CoverageMapping::getCoverageForFunction(const FunctionRecord &Function) const {
  auto MainFileID = findMainViewFileID(Function);
  if (!MainFileID)
    return CoverageData();

  CoverageData FunctionCoverage(Function.Filenames[*MainFileID]);
  std::vector<CountedRegion> Regions;
  for (const auto &CR : Function.CountedRegions)
    if (CR.FileID == *MainFileID) {
      Regions.push_back(CR);
      if (isExpansion(CR, *MainFileID))
        FunctionCoverage.Expansions.emplace_back(CR, Function);
    }
  // Capture branch regions specific to the function (excluding expansions).
  for (const auto &CR : Function.CountedBranchRegions)
    if (CR.FileID == *MainFileID)
      FunctionCoverage.BranchRegions.push_back(CR);

  LLVM_DEBUG(dbgs() << "Emitting segments for function: " << Function.Name
                    << "\n");
  FunctionCoverage.Segments = SegmentBuilder::buildSegments(Regions);

  return FunctionCoverage;
}

CoverageData CoverageMapping::getCoverageForExpansion(
    const ExpansionRecord &Expansion) const {
  CoverageData ExpansionCoverage(
      Expansion.Function.Filenames[Expansion.FileID]);
  std::vector<CountedRegion> Regions;
  for (const auto &CR : Expansion.Function.CountedRegions)
    if (CR.FileID == Expansion.FileID) {
      Regions.push_back(CR);
      if (isExpansion(CR, Expansion.FileID))
        ExpansionCoverage.Expansions.emplace_back(CR, Expansion.Function);
    }
  for (const auto &CR : Expansion.Function.CountedBranchRegions)
    // Capture branch regions that only pertain to the corresponding expansion.
    if (CR.FileID == Expansion.FileID)
      ExpansionCoverage.BranchRegions.push_back(CR);

  LLVM_DEBUG(dbgs() << "Emitting segments for expansion of file "
                    << Expansion.FileID << "\n");
  ExpansionCoverage.Segments = SegmentBuilder::buildSegments(Regions);

  return ExpansionCoverage;
}

LineCoverageStats::LineCoverageStats(
    ArrayRef<const CoverageSegment *> LineSegments,
    const CoverageSegment *WrappedSegment, unsigned Line)
    : ExecutionCount(0), HasMultipleRegions(false), Mapped(false), Line(Line),
      LineSegments(LineSegments), WrappedSegment(WrappedSegment) {
  // Find the minimum number of regions which start in this line.
  unsigned MinRegionCount = 0;
  auto isStartOfRegion = [](const CoverageSegment *S) {
    return !S->IsGapRegion && S->HasCount && S->IsRegionEntry;
  };
  for (unsigned I = 0; I < LineSegments.size() && MinRegionCount < 2; ++I)
    if (isStartOfRegion(LineSegments[I]))
      ++MinRegionCount;

  bool StartOfSkippedRegion = !LineSegments.empty() &&
                              !LineSegments.front()->HasCount &&
                              LineSegments.front()->IsRegionEntry;

  HasMultipleRegions = MinRegionCount > 1;
  Mapped =
      !StartOfSkippedRegion &&
      ((WrappedSegment && WrappedSegment->HasCount) || (MinRegionCount > 0));

  if (!Mapped)
    return;

  // Pick the max count from the non-gap, region entry segments and the
  // wrapped count.
  if (WrappedSegment)
    ExecutionCount = WrappedSegment->Count;
  if (!MinRegionCount)
    return;
  for (const auto *LS : LineSegments)
    if (isStartOfRegion(LS))
      ExecutionCount = std::max(ExecutionCount, LS->Count);
}

LineCoverageIterator &LineCoverageIterator::operator++() {
  if (Next == CD.end()) {
    Stats = LineCoverageStats();
    Ended = true;
    return *this;
  }
  if (Segments.size())
    WrappedSegment = Segments.back();
  Segments.clear();
  while (Next != CD.end() && Next->Line == Line)
    Segments.push_back(&*Next++);
  Stats = LineCoverageStats(Segments, WrappedSegment, Line);
  ++Line;
  return *this;
}

static std::string getCoverageMapErrString(coveragemap_error Err) {
  switch (Err) {
  case coveragemap_error::success:
    return "Success";
  case coveragemap_error::eof:
    return "End of File";
  case coveragemap_error::no_data_found:
    return "No coverage data found";
  case coveragemap_error::unsupported_version:
    return "Unsupported coverage format version";
  case coveragemap_error::truncated:
    return "Truncated coverage data";
  case coveragemap_error::malformed:
    return "Malformed coverage data";
  case coveragemap_error::decompression_failed:
    return "Failed to decompress coverage data (zlib)";
  case coveragemap_error::invalid_or_missing_arch_specifier:
    return "`-arch` specifier is invalid or missing for universal binary";
  }
  llvm_unreachable("A value of coveragemap_error has no message.");
}

namespace {

// FIXME: This class is only here to support the transition to llvm::Error. It
// will be removed once this transition is complete. Clients should prefer to
// deal with the Error value directly, rather than converting to error_code.
class CoverageMappingErrorCategoryType : public std::error_category {
  const char *name() const noexcept override { return "llvm.coveragemap"; }
  std::string message(int IE) const override {
    return getCoverageMapErrString(static_cast<coveragemap_error>(IE));
  }
};

} // end anonymous namespace

std::string CoverageMapError::message() const {
  return getCoverageMapErrString(Err);
}

const std::error_category &llvm::coverage::coveragemap_category() {
  static CoverageMappingErrorCategoryType ErrorCategory;
  return ErrorCategory;
}

char CoverageMapError::ID = 0;
