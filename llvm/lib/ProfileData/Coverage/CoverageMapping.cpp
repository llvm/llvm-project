//===- CoverageMapping.cpp - Code coverage mapping support ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ProfileData/Coverage/CoverageMappingReader.h"
#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <memory>
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

void CounterExpressionBuilder::extractTerms(
    Counter C, int Sign, SmallVectorImpl<std::pair<unsigned, int>> &Terms) {
  switch (C.getKind()) {
  case Counter::Zero:
    break;
  case Counter::CounterValueReference:
    Terms.push_back(std::make_pair(C.getCounterID(), Sign));
    break;
  case Counter::Expression:
    const auto &E = Expressions[C.getExpressionID()];
    extractTerms(E.LHS, Sign, Terms);
    extractTerms(E.RHS, E.Kind == CounterExpression::Subtract ? -Sign : Sign,
                 Terms);
    break;
  }
}

Counter CounterExpressionBuilder::simplify(Counter ExpressionTree) {
  // Gather constant terms.
  SmallVector<std::pair<unsigned, int>, 32> Terms;
  extractTerms(ExpressionTree, +1, Terms);

  // If there are no terms, this is just a zero. The algorithm below assumes at
  // least one term.
  if (Terms.size() == 0)
    return Counter::getZero();

  // Group the terms by counter ID.
  std::sort(Terms.begin(), Terms.end(),
            [](const std::pair<unsigned, int> &LHS,
               const std::pair<unsigned, int> &RHS) {
    return LHS.first < RHS.first;
  });

  // Combine terms by counter ID to eliminate counters that sum to zero.
  auto Prev = Terms.begin();
  for (auto I = Prev + 1, E = Terms.end(); I != E; ++I) {
    if (I->first == Prev->first) {
      Prev->second += I->second;
      continue;
    }
    ++Prev;
    *Prev = *I;
  }
  Terms.erase(++Prev, Terms.end());

  Counter C;
  // Create additions. We do this before subtractions to avoid constructs like
  // ((0 - X) + Y), as opposed to (Y - X).
  for (auto Term : Terms) {
    if (Term.second <= 0)
      continue;
    for (int I = 0; I < Term.second; ++I)
      if (C.isZero())
        C = Counter::getCounter(Term.first);
      else
        C = get(CounterExpression(CounterExpression::Add, C,
                                  Counter::getCounter(Term.first)));
  }

  // Create subtractions.
  for (auto Term : Terms) {
    if (Term.second >= 0)
      continue;
    for (int I = 0; I < -Term.second; ++I)
      C = get(CounterExpression(CounterExpression::Subtract, C,
                                Counter::getCounter(Term.first)));
  }
  return C;
}

Counter CounterExpressionBuilder::add(Counter LHS, Counter RHS) {
  return simplify(get(CounterExpression(CounterExpression::Add, LHS, RHS)));
}

Counter CounterExpressionBuilder::subtract(Counter LHS, Counter RHS) {
  return simplify(
      get(CounterExpression(CounterExpression::Subtract, LHS, RHS)));
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

void FunctionRecordIterator::skipOtherFiles() {
  while (Current != Records.end() && !Filename.empty() &&
         Filename != Current->Filenames[0])
    ++Current;
  if (Current == Records.end())
    *this = FunctionRecordIterator();
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

  // Don't load records for functions we've already seen.
  if (!FunctionNames.insert(OrigFuncName).second)
    return Error::success();

  CounterMappingContext Ctx(Record.Expressions);

  std::vector<uint64_t> Counts;
  if (Error E = ProfileReader.getFunctionCounts(Record.FunctionName,
                                                Record.FunctionHash, Counts)) {
    instrprof_error IPE = InstrProfError::take(std::move(E));
    if (IPE == instrprof_error::hash_mismatch) {
      MismatchedFunctionCount++;
      return Error::success();
    } else if (IPE != instrprof_error::unknown_function)
      return make_error<InstrProfError>(IPE);
    Counts.assign(Record.MappingRegions.size(), 0);
  }
  Ctx.setCounts(Counts);

  assert(!Record.MappingRegions.empty() && "Function has no regions");

  FunctionRecord Function(OrigFuncName, Record.Filenames);
  for (const auto &Region : Record.MappingRegions) {
    Expected<int64_t> ExecutionCount = Ctx.evaluate(Region.Count);
    if (auto E = ExecutionCount.takeError()) {
      consumeError(std::move(E));
      return Error::success();
    }
    Function.pushRegion(Region, *ExecutionCount);
  }
  if (Function.CountedRegions.size() != Record.MappingRegions.size()) {
    MismatchedFunctionCount++;
    return Error::success();
  }

  Functions.push_back(std::move(Function));
  return Error::success();
}

Expected<std::unique_ptr<CoverageMapping>>
CoverageMapping::load(CoverageMappingReader &CoverageReader,
                      IndexedInstrProfReader &ProfileReader) {
  auto Coverage = std::unique_ptr<CoverageMapping>(new CoverageMapping());

  for (const auto &Record : CoverageReader)
    if (Error E = Coverage->loadFunctionRecord(Record, ProfileReader))
      return std::move(E);

  return std::move(Coverage);
}

Expected<std::unique_ptr<CoverageMapping>> CoverageMapping::load(
    ArrayRef<std::unique_ptr<CoverageMappingReader>> CoverageReaders,
    IndexedInstrProfReader &ProfileReader) {
  auto Coverage = std::unique_ptr<CoverageMapping>(new CoverageMapping());

  for (const auto &CoverageReader : CoverageReaders)
    for (const auto &Record : *CoverageReader)
      if (Error E = Coverage->loadFunctionRecord(Record, ProfileReader))
        return std::move(E);

  return std::move(Coverage);
}

Expected<std::unique_ptr<CoverageMapping>>
CoverageMapping::load(ArrayRef<StringRef> ObjectFilenames,
                      StringRef ProfileFilename, StringRef Arch) {
  auto ProfileReaderOrErr = IndexedInstrProfReader::create(ProfileFilename);
  if (Error E = ProfileReaderOrErr.takeError())
    return std::move(E);
  auto ProfileReader = std::move(ProfileReaderOrErr.get());

  SmallVector<std::unique_ptr<CoverageMappingReader>, 4> Readers;
  SmallVector<std::unique_ptr<MemoryBuffer>, 4> Buffers;
  for (StringRef ObjectFilename : ObjectFilenames) {
    auto CovMappingBufOrErr = MemoryBuffer::getFileOrSTDIN(ObjectFilename);
    if (std::error_code EC = CovMappingBufOrErr.getError())
      return errorCodeToError(EC);
    auto CoverageReaderOrErr =
        BinaryCoverageReader::create(CovMappingBufOrErr.get(), Arch);
    if (Error E = CoverageReaderOrErr.takeError())
      return std::move(E);
    Readers.push_back(std::move(CoverageReaderOrErr.get()));
    Buffers.push_back(std::move(CovMappingBufOrErr.get()));
  }
  return load(Readers, *ProfileReader);
}

namespace {

/// \brief Distributes functions into instantiation sets.
///
/// An instantiation set is a collection of functions that have the same source
/// code, ie, template functions specializations.
class FunctionInstantiationSetCollector {
  typedef DenseMap<std::pair<unsigned, unsigned>,
                   std::vector<const FunctionRecord *>> MapT;
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

  /// Start a segment with no count specified.
  void startSegment(unsigned Line, unsigned Col) {
    DEBUG(dbgs() << "Top level segment at " << Line << ":" << Col << "\n");
    Segments.emplace_back(Line, Col, /*IsRegionEntry=*/false);
  }

  /// Start a segment with the given Region's count.
  void startSegment(unsigned Line, unsigned Col, bool IsRegionEntry,
                    const CountedRegion &Region) {
    // Avoid creating empty regions.
    if (!Segments.empty() && Segments.back().Line == Line &&
        Segments.back().Col == Col)
      Segments.pop_back();
    DEBUG(dbgs() << "Segment at " << Line << ":" << Col);
    // Set this region's count.
    if (Region.Kind != CounterMappingRegion::SkippedRegion) {
      DEBUG(dbgs() << " with count " << Region.ExecutionCount);
      Segments.emplace_back(Line, Col, Region.ExecutionCount, IsRegionEntry);
    } else
      Segments.emplace_back(Line, Col, IsRegionEntry);
    DEBUG(dbgs() << "\n");
  }

  /// Start a segment for the given region.
  void startSegment(const CountedRegion &Region) {
    startSegment(Region.LineStart, Region.ColumnStart, true, Region);
  }

  /// Pop the top region off of the active stack, starting a new segment with
  /// the containing Region's count.
  void popRegion() {
    const CountedRegion *Active = ActiveRegions.back();
    unsigned Line = Active->LineEnd, Col = Active->ColumnEnd;
    ActiveRegions.pop_back();
    if (ActiveRegions.empty())
      startSegment(Line, Col);
    else
      startSegment(Line, Col, false, *ActiveRegions.back());
  }

  void buildSegmentsImpl(ArrayRef<CountedRegion> Regions) {
    for (const auto &Region : Regions) {
      // Pop any regions that end before this one starts.
      while (!ActiveRegions.empty() &&
             ActiveRegions.back()->endLoc() <= Region.startLoc())
        popRegion();
      // Add this region to the stack.
      ActiveRegions.push_back(&Region);
      startSegment(Region);
    }
    // Pop any regions that are left in the stack.
    while (!ActiveRegions.empty())
      popRegion();
  }

  /// Sort a nested sequence of regions from a single file.
  static void sortNestedRegions(MutableArrayRef<CountedRegion> Regions) {
    std::sort(Regions.begin(), Regions.end(), [](const CountedRegion &LHS,
                                                 const CountedRegion &RHS) {
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
  /// Build a list of CoverageSegments from a list of Regions.
  static std::vector<CoverageSegment>
  buildSegments(MutableArrayRef<CountedRegion> Regions) {
    std::vector<CoverageSegment> Segments;
    SegmentBuilder Builder(Segments);

    sortNestedRegions(Regions);
    ArrayRef<CountedRegion> CombinedRegions = combineRegions(Regions);

    Builder.buildSegmentsImpl(CombinedRegions);
    return Segments;
  }
};

} // end anonymous namespace

std::vector<StringRef> CoverageMapping::getUniqueSourceFiles() const {
  std::vector<StringRef> Filenames;
  for (const auto &Function : getCoveredFunctions())
    Filenames.insert(Filenames.end(), Function.Filenames.begin(),
                     Function.Filenames.end());
  std::sort(Filenames.begin(), Filenames.end());
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
static Optional<unsigned> findMainViewFileID(const FunctionRecord &Function) {
  SmallBitVector IsNotExpandedFile(Function.Filenames.size(), true);
  for (const auto &CR : Function.CountedRegions)
    if (CR.Kind == CounterMappingRegion::ExpansionRegion)
      IsNotExpandedFile[CR.ExpandedFileID] = false;
  int I = IsNotExpandedFile.find_first();
  if (I == -1)
    return None;
  return I;
}

/// Check if SourceFile is the file that contains the definition of
/// the Function. Return the ID of the file in that case or None otherwise.
static Optional<unsigned> findMainViewFileID(StringRef SourceFile,
                                             const FunctionRecord &Function) {
  Optional<unsigned> I = findMainViewFileID(Function);
  if (I && SourceFile == Function.Filenames[*I])
    return I;
  return None;
}

static bool isExpansion(const CountedRegion &R, unsigned FileID) {
  return R.Kind == CounterMappingRegion::ExpansionRegion && R.FileID == FileID;
}

CoverageData CoverageMapping::getCoverageForFile(StringRef Filename) const {
  CoverageData FileCoverage(Filename);
  std::vector<CountedRegion> Regions;

  for (const auto &Function : Functions) {
    auto MainFileID = findMainViewFileID(Filename, Function);
    auto FileIDs = gatherFileIDs(Filename, Function);
    for (const auto &CR : Function.CountedRegions)
      if (FileIDs.test(CR.FileID)) {
        Regions.push_back(CR);
        if (MainFileID && isExpansion(CR, *MainFileID))
          FileCoverage.Expansions.emplace_back(CR, Function);
      }
  }

  DEBUG(dbgs() << "Emitting segments for file: " << Filename << "\n");
  FileCoverage.Segments = SegmentBuilder::buildSegments(Regions);

  return FileCoverage;
}

std::vector<const FunctionRecord *>
CoverageMapping::getInstantiations(StringRef Filename) const {
  FunctionInstantiationSetCollector InstantiationSetCollector;
  for (const auto &Function : Functions) {
    auto MainFileID = findMainViewFileID(Filename, Function);
    if (!MainFileID)
      continue;
    InstantiationSetCollector.insert(Function, *MainFileID);
  }

  std::vector<const FunctionRecord *> Result;
  for (const auto &InstantiationSet : InstantiationSetCollector) {
    if (InstantiationSet.second.size() < 2)
      continue;
    Result.insert(Result.end(), InstantiationSet.second.begin(),
                  InstantiationSet.second.end());
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

  DEBUG(dbgs() << "Emitting segments for function: " << Function.Name << "\n");
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

  DEBUG(dbgs() << "Emitting segments for expansion of file " << Expansion.FileID
               << "\n");
  ExpansionCoverage.Segments = SegmentBuilder::buildSegments(Regions);

  return ExpansionCoverage;
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

static ManagedStatic<CoverageMappingErrorCategoryType> ErrorCategory;

const std::error_category &llvm::coverage::coveragemap_category() {
  return *ErrorCategory;
}

char CoverageMapError::ID = 0;
