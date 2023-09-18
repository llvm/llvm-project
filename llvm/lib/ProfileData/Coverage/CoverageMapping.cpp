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
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/BuildID.h"
#include "llvm/ProfileData/Coverage/CoverageMappingReader.h"
#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cmath>
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

Expected<BitVector> CounterMappingContext::evaluateBitmap(
    const CounterMappingRegion *MCDCDecision) const {
  unsigned ID = MCDCDecision->MCDCParams.BitmapIdx;
  unsigned NC = MCDCDecision->MCDCParams.NumConditions;
  unsigned SizeInBits = llvm::alignTo(1L << NC, CHAR_BIT);
  unsigned SizeInBytes = SizeInBits / CHAR_BIT;

  ArrayRef<uint8_t> Bytes(&BitmapBytes[ID], SizeInBytes);

  // Mask each bitmap byte into the BitVector. Go in reverse so that the
  // bitvector can just be shifted over by one byte on each iteration.
  BitVector Result(SizeInBits, false);
  for (auto Byte = std::rbegin(Bytes); Byte != std::rend(Bytes); ++Byte) {
    uint32_t Data = *Byte;
    Result <<= CHAR_BIT;
    Result.setBitsInMask(&Data, 1);
  }
  return Result;
}

class MCDCRecordProcessor {
  /// A bitmap representing the executed test vectors for a boolean expression.
  /// Each index of the bitmap corresponds to a possible test vector. An index
  /// with a bit value of '1' indicates that the corresponding Test Vector
  /// identified by that index was executed.
  BitVector &ExecutedTestVectorBitmap;

  /// Decision Region to which the ExecutedTestVectorBitmap applies.
  CounterMappingRegion &Region;

  /// Array of branch regions corresponding each conditions in the boolean
  /// expression.
  ArrayRef<CounterMappingRegion> Branches;

  /// Total number of conditions in the boolean expression.
  unsigned NumConditions;

  /// Mapping of a condition ID to its corresponding branch region.
  llvm::DenseMap<unsigned, const CounterMappingRegion *> Map;

  /// Vector used to track whether a condition is constant folded.
  MCDCRecord::BoolVector Folded;

  /// Mapping of calculated MC/DC Independence Pairs for each condition.
  MCDCRecord::TVPairMap IndependencePairs;

  /// Total number of possible Test Vectors for the boolean expression.
  MCDCRecord::TestVectors TestVectors;

  /// Actual executed Test Vectors for the boolean expression, based on
  /// ExecutedTestVectorBitmap.
  MCDCRecord::TestVectors ExecVectors;

public:
  MCDCRecordProcessor(BitVector &Bitmap, CounterMappingRegion &Region,
                      ArrayRef<CounterMappingRegion> Branches)
      : ExecutedTestVectorBitmap(Bitmap), Region(Region), Branches(Branches),
        NumConditions(Region.MCDCParams.NumConditions),
        Folded(NumConditions, false), IndependencePairs(NumConditions),
        TestVectors(pow(2, NumConditions)) {}

private:
  void recordTestVector(MCDCRecord::TestVector &TV,
                        MCDCRecord::CondState Result) {
    // Calculate an index that is used to identify the test vector in a vector
    // of test vectors.  This index also corresponds to the index values of an
    // MCDC Region's bitmap (see findExecutedTestVectors()).
    unsigned Index = 0;
    for (auto Cond = std::rbegin(TV); Cond != std::rend(TV); ++Cond) {
      Index <<= 1;
      Index |= (*Cond == MCDCRecord::MCDC_True) ? 0x1 : 0x0;
    }

    // Copy the completed test vector to the vector of testvectors.
    TestVectors[Index] = TV;

    // The final value (T,F) is equal to the last non-dontcare state on the
    // path (in a short-circuiting system).
    TestVectors[Index].push_back(Result);
  }

  void shouldCopyOffTestVectorForTruePath(MCDCRecord::TestVector &TV,
                                          unsigned ID) {
    // Branch regions are hashed based on an ID.
    const CounterMappingRegion *Branch = Map[ID];

    TV[ID - 1] = MCDCRecord::MCDC_True;
    if (Branch->MCDCParams.TrueID > 0)
      buildTestVector(TV, Branch->MCDCParams.TrueID);
    else
      recordTestVector(TV, MCDCRecord::MCDC_True);
  }

  void shouldCopyOffTestVectorForFalsePath(MCDCRecord::TestVector &TV,
                                           unsigned ID) {
    // Branch regions are hashed based on an ID.
    const CounterMappingRegion *Branch = Map[ID];

    TV[ID - 1] = MCDCRecord::MCDC_False;
    if (Branch->MCDCParams.FalseID > 0)
      buildTestVector(TV, Branch->MCDCParams.FalseID);
    else
      recordTestVector(TV, MCDCRecord::MCDC_False);
  }

  void buildTestVector(MCDCRecord::TestVector &TV, unsigned ID = 1) {
    shouldCopyOffTestVectorForTruePath(TV, ID);
    shouldCopyOffTestVectorForFalsePath(TV, ID);

    // Reset back to DontCare.
    TV[ID - 1] = MCDCRecord::MCDC_DontCare;
  }

  void findExecutedTestVectors(BitVector &ExecutedTestVectorBitmap) {
    // Walk the bits in the bitmap.  A bit set to '1' indicates that the test
    // vector at the corresponding index was executed during a test run.
    for (unsigned Idx = 0; Idx < ExecutedTestVectorBitmap.size(); Idx++) {
      if (ExecutedTestVectorBitmap[Idx] == 0)
        continue;
      assert(!TestVectors[Idx].empty() && "Test Vector doesn't exist.");
      ExecVectors.push_back(TestVectors[Idx]);
    }
  }

  // For a given condition and two executed Test Vectors, A and B, see if the
  // two test vectors match forming an Independence Pair for the condition.
  // For two test vectors to match, the following must be satisfied:
  // - The condition's value in each test vector must be opposite.
  // - The result's value in each test vector must be opposite.
  // - All other conditions' values must be equal or marked as "don't care".
  bool matchTestVectors(unsigned Aidx, unsigned Bidx, unsigned ConditionIdx) {
    const MCDCRecord::TestVector &A = ExecVectors[Aidx];
    const MCDCRecord::TestVector &B = ExecVectors[Bidx];

    // If condition values in both A and B aren't opposites, no match.
    if (!((A[ConditionIdx] ^ B[ConditionIdx]) == 1))
      return false;

    // If the results of both A and B aren't opposites, no match.
    if (!((A[NumConditions] ^ B[NumConditions]) == 1))
      return false;

    for (unsigned Idx = 0; Idx < NumConditions; Idx++) {
      // Look for other conditions that don't match. Skip over the given
      // Condition as well as any conditions marked as "don't care".
      const auto ARecordTyForCond = A[Idx];
      const auto BRecordTyForCond = B[Idx];
      if (Idx == ConditionIdx ||
          ARecordTyForCond == MCDCRecord::MCDC_DontCare ||
          BRecordTyForCond == MCDCRecord::MCDC_DontCare)
        continue;

      // If there is a condition mismatch with any of the other conditions,
      // there is no match for the test vectors.
      if (ARecordTyForCond != BRecordTyForCond)
        return false;
    }

    // Otherwise, match.
    return true;
  }

  // Find all possible Independence Pairs for a boolean expression given its
  // executed Test Vectors.  This process involves looking at each condition
  // and attempting to find two Test Vectors that "match", giving us a pair.
  void findIndependencePairs() {
    unsigned NumTVs = ExecVectors.size();

    // For each condition.
    for (unsigned C = 0; C < NumConditions; C++) {
      bool PairFound = false;

      // For each executed test vector.
      for (unsigned I = 0; !PairFound && I < NumTVs; I++) {

        // Compared to every other executed test vector.
        for (unsigned J = 0; !PairFound && J < NumTVs; J++) {
          if (I == J)
            continue;

          // If a matching pair of vectors is found, record them.
          if ((PairFound = matchTestVectors(I, J, C)))
            IndependencePairs[C] = std::make_pair(I + 1, J + 1);
        }
      }
    }
  }

public:
  /// Process the MC/DC Record in order to produce a result for a boolean
  /// expression. This process includes tracking the conditions that comprise
  /// the decision region, calculating the list of all possible test vectors,
  /// marking the executed test vectors, and then finding an Independence Pair
  /// out of the executed test vectors for each condition in the boolean
  /// expression. A condition is tracked to ensure that its ID can be mapped to
  /// its ordinal position in the boolean expression. The condition's source
  /// location is also tracked, as well as whether it is constant folded (in
  /// which case it is excuded from the metric).
  MCDCRecord processMCDCRecord() {
    unsigned I = 0;
    MCDCRecord::CondIDMap PosToID;
    MCDCRecord::LineColPairMap CondLoc;

    // Walk the Record's BranchRegions (representing Conditions) in order to:
    // - Hash the condition based on its corresponding ID. This will be used to
    //   calculate the test vectors.
    // - Keep a map of the condition's ordinal position (1, 2, 3, 4) to its
    //   actual ID.  This will be used to visualize the conditions in the
    //   correct order.
    // - Keep track of the condition source location. This will be used to
    //   visualize where the condition is.
    // - Record whether the condition is constant folded so that we exclude it
    //   from being measured.
    for (const auto &B : Branches) {
      Map[B.MCDCParams.ID] = &B;
      PosToID[I] = B.MCDCParams.ID - 1;
      CondLoc[I] = B.startLoc();
      Folded[I++] = (B.Count.isZero() && B.FalseCount.isZero());
    }

    // Initialize a base test vector as 'DontCare'.
    MCDCRecord::TestVector TV(NumConditions, MCDCRecord::MCDC_DontCare);

    // Use the base test vector to build the list of all possible test vectors.
    buildTestVector(TV);

    // Using Profile Bitmap from runtime, mark the executed test vectors.
    findExecutedTestVectors(ExecutedTestVectorBitmap);

    // Compare executed test vectors against each other to find an independence
    // pairs for each condition.  This processing takes the most time.
    findIndependencePairs();

    // Record Test vectors, executed vectors, and independence pairs.
    MCDCRecord Res(Region, ExecVectors, IndependencePairs, Folded, PosToID,
                   CondLoc);
    return Res;
  }
};

Expected<MCDCRecord> CounterMappingContext::evaluateMCDCRegion(
    CounterMappingRegion Region, BitVector ExecutedTestVectorBitmap,
    ArrayRef<CounterMappingRegion> Branches) {

  MCDCRecordProcessor MCDCProcessor(ExecutedTestVectorBitmap, Region, Branches);
  return MCDCProcessor.processMCDCRecord();
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

static unsigned getMaxBitmapSize(const CounterMappingContext &Ctx,
                                 const CoverageMappingRecord &Record) {
  unsigned MaxBitmapID = 0;
  unsigned NumConditions = 0;
  // The last DecisionRegion has the highest bitmap byte index used in the
  // function, which when combined with its number of conditions, yields the
  // full bitmap size.
  for (const auto &Region : reverse(Record.MappingRegions)) {
    if (Region.Kind == CounterMappingRegion::MCDCDecisionRegion) {
      MaxBitmapID = Region.MCDCParams.BitmapIdx;
      NumConditions = Region.MCDCParams.NumConditions;
      break;
    }
  }
  unsigned SizeInBits = llvm::alignTo(1L << NumConditions, CHAR_BIT);
  return MaxBitmapID + (SizeInBits / CHAR_BIT);
}

Error CoverageMapping::loadFunctionRecord(
    const CoverageMappingRecord &Record,
    IndexedInstrProfReader &ProfileReader) {
  StringRef OrigFuncName = Record.FunctionName;
  if (OrigFuncName.empty())
    return make_error<CoverageMapError>(coveragemap_error::malformed,
                                        "record function name is empty");

  if (Record.Filenames.empty())
    OrigFuncName = getFuncNameWithoutPrefix(OrigFuncName);
  else
    OrigFuncName = getFuncNameWithoutPrefix(OrigFuncName, Record.Filenames[0]);

  CounterMappingContext Ctx(Record.Expressions);

  std::vector<uint64_t> Counts;
  if (Error E = ProfileReader.getFunctionCounts(Record.FunctionName,
                                                Record.FunctionHash, Counts)) {
    instrprof_error IPE = std::get<0>(InstrProfError::take(std::move(E)));
    if (IPE == instrprof_error::hash_mismatch) {
      FuncHashMismatches.emplace_back(std::string(Record.FunctionName),
                                      Record.FunctionHash);
      return Error::success();
    }
    if (IPE != instrprof_error::unknown_function)
      return make_error<InstrProfError>(IPE);
    Counts.assign(getMaxCounterID(Ctx, Record) + 1, 0);
  }
  Ctx.setCounts(Counts);

  std::vector<uint8_t> BitmapBytes;
  if (Error E = ProfileReader.getFunctionBitmapBytes(
          Record.FunctionName, Record.FunctionHash, BitmapBytes)) {
    instrprof_error IPE = std::get<0>(InstrProfError::take(std::move(E)));
    if (IPE == instrprof_error::hash_mismatch) {
      FuncHashMismatches.emplace_back(std::string(Record.FunctionName),
                                      Record.FunctionHash);
      return Error::success();
    }
    if (IPE != instrprof_error::unknown_function)
      return make_error<InstrProfError>(IPE);
    BitmapBytes.assign(getMaxBitmapSize(Ctx, Record) + 1, 0);
  }
  Ctx.setBitmapBytes(BitmapBytes);

  assert(!Record.MappingRegions.empty() && "Function has no regions");

  // This coverage record is a zero region for a function that's unused in
  // some TU, but used in a different TU. Ignore it. The coverage maps from the
  // the other TU will either be loaded (providing full region counts) or they
  // won't (in which case we don't unintuitively report functions as uncovered
  // when they have non-zero counts in the profile).
  if (Record.MappingRegions.size() == 1 &&
      Record.MappingRegions[0].Count.isZero() && Counts[0] > 0)
    return Error::success();

  unsigned NumConds = 0;
  const CounterMappingRegion *MCDCDecision;
  std::vector<CounterMappingRegion> MCDCBranches;

  FunctionRecord Function(OrigFuncName, Record.Filenames);
  for (const auto &Region : Record.MappingRegions) {
    // If an MCDCDecisionRegion is seen, track the BranchRegions that follow
    // it according to Region.NumConditions.
    if (Region.Kind == CounterMappingRegion::MCDCDecisionRegion) {
      assert(NumConds == 0);
      MCDCDecision = &Region;
      NumConds = Region.MCDCParams.NumConditions;
      continue;
    }
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

    // If a MCDCDecisionRegion was seen, store the BranchRegions that
    // correspond to it in a vector, according to the number of conditions
    // recorded for the region (tracked by NumConds).
    if (NumConds > 0 && Region.Kind == CounterMappingRegion::MCDCBranchRegion) {
      MCDCBranches.push_back(Region);

      // As we move through all of the MCDCBranchRegions that follow the
      // MCDCDecisionRegion, decrement NumConds to make sure we account for
      // them all before we calculate the bitmap of executed test vectors.
      if (--NumConds == 0) {
        // Evaluating the test vector bitmap for the decision region entails
        // calculating precisely what bits are pertinent to this region alone.
        // This is calculated based on the recorded offset into the global
        // profile bitmap; the length is calculated based on the recorded
        // number of conditions.
        Expected<BitVector> ExecutedTestVectorBitmap =
            Ctx.evaluateBitmap(MCDCDecision);
        if (auto E = ExecutedTestVectorBitmap.takeError()) {
          consumeError(std::move(E));
          return Error::success();
        }

        // Since the bitmap identifies the executed test vectors for an MC/DC
        // DecisionRegion, all of the information is now available to process.
        // This is where the bulk of the MC/DC progressing takes place.
        Expected<MCDCRecord> Record = Ctx.evaluateMCDCRegion(
            *MCDCDecision, *ExecutedTestVectorBitmap, MCDCBranches);
        if (auto E = Record.takeError()) {
          consumeError(std::move(E));
          return Error::success();
        }

        // Save the MC/DC Record so that it can be visualized later.
        Function.pushMCDCRecord(*Record);
        MCDCBranches.clear();
      }
    }
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
        return make_error<CoverageMapError>(CME.get(), CME.getMessage());
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
  InstrProfSymtab &ProfSymTab = ProfileReader.getSymtab();

  SmallVector<object::BuildIDRef> BinaryIDs;
  auto CoverageReadersOrErr = BinaryCoverageReader::create(
      CovMappingBufRef, Arch, Buffers, ProfSymTab,
      CompilationDir, FoundBinaryIDs ? &BinaryIDs : nullptr);
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

Expected<std::unique_ptr<CoverageMapping>> CoverageMapping::load(
    ArrayRef<StringRef> ObjectFilenames, StringRef ProfileFilename,
    vfs::FileSystem &FS, ArrayRef<StringRef> Arches, StringRef CompilationDir,
    const object::BuildIDFetcher *BIDFetcher, bool CheckBinaryIDs) {
  auto ProfileReaderOrErr = IndexedInstrProfReader::create(ProfileFilename, FS);
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
    std::vector<object::BuildID> ProfileBinaryIDs;
    if (Error E = ProfileReader->readBinaryIds(ProfileBinaryIDs))
      return createFileError(ProfileFilename, std::move(E));

    SmallVector<object::BuildIDRef> BinaryIDsToFetch;
    if (!ProfileBinaryIDs.empty()) {
      const auto &Compare = [](object::BuildIDRef A, object::BuildIDRef B) {
        return std::lexicographical_compare(A.begin(), A.end(), B.begin(),
                                            B.end());
      };
      llvm::sort(FoundBinaryIDs, Compare);
      std::set_difference(
          ProfileBinaryIDs.begin(), ProfileBinaryIDs.end(),
          FoundBinaryIDs.begin(), FoundBinaryIDs.end(),
          std::inserter(BinaryIDsToFetch, BinaryIDsToFetch.end()), Compare);
    }

    for (object::BuildIDRef BinaryID : BinaryIDsToFetch) {
      std::optional<std::string> PathOpt = BIDFetcher->fetch(BinaryID);
      if (PathOpt) {
        std::string Path = std::move(*PathOpt);
        StringRef Arch = Arches.size() == 1 ? Arches.front() : StringRef();
        if (Error E = loadFromFile(Path, Arch, CompilationDir, *ProfileReader,
                                  *Coverage, DataFound))
          return std::move(E);
      } else if (CheckBinaryIDs) {
        return createFileError(
            ProfileFilename,
            createStringError(errc::no_such_file_or_directory,
                              "Missing binary ID: " +
                                  llvm::toHex(BinaryID, /*LowerCase=*/true)));
      }
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
    // Capture MCDC records specific to the function.
    for (const auto &MR : Function.MCDCRecords)
      if (FileIDs.test(MR.getDecisionRegion().FileID))
        FileCoverage.MCDCRecords.push_back(MR);
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

  // Capture MCDC records specific to the function.
  for (const auto &MR : Function.MCDCRecords)
    if (MR.getDecisionRegion().FileID == *MainFileID)
      FunctionCoverage.MCDCRecords.push_back(MR);

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

static std::string getCoverageMapErrString(coveragemap_error Err,
                                           const std::string &ErrMsg = "") {
  std::string Msg;
  raw_string_ostream OS(Msg);

  switch ((uint32_t)Err) {
  case (uint32_t)coveragemap_error::success:
    OS << "success";
    break;
  case (uint32_t)coveragemap_error::eof:
    OS << "end of File";
    break;
  case (uint32_t)coveragemap_error::no_data_found:
    OS << "no coverage data found";
    break;
  case (uint32_t)coveragemap_error::unsupported_version:
    OS << "unsupported coverage format version";
    break;
  case (uint32_t)coveragemap_error::truncated:
    OS << "truncated coverage data";
    break;
  case (uint32_t)coveragemap_error::malformed:
    OS << "malformed coverage data";
    break;
  case (uint32_t)coveragemap_error::decompression_failed:
    OS << "failed to decompress coverage data (zlib)";
    break;
  case (uint32_t)coveragemap_error::invalid_or_missing_arch_specifier:
    OS << "`-arch` specifier is invalid or missing for universal binary";
    break;
  default:
    llvm_unreachable("invalid coverage mapping error.");
  }

  // If optional error message is not empty, append it to the message.
  if (!ErrMsg.empty())
    OS << ": " << ErrMsg;

  return Msg;
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
  return getCoverageMapErrString(Err, Msg);
}

const std::error_category &llvm::coverage::coveragemap_category() {
  static CoverageMappingErrorCategoryType ErrorCategory;
  return ErrorCategory;
}

char CoverageMapError::ID = 0;
