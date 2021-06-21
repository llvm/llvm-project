//===- IRSimilarityIdentifier.cpp - Find similarity in a module -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// Implementation file for the IRSimilarityIdentifier for identifying
// similarities in IR including the IRInstructionMapper.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/IRSimilarityIdentifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/User.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/SuffixTree.h"

using namespace llvm;
using namespace IRSimilarity;

IRInstructionData::IRInstructionData(Instruction &I, bool Legality,
                                     IRInstructionDataList &IDList)
    : Inst(&I), Legal(Legality), IDL(&IDList) {
  // We check for whether we have a comparison instruction.  If it is, we
  // find the "less than" version of the predicate for consistency for
  // comparison instructions throught the program.
  if (CmpInst *C = dyn_cast<CmpInst>(&I)) {
    CmpInst::Predicate Predicate = predicateForConsistency(C);
    if (Predicate != C->getPredicate())
      RevisedPredicate = Predicate;
  }

  // Here we collect the operands and their types for determining whether
  // the structure of the operand use matches between two different candidates.
  for (Use &OI : I.operands()) {
    if (isa<CmpInst>(I) && RevisedPredicate.hasValue()) {
      // If we have a CmpInst where the predicate is reversed, it means the
      // operands must be reversed as well.
      OperVals.insert(OperVals.begin(), OI.get());
      continue;
    }

    OperVals.push_back(OI.get());
  }
}

CmpInst::Predicate IRInstructionData::predicateForConsistency(CmpInst *CI) {
  switch (CI->getPredicate()) {
  case CmpInst::FCMP_OGT:
  case CmpInst::FCMP_UGT:
  case CmpInst::FCMP_OGE:
  case CmpInst::FCMP_UGE:
  case CmpInst::ICMP_SGT:
  case CmpInst::ICMP_UGT:
  case CmpInst::ICMP_SGE:
  case CmpInst::ICMP_UGE:
    return CI->getSwappedPredicate();
  default:
    return CI->getPredicate();
  }
}

CmpInst::Predicate IRInstructionData::getPredicate() const {
  assert(isa<CmpInst>(Inst) &&
         "Can only get a predicate from a compare instruction");

  if (RevisedPredicate.hasValue())
    return RevisedPredicate.getValue();
  
  return cast<CmpInst>(Inst)->getPredicate();
}

static StringRef getCalledFunctionName(CallInst &CI) {
  assert(CI.getCalledFunction() != nullptr && "Called Function is nullptr?");

  return CI.getCalledFunction()->getName();
}

bool IRSimilarity::isClose(const IRInstructionData &A,
                           const IRInstructionData &B) {

  if (!A.Legal || !B.Legal)
    return false;

  // Check if we are performing the same sort of operation on the same types
  // but not on the same values.
  if (!A.Inst->isSameOperationAs(B.Inst)) {
    // If there is a predicate, this means that either there is a swapped
    // predicate, or that the types are different, we want to make sure that
    // the predicates are equivalent via swapping.
    if (isa<CmpInst>(A.Inst) && isa<CmpInst>(B.Inst)) {

      if (A.getPredicate() != B.getPredicate())
        return false;

      // If the predicates are the same via swap, make sure that the types are
      // still the same.
      auto ZippedTypes = zip(A.OperVals, B.OperVals);

      return all_of(
          ZippedTypes, [](std::tuple<llvm::Value *, llvm::Value *> R) {
            return std::get<0>(R)->getType() == std::get<1>(R)->getType();
          });
    }

    return false;
  }

  // Since any GEP Instruction operands after the first operand cannot be
  // defined by a register, we must make sure that the operands after the first
  // are the same in the two instructions
  if (auto *GEP = dyn_cast<GetElementPtrInst>(A.Inst)) {
    auto *OtherGEP = cast<GetElementPtrInst>(B.Inst);

    // If the instructions do not have the same inbounds restrictions, we do
    // not consider them the same.
    if (GEP->isInBounds() != OtherGEP->isInBounds())
      return false;

    auto ZippedOperands = zip(GEP->indices(), OtherGEP->indices());

    // We increment here since we do not care about the first instruction,
    // we only care about the following operands since they must be the
    // exact same to be considered similar.
    return all_of(drop_begin(ZippedOperands),
                  [](std::tuple<llvm::Use &, llvm::Use &> R) {
                    return std::get<0>(R) == std::get<1>(R);
                  });
  }

  // If the instructions are functions, we make sure that the function name is
  // the same.  We already know that the types are since is isSameOperationAs is
  // true.
  if (isa<CallInst>(A.Inst) && isa<CallInst>(B.Inst)) {
    CallInst *CIA = cast<CallInst>(A.Inst);
    CallInst *CIB = cast<CallInst>(B.Inst);
    if (getCalledFunctionName(*CIA).compare(getCalledFunctionName(*CIB)) != 0)
      return false;
  }

  return true;
}

// TODO: This is the same as the MachineOutliner, and should be consolidated
// into the same interface.
void IRInstructionMapper::convertToUnsignedVec(
    BasicBlock &BB, std::vector<IRInstructionData *> &InstrList,
    std::vector<unsigned> &IntegerMapping) {
  BasicBlock::iterator It = BB.begin();

  std::vector<unsigned> IntegerMappingForBB;
  std::vector<IRInstructionData *> InstrListForBB;

  HaveLegalRange = false;
  CanCombineWithPrevInstr = false;
  AddedIllegalLastTime = true;

  for (BasicBlock::iterator Et = BB.end(); It != Et; ++It) {
    switch (InstClassifier.visit(*It)) {
    case InstrType::Legal:
      mapToLegalUnsigned(It, IntegerMappingForBB, InstrListForBB);
      break;
    case InstrType::Illegal:
      mapToIllegalUnsigned(It, IntegerMappingForBB, InstrListForBB);
      break;
    case InstrType::Invisible:
      AddedIllegalLastTime = false;
      break;
    }
  }

  if (HaveLegalRange) {
    mapToIllegalUnsigned(It, IntegerMappingForBB, InstrListForBB, true);
    for (IRInstructionData *ID : InstrListForBB)
      this->IDL->push_back(*ID);
    llvm::append_range(InstrList, InstrListForBB);
    llvm::append_range(IntegerMapping, IntegerMappingForBB);
  }
}

// TODO: This is the same as the MachineOutliner, and should be consolidated
// into the same interface.
unsigned IRInstructionMapper::mapToLegalUnsigned(
    BasicBlock::iterator &It, std::vector<unsigned> &IntegerMappingForBB,
    std::vector<IRInstructionData *> &InstrListForBB) {
  // We added something legal, so we should unset the AddedLegalLastTime
  // flag.
  AddedIllegalLastTime = false;

  // If we have at least two adjacent legal instructions (which may have
  // invisible instructions in between), remember that.
  if (CanCombineWithPrevInstr)
    HaveLegalRange = true;
  CanCombineWithPrevInstr = true;

  // Get the integer for this instruction or give it the current
  // LegalInstrNumber.
  IRInstructionData *ID = allocateIRInstructionData(*It, true, *IDL);
  InstrListForBB.push_back(ID);

  // Add to the instruction list
  bool WasInserted;
  DenseMap<IRInstructionData *, unsigned, IRInstructionDataTraits>::iterator
      ResultIt;
  std::tie(ResultIt, WasInserted) =
      InstructionIntegerMap.insert(std::make_pair(ID, LegalInstrNumber));
  unsigned INumber = ResultIt->second;

  // There was an insertion.
  if (WasInserted)
    LegalInstrNumber++;

  IntegerMappingForBB.push_back(INumber);

  // Make sure we don't overflow or use any integers reserved by the DenseMap.
  assert(LegalInstrNumber < IllegalInstrNumber &&
         "Instruction mapping overflow!");

  assert(LegalInstrNumber != DenseMapInfo<unsigned>::getEmptyKey() &&
         "Tried to assign DenseMap tombstone or empty key to instruction.");
  assert(LegalInstrNumber != DenseMapInfo<unsigned>::getTombstoneKey() &&
         "Tried to assign DenseMap tombstone or empty key to instruction.");

  return INumber;
}

IRInstructionData *
IRInstructionMapper::allocateIRInstructionData(Instruction &I, bool Legality,
                                               IRInstructionDataList &IDL) {
  return new (InstDataAllocator->Allocate()) IRInstructionData(I, Legality, IDL);
}

IRInstructionDataList *
IRInstructionMapper::allocateIRInstructionDataList() {
  return new (IDLAllocator->Allocate()) IRInstructionDataList();
}

// TODO: This is the same as the MachineOutliner, and should be consolidated
// into the same interface.
unsigned IRInstructionMapper::mapToIllegalUnsigned(
    BasicBlock::iterator &It, std::vector<unsigned> &IntegerMappingForBB,
    std::vector<IRInstructionData *> &InstrListForBB, bool End) {
  // Can't combine an illegal instruction. Set the flag.
  CanCombineWithPrevInstr = false;

  // Only add one illegal number per range of legal numbers.
  if (AddedIllegalLastTime)
    return IllegalInstrNumber;

  IRInstructionData *ID = nullptr;
  if (!End)
    ID = allocateIRInstructionData(*It, false, *IDL);
  InstrListForBB.push_back(ID);

  // Remember that we added an illegal number last time.
  AddedIllegalLastTime = true;
  unsigned INumber = IllegalInstrNumber;
  IntegerMappingForBB.push_back(IllegalInstrNumber--);

  assert(LegalInstrNumber < IllegalInstrNumber &&
         "Instruction mapping overflow!");

  assert(IllegalInstrNumber != DenseMapInfo<unsigned>::getEmptyKey() &&
         "IllegalInstrNumber cannot be DenseMap tombstone or empty key!");

  assert(IllegalInstrNumber != DenseMapInfo<unsigned>::getTombstoneKey() &&
         "IllegalInstrNumber cannot be DenseMap tombstone or empty key!");

  return INumber;
}

IRSimilarityCandidate::IRSimilarityCandidate(unsigned StartIdx, unsigned Len,
                                             IRInstructionData *FirstInstIt,
                                             IRInstructionData *LastInstIt)
    : StartIdx(StartIdx), Len(Len) {

  assert(FirstInstIt != nullptr && "Instruction is nullptr!");
  assert(LastInstIt != nullptr && "Instruction is nullptr!");
  assert(StartIdx + Len > StartIdx &&
         "Overflow for IRSimilarityCandidate range?");
  assert(Len - 1 == static_cast<unsigned>(std::distance(
                        iterator(FirstInstIt), iterator(LastInstIt))) &&
         "Length of the first and last IRInstructionData do not match the "
         "given length");

  // We iterate over the given instructions, and map each unique value
  // to a unique number in the IRSimilarityCandidate ValueToNumber and
  // NumberToValue maps.  A constant get its own value globally, the individual
  // uses of the constants are not considered to be unique.
  //
  // IR:                    Mapping Added:
  // %add1 = add i32 %a, c1    %add1 -> 3, %a -> 1, c1 -> 2
  // %add2 = add i32 %a, %1    %add2 -> 4
  // %add3 = add i32 c2, c1    %add3 -> 6, c2 -> 5
  //
  // when replace with global values, starting from 1, would be
  //
  // 3 = add i32 1, 2
  // 4 = add i32 1, 3
  // 6 = add i32 5, 2
  unsigned LocalValNumber = 1;
  IRInstructionDataList::iterator ID = iterator(*FirstInstIt);
  for (unsigned Loc = StartIdx; Loc < StartIdx + Len; Loc++, ID++) {
    // Map the operand values to an unsigned integer if it does not already
    // have an unsigned integer assigned to it.
    for (Value *Arg : ID->OperVals)
      if (ValueToNumber.find(Arg) == ValueToNumber.end()) {
        ValueToNumber.try_emplace(Arg, LocalValNumber);
        NumberToValue.try_emplace(LocalValNumber, Arg);
        LocalValNumber++;
      }

    // Mapping the instructions to an unsigned integer if it is not already
    // exist in the mapping.
    if (ValueToNumber.find(ID->Inst) == ValueToNumber.end()) {
      ValueToNumber.try_emplace(ID->Inst, LocalValNumber);
      NumberToValue.try_emplace(LocalValNumber, ID->Inst);
      LocalValNumber++;
    }
  }

  // Setting the first and last instruction data pointers for the candidate.  If
  // we got through the entire for loop without hitting an assert, we know
  // that both of these instructions are not nullptrs.
  FirstInst = FirstInstIt;
  LastInst = LastInstIt;
}

bool IRSimilarityCandidate::isSimilar(const IRSimilarityCandidate &A,
                                      const IRSimilarityCandidate &B) {
  if (A.getLength() != B.getLength())
    return false;

  auto InstrDataForBoth =
      zip(make_range(A.begin(), A.end()), make_range(B.begin(), B.end()));

  return all_of(InstrDataForBoth,
                [](std::tuple<IRInstructionData &, IRInstructionData &> R) {
                  IRInstructionData &A = std::get<0>(R);
                  IRInstructionData &B = std::get<1>(R);
                  if (!A.Legal || !B.Legal)
                    return false;
                  return isClose(A, B);
                });
}

/// Determine if one or more of the assigned global value numbers for the
/// operands in \p TargetValueNumbers is in the current mapping set for operand
/// numbers in \p SourceOperands.  The set of possible corresponding global
/// value numbers are replaced with the most recent version of compatible
/// values.
///
/// \param [in] SourceValueToNumberMapping - The mapping of a Value to global
/// value number for the source IRInstructionCandidate.
/// \param [in, out] CurrentSrcTgtNumberMapping - The current mapping of source
/// IRSimilarityCandidate global value numbers to a set of possible numbers in
/// the target.
/// \param [in] SourceOperands - The operands in the original
/// IRSimilarityCandidate in the current instruction.
/// \param [in] TargetValueNumbers - The global value numbers of the operands in
/// the corresponding Instruction in the other IRSimilarityCandidate.
/// \returns true if there exists a possible mapping between the source
/// Instruction operands and the target Instruction operands, and false if not.
static bool checkNumberingAndReplaceCommutative(
  const DenseMap<Value *, unsigned> &SourceValueToNumberMapping,
  DenseMap<unsigned, DenseSet<unsigned>> &CurrentSrcTgtNumberMapping,
  ArrayRef<Value *> &SourceOperands,
  DenseSet<unsigned> &TargetValueNumbers){

  DenseMap<unsigned, DenseSet<unsigned>>::iterator ValueMappingIt;

  unsigned ArgVal;
  bool WasInserted;

  // Iterate over the operands in the source IRSimilarityCandidate to determine
  // whether there exists an operand in the other IRSimilarityCandidate that
  // creates a valid mapping of Value to Value between the
  // IRSimilarityCaniddates.
  for (Value *V : SourceOperands) {
    ArgVal = SourceValueToNumberMapping.find(V)->second;

    std::tie(ValueMappingIt, WasInserted) = CurrentSrcTgtNumberMapping.insert(
        std::make_pair(ArgVal, TargetValueNumbers));

    // Instead of finding a current mapping, we inserted a set.  This means a
    // mapping did not exist for the source Instruction operand, it has no
    // current constraints we need to check.
    if (WasInserted)
      continue;

    // If a mapping already exists for the source operand to the values in the
    // other IRSimilarityCandidate we need to iterate over the items in other
    // IRSimilarityCandidate's Instruction to determine whether there is a valid
    // mapping of Value to Value.
    DenseSet<unsigned> NewSet;
    for (unsigned &Curr : ValueMappingIt->second)
      // If we can find the value in the mapping, we add it to the new set.
      if (TargetValueNumbers.contains(Curr))
        NewSet.insert(Curr);

    // If we could not find a Value, return 0.
    if (NewSet.empty())
      return false;
    
    // Otherwise replace the old mapping with the newly constructed one.
    if (NewSet.size() != ValueMappingIt->second.size())
      ValueMappingIt->second.swap(NewSet);

    // We have reached no conclusions about the mapping, and cannot remove
    // any items from the other operands, so we move to check the next operand.
    if (ValueMappingIt->second.size() != 1)
      continue;


    unsigned ValToRemove = *ValueMappingIt->second.begin();
    // When there is only one item left in the mapping for and operand, remove
    // the value from the other operands.  If it results in there being no
    // mapping, return false, it means the mapping is wrong
    for (Value *InnerV : SourceOperands) {
      if (V == InnerV)
        continue;

      unsigned InnerVal = SourceValueToNumberMapping.find(InnerV)->second;
      ValueMappingIt = CurrentSrcTgtNumberMapping.find(InnerVal);
      if (ValueMappingIt == CurrentSrcTgtNumberMapping.end())
        continue;

      ValueMappingIt->second.erase(ValToRemove);
      if (ValueMappingIt->second.empty())
        return false;
    }
  }

  return true;
}

/// Determine if operand number \p TargetArgVal is in the current mapping set
/// for operand number \p SourceArgVal.
///
/// \param [in, out] CurrentSrcTgtNumberMapping current mapping of global
/// value numbers from source IRSimilarityCandidate to target
/// IRSimilarityCandidate.
/// \param [in] SourceArgVal The global value number for an operand in the
/// in the original candidate.
/// \param [in] TargetArgVal The global value number for the corresponding
/// operand in the other candidate.
/// \returns True if there exists a mapping and false if not.
bool checkNumberingAndReplace(
    DenseMap<unsigned, DenseSet<unsigned>> &CurrentSrcTgtNumberMapping,
    unsigned SourceArgVal, unsigned TargetArgVal) {
  // We are given two unsigned integers representing the global values of
  // the operands in different IRSimilarityCandidates and a current mapping
  // between the two.
  //
  // Source Operand GVN: 1
  // Target Operand GVN: 2
  // CurrentMapping: {1: {1, 2}}
  //
  // Since we have mapping, and the target operand is contained in the set, we
  // update it to:
  // CurrentMapping: {1: {2}}
  // and can return true. But, if the mapping was
  // CurrentMapping: {1: {3}}
  // we would return false.

  bool WasInserted;
  DenseMap<unsigned, DenseSet<unsigned>>::iterator Val;

  std::tie(Val, WasInserted) = CurrentSrcTgtNumberMapping.insert(
      std::make_pair(SourceArgVal, DenseSet<unsigned>({TargetArgVal})));

  // If we created a new mapping, then we are done.
  if (WasInserted)
    return true;

  // If there is more than one option in the mapping set, and the target value
  // is included in the mapping set replace that set with one that only includes
  // the target value, as it is the only valid mapping via the non commutative
  // instruction.

  DenseSet<unsigned> &TargetSet = Val->second;
  if (TargetSet.size() > 1 && TargetSet.contains(TargetArgVal)) {
    TargetSet.clear();
    TargetSet.insert(TargetArgVal);
    return true;
  }

  // Return true if we can find the value in the set.
  return TargetSet.contains(TargetArgVal);
}

bool IRSimilarityCandidate::compareNonCommutativeOperandMapping(
    OperandMapping A, OperandMapping B) {
  // Iterators to keep track of where we are in the operands for each
  // Instruction.
  ArrayRef<Value *>::iterator VItA = A.OperVals.begin();
  ArrayRef<Value *>::iterator VItB = B.OperVals.begin();
  unsigned OperandLength = A.OperVals.size();

  // For each operand, get the value numbering and ensure it is consistent.
  for (unsigned Idx = 0; Idx < OperandLength; Idx++, VItA++, VItB++) {
    unsigned OperValA = A.IRSC.ValueToNumber.find(*VItA)->second;
    unsigned OperValB = B.IRSC.ValueToNumber.find(*VItB)->second;

    // Attempt to add a set with only the target value.  If there is no mapping
    // we can create it here.
    //
    // For an instruction like a subtraction:
    // IRSimilarityCandidateA:  IRSimilarityCandidateB:
    // %resultA = sub %a, %b    %resultB = sub %d, %e
    //
    // We map %a -> %d and %b -> %e.
    //
    // And check to see whether their mapping is consistent in
    // checkNumberingAndReplace.

    if (!checkNumberingAndReplace(A.ValueNumberMapping, OperValA, OperValB))
      return false;

    if (!checkNumberingAndReplace(B.ValueNumberMapping, OperValB, OperValA))
      return false;
  }
  return true;
}

bool IRSimilarityCandidate::compareCommutativeOperandMapping(
    OperandMapping A, OperandMapping B) {
  DenseSet<unsigned> ValueNumbersA;      
  DenseSet<unsigned> ValueNumbersB;

  ArrayRef<Value *>::iterator VItA = A.OperVals.begin();
  ArrayRef<Value *>::iterator VItB = B.OperVals.begin();
  unsigned OperandLength = A.OperVals.size();

  // Find the value number sets for the operands.
  for (unsigned Idx = 0; Idx < OperandLength;
       Idx++, VItA++, VItB++) {
    ValueNumbersA.insert(A.IRSC.ValueToNumber.find(*VItA)->second);
    ValueNumbersB.insert(B.IRSC.ValueToNumber.find(*VItB)->second);
  }

  // Iterate over the operands in the first IRSimilarityCandidate and make sure
  // there exists a possible mapping with the operands in the second
  // IRSimilarityCandidate.
  if (!checkNumberingAndReplaceCommutative(A.IRSC.ValueToNumber,
                                           A.ValueNumberMapping, A.OperVals,
                                           ValueNumbersB))
    return false;

  // Iterate over the operands in the second IRSimilarityCandidate and make sure
  // there exists a possible mapping with the operands in the first
  // IRSimilarityCandidate.
  if (!checkNumberingAndReplaceCommutative(B.IRSC.ValueToNumber,
                                           B.ValueNumberMapping, B.OperVals,
                                           ValueNumbersA))
    return false;

  return true;
}

bool IRSimilarityCandidate::compareStructure(const IRSimilarityCandidate &A,
                                             const IRSimilarityCandidate &B) {
  if (A.getLength() != B.getLength())
    return false;

  if (A.ValueToNumber.size() != B.ValueToNumber.size())
    return false;

  iterator ItA = A.begin();
  iterator ItB = B.begin();

  // These sets create a create a mapping between the values in one candidate
  // to values in the other candidate.  If we create a set with one element,
  // and that same element maps to the original element in the candidate
  // we have a good mapping.
  DenseMap<unsigned, DenseSet<unsigned>> ValueNumberMappingA;
  DenseMap<unsigned, DenseSet<unsigned>> ValueNumberMappingB;
  DenseMap<unsigned, DenseSet<unsigned>>::iterator ValueMappingIt;

  bool WasInserted;

  // Iterate over the instructions contained in each candidate
  unsigned SectionLength = A.getStartIdx() + A.getLength();
  for (unsigned Loc = A.getStartIdx(); Loc < SectionLength;
       ItA++, ItB++, Loc++) {
    // Make sure the instructions are similar to one another.
    if (!isClose(*ItA, *ItB))
      return false;

    Instruction *IA = ItA->Inst;
    Instruction *IB = ItB->Inst;

    if (!ItA->Legal || !ItB->Legal)
      return false;

    // Get the operand sets for the instructions.
    ArrayRef<Value *> OperValsA = ItA->OperVals;
    ArrayRef<Value *> OperValsB = ItB->OperVals;

    unsigned InstValA = A.ValueToNumber.find(IA)->second;
    unsigned InstValB = B.ValueToNumber.find(IB)->second;

    // Ensure that the mappings for the instructions exists.
    std::tie(ValueMappingIt, WasInserted) = ValueNumberMappingA.insert(
        std::make_pair(InstValA, DenseSet<unsigned>({InstValB})));
    if (!WasInserted && !ValueMappingIt->second.contains(InstValB))
      return false;

    std::tie(ValueMappingIt, WasInserted) = ValueNumberMappingB.insert(
        std::make_pair(InstValB, DenseSet<unsigned>({InstValA})));
    if (!WasInserted && !ValueMappingIt->second.contains(InstValA))
      return false;

    // We have different paths for commutative instructions and non-commutative
    // instructions since commutative instructions could allow multiple mappings
    // to certain values.
    if (IA->isCommutative() && !isa<FPMathOperator>(IA)) {
      if (!compareCommutativeOperandMapping(
              {A, OperValsA, ValueNumberMappingA},
              {B, OperValsB, ValueNumberMappingB}))
        return false;
      continue;
    }

    // Handle the non-commutative cases.
    if (!compareNonCommutativeOperandMapping(
            {A, OperValsA, ValueNumberMappingA},
            {B, OperValsB, ValueNumberMappingB}))
      return false;
  }
  return true;
}

bool IRSimilarityCandidate::overlap(const IRSimilarityCandidate &A,
                                    const IRSimilarityCandidate &B) {
  auto DoesOverlap = [](const IRSimilarityCandidate &X,
                        const IRSimilarityCandidate &Y) {
    // Check:
    // XXXXXX        X starts before Y ends
    //      YYYYYYY  Y starts after X starts
    return X.StartIdx <= Y.getEndIdx() && Y.StartIdx >= X.StartIdx;
  };

  return DoesOverlap(A, B) || DoesOverlap(B, A);
}

void IRSimilarityIdentifier::populateMapper(
    Module &M, std::vector<IRInstructionData *> &InstrList,
    std::vector<unsigned> &IntegerMapping) {

  std::vector<IRInstructionData *> InstrListForModule;
  std::vector<unsigned> IntegerMappingForModule;
  // Iterate over the functions in the module to map each Instruction in each
  // BasicBlock to an unsigned integer.
  for (Function &F : M) {

    if (F.empty())
      continue;

    for (BasicBlock &BB : F) {

      if (BB.sizeWithoutDebug() < 2)
        continue;

      // BB has potential to have similarity since it has a size greater than 2
      // and can therefore match other regions greater than 2. Map it to a list
      // of unsigned integers.
      Mapper.convertToUnsignedVec(BB, InstrListForModule,
                                  IntegerMappingForModule);
    }
  }

  // Insert the InstrListForModule at the end of the overall InstrList so that
  // we can have a long InstrList for the entire set of Modules being analyzed.
  llvm::append_range(InstrList, InstrListForModule);
  // Do the same as above, but for IntegerMapping.
  llvm::append_range(IntegerMapping, IntegerMappingForModule);
}

void IRSimilarityIdentifier::populateMapper(
    ArrayRef<std::unique_ptr<Module>> &Modules,
    std::vector<IRInstructionData *> &InstrList,
    std::vector<unsigned> &IntegerMapping) {

  // Iterate over, and map the instructions in each module.
  for (const std::unique_ptr<Module> &M : Modules)
    populateMapper(*M, InstrList, IntegerMapping);
}

/// From a repeated subsequence, find all the different instances of the
/// subsequence from the \p InstrList, and create an IRSimilarityCandidate from
/// the IRInstructionData in subsequence.
///
/// \param [in] Mapper - The instruction mapper for sanity checks.
/// \param [in] InstrList - The vector that holds the instruction data.
/// \param [in] IntegerMapping - The vector that holds the mapped integers.
/// \param [out] CandsForRepSubstring - The vector to store the generated
/// IRSimilarityCandidates.
static void createCandidatesFromSuffixTree(
    const IRInstructionMapper& Mapper, std::vector<IRInstructionData *> &InstrList,
    std::vector<unsigned> &IntegerMapping, SuffixTree::RepeatedSubstring &RS,
    std::vector<IRSimilarityCandidate> &CandsForRepSubstring) {

  unsigned StringLen = RS.Length;

  // Create an IRSimilarityCandidate for instance of this subsequence \p RS.
  for (const unsigned &StartIdx : RS.StartIndices) {
    unsigned EndIdx = StartIdx + StringLen - 1;

    // Check that this subsequence does not contain an illegal instruction.
    bool ContainsIllegal = false;
    for (unsigned CurrIdx = StartIdx; CurrIdx <= EndIdx; CurrIdx++) {
      unsigned Key = IntegerMapping[CurrIdx];
      if (Key > Mapper.IllegalInstrNumber) {
        ContainsIllegal = true;
        break;
      }
    }

    // If we have an illegal instruction, we should not create an
    // IRSimilarityCandidate for this region.
    if (ContainsIllegal)
      continue;

    // We are getting iterators to the instructions in this region of code
    // by advancing the start and end indices from the start of the
    // InstrList.
    std::vector<IRInstructionData *>::iterator StartIt = InstrList.begin();
    std::advance(StartIt, StartIdx);
    std::vector<IRInstructionData *>::iterator EndIt = InstrList.begin();
    std::advance(EndIt, EndIdx);

    CandsForRepSubstring.emplace_back(StartIdx, StringLen, *StartIt, *EndIt);
  }
}

/// From the list of IRSimilarityCandidates, perform a comparison between each
/// IRSimilarityCandidate to determine if there are overlapping
/// IRInstructionData, or if they do not have the same structure.
///
/// \param [in] CandsForRepSubstring - The vector containing the
/// IRSimilarityCandidates.
/// \param [out] StructuralGroups - the mapping of unsigned integers to vector
/// of IRSimilarityCandidates where each of the IRSimilarityCandidates in the
/// vector are structurally similar to one another.
static void findCandidateStructures(
    std::vector<IRSimilarityCandidate> &CandsForRepSubstring,
    DenseMap<unsigned, SimilarityGroup> &StructuralGroups) {
  std::vector<IRSimilarityCandidate>::iterator CandIt, CandEndIt, InnerCandIt,
      InnerCandEndIt;

  // IRSimilarityCandidates each have a structure for operand use.  It is
  // possible that two instances of the same subsequences have different
  // structure. Each type of structure found is assigned a number.  This
  // DenseMap maps an IRSimilarityCandidate to which type of similarity
  // discovered it fits within.
  DenseMap<IRSimilarityCandidate *, unsigned> CandToGroup;

  // Find the compatibility from each candidate to the others to determine
  // which candidates overlap and which have the same structure by mapping
  // each structure to a different group.
  bool SameStructure;
  bool Inserted;
  unsigned CurrentGroupNum = 0;
  unsigned OuterGroupNum;
  DenseMap<IRSimilarityCandidate *, unsigned>::iterator CandToGroupIt;
  DenseMap<IRSimilarityCandidate *, unsigned>::iterator CandToGroupItInner;
  DenseMap<unsigned, SimilarityGroup>::iterator CurrentGroupPair;

  // Iterate over the candidates to determine its structural and overlapping
  // compatibility with other instructions
  for (CandIt = CandsForRepSubstring.begin(),
      CandEndIt = CandsForRepSubstring.end();
       CandIt != CandEndIt; CandIt++) {

    // Determine if it has an assigned structural group already.
    CandToGroupIt = CandToGroup.find(&*CandIt);
    if (CandToGroupIt == CandToGroup.end()) {
      // If not, we assign it one, and add it to our mapping.
      std::tie(CandToGroupIt, Inserted) =
          CandToGroup.insert(std::make_pair(&*CandIt, CurrentGroupNum++));
    }

    // Get the structural group number from the iterator.
    OuterGroupNum = CandToGroupIt->second;

    // Check if we already have a list of IRSimilarityCandidates for the current
    // structural group.  Create one if one does not exist.
    CurrentGroupPair = StructuralGroups.find(OuterGroupNum);
    if (CurrentGroupPair == StructuralGroups.end())
      std::tie(CurrentGroupPair, Inserted) = StructuralGroups.insert(
          std::make_pair(OuterGroupNum, SimilarityGroup({*CandIt})));

    // Iterate over the IRSimilarityCandidates following the current
    // IRSimilarityCandidate in the list to determine whether the two
    // IRSimilarityCandidates are compatible.  This is so we do not repeat pairs
    // of IRSimilarityCandidates.
    for (InnerCandIt = std::next(CandIt),
        InnerCandEndIt = CandsForRepSubstring.end();
         InnerCandIt != InnerCandEndIt; InnerCandIt++) {

      // We check if the inner item has a group already, if it does, we skip it.
      CandToGroupItInner = CandToGroup.find(&*InnerCandIt);
      if (CandToGroupItInner != CandToGroup.end())
        continue;

      // Otherwise we determine if they have the same structure and add it to
      // vector if they match.
      SameStructure =
          IRSimilarityCandidate::compareStructure(*CandIt, *InnerCandIt);
      if (!SameStructure)
        continue;

      CandToGroup.insert(std::make_pair(&*InnerCandIt, OuterGroupNum));
      CurrentGroupPair->second.push_back(*InnerCandIt);
    }
  }
}

void IRSimilarityIdentifier::findCandidates(
    std::vector<IRInstructionData *> &InstrList,
    std::vector<unsigned> &IntegerMapping) {
  SuffixTree ST(IntegerMapping);

  std::vector<IRSimilarityCandidate> CandsForRepSubstring;
  std::vector<SimilarityGroup> NewCandidateGroups;

  DenseMap<unsigned, SimilarityGroup> StructuralGroups;

  // Iterate over the subsequences found by the Suffix Tree to create
  // IRSimilarityCandidates for each repeated subsequence and determine which
  // instances are structurally similar to one another.
  for (SuffixTree::RepeatedSubstring &RS : ST) {
    createCandidatesFromSuffixTree(Mapper, InstrList, IntegerMapping, RS,
                                   CandsForRepSubstring);

    if (CandsForRepSubstring.size() < 2)
      continue;

    findCandidateStructures(CandsForRepSubstring, StructuralGroups);
    for (std::pair<unsigned, SimilarityGroup> &Group : StructuralGroups)
      // We only add the group if it contains more than one
      // IRSimilarityCandidate.  If there is only one, that means there is no
      // other repeated subsequence with the same structure.
      if (Group.second.size() > 1)
        SimilarityCandidates->push_back(Group.second);

    CandsForRepSubstring.clear();
    StructuralGroups.clear();
    NewCandidateGroups.clear();
  }
}

SimilarityGroupList &IRSimilarityIdentifier::findSimilarity(
    ArrayRef<std::unique_ptr<Module>> Modules) {
  resetSimilarityCandidates();

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> IntegerMapping;

  populateMapper(Modules, InstrList, IntegerMapping);
  findCandidates(InstrList, IntegerMapping);

  return SimilarityCandidates.getValue();
}

SimilarityGroupList &IRSimilarityIdentifier::findSimilarity(Module &M) {
  resetSimilarityCandidates();

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> IntegerMapping;

  populateMapper(M, InstrList, IntegerMapping);
  findCandidates(InstrList, IntegerMapping);

  return SimilarityCandidates.getValue();
}

INITIALIZE_PASS(IRSimilarityIdentifierWrapperPass, "ir-similarity-identifier",
                "ir-similarity-identifier", false, true)

IRSimilarityIdentifierWrapperPass::IRSimilarityIdentifierWrapperPass()
    : ModulePass(ID) {
  initializeIRSimilarityIdentifierWrapperPassPass(
      *PassRegistry::getPassRegistry());
}

bool IRSimilarityIdentifierWrapperPass::doInitialization(Module &M) {
  IRSI.reset(new IRSimilarityIdentifier());
  return false;
}

bool IRSimilarityIdentifierWrapperPass::doFinalization(Module &M) {
  IRSI.reset();
  return false;
}

bool IRSimilarityIdentifierWrapperPass::runOnModule(Module &M) {
  IRSI->findSimilarity(M);
  return false;
}

AnalysisKey IRSimilarityAnalysis::Key;
IRSimilarityIdentifier IRSimilarityAnalysis::run(Module &M,
                                               ModuleAnalysisManager &) {

  auto IRSI = IRSimilarityIdentifier();
  IRSI.findSimilarity(M);
  return IRSI;
}

PreservedAnalyses
IRSimilarityAnalysisPrinterPass::run(Module &M, ModuleAnalysisManager &AM) {
  IRSimilarityIdentifier &IRSI = AM.getResult<IRSimilarityAnalysis>(M);
  Optional<SimilarityGroupList> &SimilarityCandidatesOpt = IRSI.getSimilarity();

  for (std::vector<IRSimilarityCandidate> &CandVec : *SimilarityCandidatesOpt) {
    OS << CandVec.size() << " candidates of length "
       << CandVec.begin()->getLength() << ".  Found in: \n";
    for (IRSimilarityCandidate &Cand : CandVec) {
      OS << "  Function: " << Cand.front()->Inst->getFunction()->getName().str()
         << ", Basic Block: ";
      if (Cand.front()->Inst->getParent()->getName().str() == "")
        OS << "(unnamed)";
      else
        OS << Cand.front()->Inst->getParent()->getName().str();
      OS << "\n    Start Instruction: ";
      Cand.frontInstruction()->print(OS);
      OS << "\n      End Instruction: ";
      Cand.backInstruction()->print(OS);
      OS << "\n";
    }
  }

  return PreservedAnalyses::all();
}

char IRSimilarityIdentifierWrapperPass::ID = 0;
