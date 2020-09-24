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
#include "llvm/IR/User.h"

using namespace llvm;
using namespace IRSimilarity;

IRInstructionData::IRInstructionData(Instruction &I, bool Legality,
                                     IRInstructionDataList &IDList)
    : Inst(&I), Legal(Legality), IDL(&IDList) {
  // Here we collect the operands to be used to determine whether two
  // instructions are similar to one another.
  for (Use &OI : I.operands())
    OperVals.push_back(OI.get());
}

bool IRSimilarity::isClose(const IRInstructionData &A,
                           const IRInstructionData &B) {
  return A.Legal && A.Inst->isSameOperationAs(B.Inst);
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
    for_each(InstrListForBB,
             [this](IRInstructionData *ID) { this->IDL->push_back(*ID); });
    InstrList.insert(InstrList.end(), InstrListForBB.begin(),
                     InstrListForBB.end());
    IntegerMapping.insert(IntegerMapping.end(), IntegerMappingForBB.begin(),
                          IntegerMappingForBB.end());
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
  assert(Len - 1 ==
             std::distance(iterator(FirstInstIt), iterator(LastInstIt)) &&
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
