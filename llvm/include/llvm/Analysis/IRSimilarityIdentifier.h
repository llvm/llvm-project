//===- IRSimilarityIdentifier.h - Find similarity in a module --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// Interface file for the IRSimilarityIdentifier for identifying similarities in
// IR including the IRInstructionMapper, which maps an Instruction to unsigned
// integers.
//
// Two sequences of instructions are called "similar" if they perform the same
// series of operations for all inputs.
//
// \code
// %1 = add i32 %a, 10
// %2 = add i32 %a, %1
// %3 = icmp slt icmp %1, %2
// \endcode
//
// and
//
// \code
// %1 = add i32 11, %a
// %2 = sub i32 %a, %1
// %3 = icmp sgt icmp %2, %1
// \endcode
//
// ultimately have the same result, even if the inputs, and structure are
// slightly different.
//
// For instructions, we do not worry about operands that do not have fixed
// semantic meaning to the program.  We consider the opcode that the instruction
// has, the types, parameters, and extra information such as the function name,
// or comparison predicate.  These are used to create a hash to map instructions
// to integers to be used in similarity matching in sequences of instructions
//
// Terminology:
// An IRSimilarityCandidate is a region of IRInstructionData (wrapped
// Instructions), usually used to denote a region of similarity has been found.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_IRSIMILARITYIDENTIFIER_H
#define LLVM_ANALYSIS_IRSIMILARITYIDENTIFIER_H

#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Allocator.h"

namespace llvm {
namespace IRSimilarity {

struct IRInstructionDataList;

/// This represents what is and is not supported when finding similarity in
/// Instructions.
///
/// Legal Instructions are considered when looking at similarity between
/// Instructions.
///
/// Illegal Instructions cannot be considered when looking for similarity
/// between Instructions. They act as boundaries between similarity regions.
///
/// Invisible Instructions are skipped over during analysis.
// TODO: Shared with MachineOutliner
enum InstrType { Legal, Illegal, Invisible };

/// This provides the utilities for hashing an Instruction to an unsigned
/// integer. Two IRInstructionDatas produce the same hash value when their
/// underlying Instructions perform the same operation (even if they don't have
/// the same input operands.)
/// As a more concrete example, consider the following:
///
/// \code
/// %add1 = add i32 %a, %b
/// %add2 = add i32 %c, %d
/// %add3 = add i64 %e, %f
/// \endcode
///
// Then the IRInstructionData wrappers for these Instructions may be hashed like
/// so:
///
/// \code
/// ; These two adds have the same types and operand types, so they hash to the
/// ; same number.
/// %add1 = add i32 %a, %b ; Hash: 1
/// %add2 = add i32 %c, %d ; Hash: 1
/// ; This add produces an i64. This differentiates it from %add1 and %add2. So,
/// ; it hashes to a different number.
/// %add3 = add i64 %e, %f; Hash: 2
/// \endcode
///
///
/// This hashing scheme will be used to represent the program as a very long
/// string. This string can then be placed in a data structure which can be used
/// for similarity queries.
///
/// TODO: Handle types of Instructions which can be equal even with different
/// operands. (E.g. comparisons with swapped predicates.)
/// TODO: Handle CallInsts, which are only checked for function type
/// by \ref isSameOperationAs.
/// TODO: Handle GetElementPtrInsts, as some of the operands have to be the
/// exact same, and some do not.
struct IRInstructionData : ilist_node<IRInstructionData> {

  /// The source Instruction that is being wrapped.
  Instruction *Inst = nullptr;
  /// The values of the operands in the Instruction.
  SmallVector<Value *, 4> OperVals;
  /// The legality of the wrapped instruction. This is informed by InstrType,
  /// and is used when checking when two instructions are considered similar.
  /// If either instruction is not legal, the instructions are automatically not
  /// considered similar.
  bool Legal;

  /// Gather the information that is difficult to gather for an Instruction, or
  /// is changed. i.e. the operands of an Instruction and the Types of those
  /// operands. This extra information allows for similarity matching to make
  /// assertions that allow for more flexibility when checking for whether an
  /// Instruction performs the same operation.
  IRInstructionData(Instruction &I, bool Legality, IRInstructionDataList &IDL);

  /// Hashes \p Value based on its opcode, types, and operand types.
  /// Two IRInstructionData instances produce the same hash when they perform
  /// the same operation.
  ///
  /// As a simple example, consider the following instructions.
  ///
  /// \code
  /// %add1 = add i32 %x1, %y1
  /// %add2 = add i32 %x2, %y2
  ///
  /// %sub = sub i32 %x1, %y1
  ///
  /// %add_i64 = add i64 %x2, %y2
  /// \endcode
  ///
  /// Because the first two adds operate the same types, and are performing the
  /// same action, they will be hashed to the same value.
  ///
  /// However, the subtraction instruction is not the same as an addition, and
  /// will be hashed to a different value.
  ///
  /// Finally, the last add has a different type compared to the first two add
  /// instructions, so it will also be hashed to a different value that any of
  /// the previous instructions.
  ///
  /// \param [in] ID - The IRInstructionData instance to be hashed.
  /// \returns A hash_value of the IRInstructionData.
  friend hash_code hash_value(const IRInstructionData &ID) {
    SmallVector<Type *, 4> OperTypes;
    for (Value *V : ID.OperVals)
      OperTypes.push_back(V->getType());

    return llvm::hash_combine(
        llvm::hash_value(ID.Inst->getOpcode()),
        llvm::hash_value(ID.Inst->getType()),
        llvm::hash_combine_range(OperTypes.begin(), OperTypes.end()));
  }

  IRInstructionDataList *IDL = nullptr;
};

struct IRInstructionDataList : simple_ilist<IRInstructionData> {};

/// Compare one IRInstructionData class to another IRInstructionData class for
/// whether they are performing a the same operation, and can mapped to the
/// same value. For regular instructions if the hash value is the same, then
/// they will also be close.
///
/// \param A - The first IRInstructionData class to compare
/// \param B - The second IRInstructionData class to compare
/// \returns true if \p A and \p B are similar enough to be mapped to the same
/// value.
bool isClose(const IRInstructionData &A, const IRInstructionData &B);

struct IRInstructionDataTraits : DenseMapInfo<IRInstructionData *> {
  static inline IRInstructionData *getEmptyKey() { return nullptr; }
  static inline IRInstructionData *getTombstoneKey() {
    return reinterpret_cast<IRInstructionData *>(-1);
  }

  static unsigned getHashValue(const IRInstructionData *E) {
    using llvm::hash_value;
    assert(E && "IRInstructionData is a nullptr?");
    return hash_value(*E);
  }

  static bool isEqual(const IRInstructionData *LHS,
                      const IRInstructionData *RHS) {
    if (RHS == getEmptyKey() || RHS == getTombstoneKey() ||
        LHS == getEmptyKey() || LHS == getTombstoneKey())
      return LHS == RHS;

    assert(LHS && RHS && "nullptr should have been caught by getEmptyKey?");
    return isClose(*LHS, *RHS);
  }
};

/// Helper struct for converting the Instructions in a Module into a vector of
/// unsigned integers. This vector of unsigned integers can be thought of as a
/// "numeric string". This numeric string can then be queried by, for example,
/// data structures that find repeated substrings.
///
/// This hashing is done per BasicBlock in the module. To hash Instructions
/// based off of their operations, each Instruction is wrapped in an
/// IRInstructionData struct. The unsigned integer for an IRInstructionData
/// depends on:
/// - The hash provided by the IRInstructionData.
/// - Which member of InstrType the IRInstructionData is classified as.
// See InstrType for more details on the possible classifications, and how they
// manifest in the numeric string.
///
/// The numeric string for an individual BasicBlock is terminated by an unique
/// unsigned integer. This prevents data structures which rely on repetition
/// from matching across BasicBlocks. (For example, the SuffixTree.)
/// As a concrete example, if we have the following two BasicBlocks:
/// \code
/// bb0:
/// %add1 = add i32 %a, %b
/// %add2 = add i32 %c, %d
/// %add3 = add i64 %e, %f
/// bb1:
/// %sub = sub i32 %c, %d
/// \endcode
/// We may hash the Instructions like this (via IRInstructionData):
/// \code
/// bb0:
/// %add1 = add i32 %a, %b ; Hash: 1
/// %add2 = add i32 %c, %d; Hash: 1
/// %add3 = add i64 %e, %f; Hash: 2
/// bb1:
/// %sub = sub i32 %c, %d; Hash: 3
/// %add4 = add i32 %c, %d ; Hash: 1
/// \endcode
/// And produce a "numeric string representation" like so:
/// 1, 1, 2, unique_integer_1, 3, 1, unique_integer_2
///
/// TODO: This is very similar to the MachineOutliner, and should be
/// consolidated into the same interface.
struct IRInstructionMapper {
  /// The starting illegal instruction number to map to.
  ///
  /// Set to -3 for compatibility with DenseMapInfo<unsigned>.
  unsigned IllegalInstrNumber = static_cast<unsigned>(-3);

  /// The next available integer to assign to a legal Instruction to.
  unsigned LegalInstrNumber = 0;

  /// Correspondence from IRInstructionData to unsigned integers.
  DenseMap<IRInstructionData *, unsigned, IRInstructionDataTraits>
      InstructionIntegerMap;

  /// Set if we added an illegal number in the previous step.
  /// Since each illegal number is unique, we only need one of them between
  /// each range of legal numbers. This lets us make sure we don't add more
  /// than one illegal number per range.
  bool AddedIllegalLastTime = false;

  /// Marks whether we found a illegal instruction in the previous step.
  bool CanCombineWithPrevInstr = false;

  /// Marks whether we have found a set of instructions that is long enough
  /// to be considered for similarity.
  bool HaveLegalRange = false;

  /// This allocator pointer is in charge of holding on to the IRInstructionData
  /// so it is not deallocated until whatever external tool is using it is done
  /// with the information.
  SpecificBumpPtrAllocator<IRInstructionData> *InstDataAllocator = nullptr;

  /// This allocator pointer is in charge of creating the IRInstructionDataList
  /// so it is not deallocated until whatever external tool is using it is done
  /// with the information.
  SpecificBumpPtrAllocator<IRInstructionDataList> *IDLAllocator = nullptr;

  /// Get an allocated IRInstructionData struct using the InstDataAllocator.
  ///
  /// \param I - The Instruction to wrap with IRInstructionData.
  /// \param Legality - A boolean value that is true if the instruction is to
  /// be considered for similarity, and false if not.
  /// \param IDL - The InstructionDataList that the IRInstructionData is
  /// inserted into.
  /// \returns An allocated IRInstructionData struct.
  IRInstructionData *allocateIRInstructionData(Instruction &I, bool Legality,
                                               IRInstructionDataList &IDL);

  /// Get an allocated IRInstructionDataList object using the IDLAllocator.
  ///
  /// \returns An allocated IRInstructionDataList object.
  IRInstructionDataList *allocateIRInstructionDataList();

  IRInstructionDataList *IDL = nullptr;

  /// Maps the Instructions in a BasicBlock \p BB to legal or illegal integers
  /// determined by \p InstrType. Two Instructions are mapped to the same value
  /// if they are close as defined by the InstructionData class above.
  ///
  /// \param [in] BB - The BasicBlock to be mapped to integers.
  /// \param [in,out] InstrList - Vector of IRInstructionData to append to.
  /// \param [in,out] IntegerMapping - Vector of unsigned integers to append to.
  void convertToUnsignedVec(BasicBlock &BB,
                            std::vector<IRInstructionData *> &InstrList,
                            std::vector<unsigned> &IntegerMapping);

  /// Maps an Instruction to a legal integer.
  ///
  /// \param [in] It - The Instruction to be mapped to an integer.
  /// \param [in,out] IntegerMappingForBB - Vector of unsigned integers to
  /// append to.
  /// \param [in,out] InstrListForBB - Vector of InstructionData to append to.
  /// \returns The integer \p It was mapped to.
  unsigned mapToLegalUnsigned(BasicBlock::iterator &It,
                              std::vector<unsigned> &IntegerMappingForBB,
                              std::vector<IRInstructionData *> &InstrListForBB);

  /// Maps an Instruction to an illegal integer.
  ///
  /// \param [in] It - The \p Instruction to be mapped to an integer.
  /// \param [in,out] IntegerMappingForBB - Vector of unsigned integers to
  /// append to.
  /// \param [in,out] InstrListForBB - Vector of IRInstructionData to append to.
  /// \param End - true if creating a dummy IRInstructionData at the end of a
  /// basic block.
  /// \returns The integer \p It was mapped to.
  unsigned mapToIllegalUnsigned(
      BasicBlock::iterator &It, std::vector<unsigned> &IntegerMappingForBB,
      std::vector<IRInstructionData *> &InstrListForBB, bool End = false);

  IRInstructionMapper(SpecificBumpPtrAllocator<IRInstructionData> *IDA,
                      SpecificBumpPtrAllocator<IRInstructionDataList> *IDLA)
      : InstDataAllocator(IDA), IDLAllocator(IDLA) {
    // Make sure that the implementation of DenseMapInfo<unsigned> hasn't
    // changed.
    assert(DenseMapInfo<unsigned>::getEmptyKey() == static_cast<unsigned>(-1) &&
           "DenseMapInfo<unsigned>'s empty key isn't -1!");
    assert(DenseMapInfo<unsigned>::getTombstoneKey() ==
               static_cast<unsigned>(-2) &&
           "DenseMapInfo<unsigned>'s tombstone key isn't -2!");

    IDL = new (IDLAllocator->Allocate())
        IRInstructionDataList();
  }

  /// Custom InstVisitor to classify different instructions for whether it can
  /// be analyzed for similarity.
  struct InstructionClassification
      : public InstVisitor<InstructionClassification, InstrType> {
    InstructionClassification() {}

    // TODO: Determine a scheme to resolve when the label is similar enough.
    InstrType visitBranchInst(BranchInst &BI) { return Illegal; }
    // TODO: Determine a scheme to resolve when the labels are similar enough.
    InstrType visitPHINode(PHINode &PN) { return Illegal; }
    // TODO: Handle allocas.
    InstrType visitAllocaInst(AllocaInst &AI) { return Illegal; }
    // We exclude variable argument instructions since variable arguments
    // requires extra checking of the argument list.
    InstrType visitVAArgInst(VAArgInst &VI) { return Illegal; }
    // We exclude all exception handling cases since they are so context
    // dependent.
    InstrType visitLandingPadInst(LandingPadInst &LPI) { return Illegal; }
    InstrType visitFuncletPadInst(FuncletPadInst &FPI) { return Illegal; }
    // DebugInfo should be included in the regions, but should not be
    // analyzed for similarity as it has no bearing on the outcome of the
    // program.
    InstrType visitDbgInfoIntrinsic(DbgInfoIntrinsic &DII) { return Invisible; }
    // TODO: Handle GetElementPtrInsts
    InstrType visitGetElementPtrInst(GetElementPtrInst &GEPI) {
      return Illegal;
    }
    // TODO: Handle specific intrinsics.
    InstrType visitIntrinsicInst(IntrinsicInst &II) { return Illegal; }
    // TODO: Handle CallInsts.
    InstrType visitCallInst(CallInst &CI) { return Illegal; }
    // TODO: We do not current handle similarity that changes the control flow.
    InstrType visitInvokeInst(InvokeInst &II) { return Illegal; }
    // TODO: We do not current handle similarity that changes the control flow.
    InstrType visitCallBrInst(CallBrInst &CBI) { return Illegal; }
    // TODO: Handle interblock similarity.
    InstrType visitTerminator(Instruction &I) { return Illegal; }
    InstrType visitInstruction(Instruction &I) { return Legal; }
  };

  /// Maps an Instruction to a member of InstrType.
  InstructionClassification InstClassifier;
};

/// This is a class that wraps a range of IRInstructionData from one point to
/// another in the vector of IRInstructionData, which is a region of the
/// program.  It is also responsible for defining the structure within this
/// region of instructions.
///
/// The structure of a region is defined through a value numbering system
/// assigned to each unique value in a region at the creation of the
/// IRSimilarityCandidate.
///
/// For example, for each Instruction we add a mapping for each new
/// value seen in that Instruction.
/// IR:                    Mapping Added:
/// %add1 = add i32 %a, c1    %add1 -> 3, %a -> 1, c1 -> 2
/// %add2 = add i32 %a, %1    %add2 -> 4
/// %add3 = add i32 c2, c1    %add3 -> 6, c2 -> 5
///
/// We can compare IRSimilarityCandidates against one another.
/// The \ref isSimilar function compares each IRInstructionData against one
/// another and if we have the same sequences of IRInstructionData that would
/// create the same hash, we have similar IRSimilarityCandidates.
class IRSimilarityCandidate {
private:
  /// The start index of this IRSimilarityCandidate in the instruction list.
  unsigned StartIdx = 0;

  /// The number of instructions in this IRSimilarityCandidate.
  unsigned Len = 0;

  /// The first instruction in this IRSimilarityCandidate.
  IRInstructionData *FirstInst = nullptr;

  /// The last instruction in this IRSimilarityCandidate.
  IRInstructionData *LastInst = nullptr;

  /// Global Value Numbering structures
  /// @{
  /// Stores the mapping of the value to the number assigned to it in the
  /// IRSimilarityCandidate.
  DenseMap<Value *, unsigned> ValueToNumber;
  /// Stores the mapping of the number to the value assigned this number.
  DenseMap<unsigned, Value *> NumberToValue;
  /// @}

public:
  /// \param StartIdx - The starting location of the region.
  /// \param StartIdx - The length of the region.
  /// \param FirstInstIt - The starting IRInstructionData of the region.
  /// \param LastInstIt - The ending IRInstructionData of the region.
  IRSimilarityCandidate(unsigned StartIdx, unsigned Len,
                        IRInstructionData *FirstInstIt,
                        IRInstructionData *LastInstIt);

  /// \param A - The first IRInstructionCandidate to compare.
  /// \param B - The second IRInstructionCandidate to compare.
  /// \returns True when every IRInstructionData in \p A is similar to every
  /// IRInstructionData in \p B.
  static bool isSimilar(const IRSimilarityCandidate &A,
                        const IRSimilarityCandidate &B);
  /// Compare the start and end indices of the two IRSimilarityCandidates for
  /// whether they overlap. If the start instruction of one
  /// IRSimilarityCandidate is less than the end instruction of the other, and
  /// the start instruction of one is greater than the start instruction of the
  /// other, they overlap.
  ///
  /// \returns true if the IRSimilarityCandidates do not have overlapping
  /// instructions.
  static bool overlap(const IRSimilarityCandidate &A,
                      const IRSimilarityCandidate &B);

  /// \returns the number of instructions in this Candidate.
  unsigned getLength() const { return Len; }

  /// \returns the start index of this IRSimilarityCandidate.
  unsigned getStartIdx() const { return StartIdx; }

  /// \returns the end index of this IRSimilarityCandidate.
  unsigned getEndIdx() const { return StartIdx + Len - 1; }

  /// \returns The first IRInstructionData.
  IRInstructionData *front() const { return FirstInst; }
  /// \returns The last IRInstructionData.
  IRInstructionData *back() const { return LastInst; }

  /// \returns The first Instruction.
  Instruction *frontInstruction() { return FirstInst->Inst; }
  /// \returns The last Instruction
  Instruction *backInstruction() { return LastInst->Inst; }

  /// \returns The BasicBlock the IRSimilarityCandidate starts in.
  BasicBlock *getStartBB() { return FirstInst->Inst->getParent(); }
  /// \returns The BasicBlock the IRSimilarityCandidate ends in.
  BasicBlock *getEndBB() { return LastInst->Inst->getParent(); }

  /// \returns The Function that the IRSimilarityCandidate is located in.
  Function *getFunction() { return getStartBB()->getParent(); }

  /// Finds the positive number associated with \p V if it has been mapped.
  /// \param [in] V - the Value to find.
  /// \returns The positive number corresponding to the value.
  /// \returns None if not present.
  Optional<unsigned> getGVN(Value *V) {
    assert(V != nullptr && "Value is a nullptr?");
    DenseMap<Value *, unsigned>::iterator VNIt = ValueToNumber.find(V);
    if (VNIt == ValueToNumber.end())
      return None;
    return VNIt->second;
  }

  /// Finds the Value associate with \p Num if it exists.
  /// \param [in] Num - the number to find.
  /// \returns The Value associated with the number.
  /// \returns None if not present.
  Optional<Value *> fromGVN(unsigned Num) {
    DenseMap<unsigned, Value *>::iterator VNIt = NumberToValue.find(Num);
    if (VNIt == NumberToValue.end())
      return None;
    assert(VNIt->second != nullptr && "Found value is a nullptr!");
    return VNIt->second;
  }

  /// \param RHS -The IRSimilarityCandidate to compare against
  /// \returns true if the IRSimilarityCandidate is occurs after the
  /// IRSimilarityCandidate in the program.
  bool operator<(const IRSimilarityCandidate &RHS) const {
    return getStartIdx() > RHS.getStartIdx();
  }

  using iterator = IRInstructionDataList::iterator;
  iterator begin() const { return iterator(front()); }
  iterator end() const { return std::next(iterator(back())); }
};
} // end namespace IRSimilarity
} // end namespace llvm

#endif // LLVM_ANALYSIS_IRSIMILARITYIDENTIFIER_H
