//===- Transform/Utils/CodeExtractor.h - Code extraction util ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A utility to support extracting code from one function into its own
// stand-alone function.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_CODEEXTRACTOR_H
#define LLVM_TRANSFORMS_UTILS_CODEEXTRACTOR_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include <limits>

namespace llvm {

template <typename PtrType> class SmallPtrSetImpl;
class AllocaInst;
class BasicBlock;
class BlockFrequency;
class BlockFrequencyInfo;
class BranchProbabilityInfo;
class AssumptionCache;
class CallInst;
class DominatorTree;
class Function;
class Instruction;
class Module;
class Type;
class Value;
class StructType;

/// A cache for the CodeExtractor analysis. The operation \ref
/// CodeExtractor::extractCodeRegion is guaranteed not to invalidate this
/// object. This object should conservatively be considered invalid if any
/// other mutating operations on the IR occur.
///
/// Constructing this object is O(n) in the size of the function.
class CodeExtractorAnalysisCache {
  /// The allocas in the function.
  SmallVector<AllocaInst *, 16> Allocas;

  /// Base memory addresses of load/store instructions, grouped by block.
  DenseMap<BasicBlock *, DenseSet<Value *>> BaseMemAddrs;

  /// Blocks which contain instructions which may have unknown side-effects
  /// on memory.
  DenseSet<BasicBlock *> SideEffectingBlocks;

  void findSideEffectInfoForBlock(BasicBlock &BB);

public:
  CodeExtractorAnalysisCache(Function &F);

  /// Get the allocas in the function at the time the analysis was created.
  /// Note that some of these allocas may no longer be present in the function,
  /// due to \ref CodeExtractor::extractCodeRegion.
  ArrayRef<AllocaInst *> getAllocas() const { return Allocas; }

  /// Check whether \p BB contains an instruction thought to load from, store
  /// to, or otherwise clobber the alloca \p Addr.
  bool doesBlockContainClobberOfAddr(BasicBlock &BB, AllocaInst *Addr) const;
};

  /// Utility class for extracting code into a new function.
  ///
  /// This utility provides a simple interface for extracting some sequence of
  /// code into its own function, replacing it with a call to that function. It
  /// also provides various methods to query about the nature and result of
  /// such a transformation.
  ///
  /// The rough algorithm used is:
  /// 1) Find both the inputs and outputs for the extracted region.
  /// 2) Pass the inputs as arguments, remapping them within the extracted
  ///    function to arguments.
  /// 3) Add allocas for any scalar outputs, adding all of the outputs' allocas
  ///    as arguments, and inserting stores to the arguments for any scalars.
  class CodeExtractor {
    using ValueSet = SetVector<Value *>;

    // Various bits of state computed on construction.
    DominatorTree *const DT;
    const bool AggregateArgs;
    BlockFrequencyInfo *BFI;
    BranchProbabilityInfo *BPI;
    AssumptionCache *AC;

    // A block outside of the extraction set where any intermediate
    // allocations will be placed inside. If this is null, allocations
    // will be placed in the entry block of the function.
    BasicBlock *AllocationBlock;

    // If true, varargs functions can be extracted.
    bool AllowVarArgs;

    // Bits of intermediate state computed at various phases of extraction.
    SetVector<BasicBlock *> Blocks;

    /// Lists of blocks that are branched from the code region to be extracted,
    /// also called the exit blocks. Each block is contained at most once. Its
    /// order defines the return value of the extracted function.
    ///
    /// When there is just one (or no) exit block, the return value is
    /// irrelevant.
    ///
    /// When there are exactly two exit blocks, the extracted function returns a
    /// boolean. For ExtractedFuncRetVals[0], it returns 'true'. For
    /// ExtractedFuncRetVals[1] it returns 'false'.
    /// NOTE: Since a boolean is represented by i1, ExtractedFuncRetVals[0]
    ///       returns 1 and ExtractedFuncRetVals[1] returns 0, which opposite
    ///       of the regular pattern below.
    ///
    /// When there are 3 or more exit blocks, leaving the extracted function via
    /// the first block it returns 0. When leaving via the second entry it
    /// returns 1, etc.
    SmallVector<BasicBlock *> ExtractedFuncRetVals;

    // Suffix to use when creating extracted function (appended to the original
    // function name + "."). If empty, the default is to use the entry block
    // label, if non-empty, otherwise "extracted".
    std::string Suffix;

    // If true, the outlined function has aggregate argument in zero address
    // space.
    bool ArgsInZeroAddressSpace;

  public:
    /// Create a code extractor for a sequence of blocks.
    ///
    /// Given a sequence of basic blocks where the first block in the sequence
    /// dominates the rest, prepare a code extractor object for pulling this
    /// sequence out into its new function. When a DominatorTree is also given,
    /// extra checking and transformations are enabled. If AllowVarArgs is true,
    /// vararg functions can be extracted. This is safe, if all vararg handling
    /// code is extracted, including vastart. If AllowAlloca is true, then
    /// extraction of blocks containing alloca instructions would be possible,
    /// however code extractor won't validate whether extraction is legal.
    /// Any new allocations will be placed in the AllocationBlock, unless
    /// it is null, in which case it will be placed in the entry block of
    /// the function from which the code is being extracted.
    /// If ArgsInZeroAddressSpace param is set to true, then the aggregate
    /// param pointer of the outlined function is declared in zero address
    /// space.
    CodeExtractor(ArrayRef<BasicBlock *> BBs, DominatorTree *DT = nullptr,
                  bool AggregateArgs = false, BlockFrequencyInfo *BFI = nullptr,
                  BranchProbabilityInfo *BPI = nullptr,
                  AssumptionCache *AC = nullptr, bool AllowVarArgs = false,
                  bool AllowAlloca = false,
                  BasicBlock *AllocationBlock = nullptr,
                  std::string Suffix = "", bool ArgsInZeroAddressSpace = false);

    /// Perform the extraction, returning the new function.
    ///
    /// Returns zero when called on a CodeExtractor instance where isEligible
    /// returns false.
    Function *extractCodeRegion(const CodeExtractorAnalysisCache &CEAC);

    /// Perform the extraction, returning the new function and providing an
    /// interface to see what was categorized as inputs and outputs.
    ///
    /// \param CEAC - Cache to speed up operations for the CodeExtractor when
    /// hoisting, and extracting lifetime values and assumes.
    /// \param Inputs [out] - filled with  values marked as inputs to the
    /// newly outlined function.
     /// \param Outputs [out] - filled with values marked as outputs to the
    /// newly outlined function.
    /// \returns zero when called on a CodeExtractor instance where isEligible
    /// returns false.
    Function *extractCodeRegion(const CodeExtractorAnalysisCache &CEAC,
                                ValueSet &Inputs, ValueSet &Outputs);

    /// Verify that assumption cache isn't stale after a region is extracted.
    /// Returns true when verifier finds errors. AssumptionCache is passed as
    /// parameter to make this function stateless.
    static bool verifyAssumptionCache(const Function &OldFunc,
                                      const Function &NewFunc,
                                      AssumptionCache *AC);

    /// Test whether this code extractor is eligible.
    ///
    /// Based on the blocks used when constructing the code extractor,
    /// determine whether it is eligible for extraction.
    ///
    /// Checks that varargs handling (with vastart and vaend) is only done in
    /// the outlined blocks.
    bool isEligible() const;

    /// Compute the set of input values and output values for the code.
    ///
    /// These can be used either when performing the extraction or to evaluate
    /// the expected size of a call to the extracted function. Note that this
    /// work cannot be cached between the two as once we decide to extract
    /// a code sequence, that sequence is modified, including changing these
    /// sets, before extraction occurs. These modifications won't have any
    /// significant impact on the cost however.
    void findInputsOutputs(ValueSet &Inputs, ValueSet &Outputs,
                           const ValueSet &Allocas,
                           bool CollectGlobalInputs = false) const;

    /// Check if life time marker nodes can be hoisted/sunk into the outline
    /// region.
    ///
    /// Returns true if it is safe to do the code motion.
    bool
    isLegalToShrinkwrapLifetimeMarkers(const CodeExtractorAnalysisCache &CEAC,
                                       Instruction *AllocaAddr) const;

    /// Find the set of allocas whose life ranges are contained within the
    /// outlined region.
    ///
    /// Allocas which have life_time markers contained in the outlined region
    /// should be pushed to the outlined function. The address bitcasts that
    /// are used by the lifetime markers are also candidates for shrink-
    /// wrapping. The instructions that need to be sunk are collected in
    /// 'Allocas'.
    void findAllocas(const CodeExtractorAnalysisCache &CEAC,
                     ValueSet &SinkCands, ValueSet &HoistCands,
                     BasicBlock *&ExitBlock) const;

    /// Find or create a block within the outline region for placing hoisted
    /// code.
    ///
    /// CommonExitBlock is block outside the outline region. It is the common
    /// successor of blocks inside the region. If there exists a single block
    /// inside the region that is the predecessor of CommonExitBlock, that block
    /// will be returned. Otherwise CommonExitBlock will be split and the
    /// original block will be added to the outline region.
    BasicBlock *findOrCreateBlockForHoisting(BasicBlock *CommonExitBlock);

    /// Exclude a value from aggregate argument passing when extracting a code
    /// region, passing it instead as a scalar.
    void excludeArgFromAggregate(Value *Arg);

  private:
    struct LifetimeMarkerInfo {
      bool SinkLifeStart = false;
      bool HoistLifeEnd = false;
      Instruction *LifeStart = nullptr;
      Instruction *LifeEnd = nullptr;
    };

    ValueSet ExcludeArgsFromAggregate;

    LifetimeMarkerInfo
    getLifetimeMarkers(const CodeExtractorAnalysisCache &CEAC,
                       Instruction *Addr, BasicBlock *ExitBlock) const;

    /// Updates the list of SwitchCases (corresponding to exit blocks) after
    /// changes of the control flow or the Blocks list.
    void computeExtractedFuncRetVals();

    /// Return the type used for the return code of the extracted function to
    /// indicate which exit block to jump to.
    Type *getSwitchType();

    void severSplitPHINodesOfEntry(BasicBlock *&Header);
    void severSplitPHINodesOfExits();
    void splitReturnBlocks();

    void moveCodeToFunction(Function *newFunction);

    void calculateNewCallTerminatorWeights(
        BasicBlock *CodeReplacer,
        const DenseMap<BasicBlock *, BlockFrequency> &ExitWeights,
        BranchProbabilityInfo *BPI);

    /// Normalizes the control flow of the extracted regions, such as ensuring
    /// that the extracted region does not contain a return instruction.
    void normalizeCFGForExtraction(BasicBlock *&header);

    /// Generates the function declaration for the function containing the
    /// extracted code.
    Function *constructFunctionDeclaration(const ValueSet &inputs,
                                           const ValueSet &outputs,
                                           BlockFrequency EntryFreq,
                                           const Twine &Name,
                                           ValueSet &StructValues,
                                           StructType *&StructTy);

    /// Generates the code for the extracted function. That is: a prolog, the
    /// moved or copied code from the original function, and epilogs for each
    /// exit.
    void emitFunctionBody(const ValueSet &inputs, const ValueSet &outputs,
                          const ValueSet &StructValues, Function *newFunction,
                          StructType *StructArgTy, BasicBlock *header,
                          const ValueSet &SinkingCands);

    /// Generates a Basic Block that calls the extracted function.
    CallInst *emitReplacerCall(const ValueSet &inputs, const ValueSet &outputs,
                               const ValueSet &StructValues,
                               Function *newFunction, StructType *StructArgTy,
                               Function *oldFunction, BasicBlock *ReplIP,
                               BlockFrequency EntryFreq,
                               ArrayRef<Value *> LifetimesStart,
                               std::vector<Value *> &Reloads);

    /// Connects the basic block containing the call to the extracted function
    /// into the original function's control flow.
    void insertReplacerCall(
        Function *oldFunction, BasicBlock *header, BasicBlock *codeReplacer,
        const ValueSet &outputs, ArrayRef<Value *> Reloads,
        const DenseMap<BasicBlock *, BlockFrequency> &ExitWeights);
  };

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_CODEEXTRACTOR_H
