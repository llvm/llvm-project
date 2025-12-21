//===-- AMDGPUTDMOptimization.cpp - TDM Descriptor Optimization ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass optimizes Tensor Data Movement (TDM) descriptor creation patterns.
// It identifies insertelement chains that create descriptors and transforms
// them to use alloca+field updates, which SROA later optimizes to
// INSERT_SUBREG.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUSubtarget.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-tdm-optimization"

static cl::opt<unsigned>
    TDMOptBenefitThreshold("amdgpu-tdm-opt-threshold", cl::Hidden, cl::init(10),
                           cl::desc("Minimum optimization benefit threshold "
                                    "for TDM descriptor optimization"));

namespace llvm {
void initializeAMDGPUTDMOptimizationPass(PassRegistry &);
}

namespace {

//===----------------------------------------------------------------------===//
// Pattern Detection Data Structures
//===----------------------------------------------------------------------===//

/// Represents a single descriptor creation pattern
struct DescriptorPattern {
  Type *DescType;   ///< <4 x i32> or <8 x i32>
  Value *BaseValue; ///< Base template (constant or computed)
  SmallVector<InsertElementInst *, 8>
      Chain; ///< Chain of insertelement instructions
  SmallVector<unsigned, 8> VariableFields; ///< Fields that change
  SmallVector<unsigned, 8> ConstantFields; ///< Fields that stay constant
  BasicBlock *Location;                    ///< Where the pattern is located
  Loop *ContainingLoop; ///< Loop containing this pattern (if any)

  /// Calculate field reuse ratio (constant fields / total fields)
  float getFieldReuseRatio() const {
    unsigned totalFields = cast<FixedVectorType>(DescType)->getNumElements();
    return (float)ConstantFields.size() / totalFields;
  }

  /// Check if this pattern is worth optimizing
  /// Note: This is a preliminary check. The final decision also considers
  /// whether multiple patterns can be chained together (see groupSimilarPatterns).
  bool isWorthOptimizing() const {
    // Optimize if significant field reuse potential
    if (getFieldReuseRatio() >= 0.5f)
      return true;

    // Optimize address descriptors (common case)
    if (isAddressDescriptor() && ConstantFields.size() >= 1)
      return true;

    return false;
  }

  /// Check if this is an address descriptor (<4 x i32>)
  bool isAddressDescriptor() const {
    auto *VecTy = cast<FixedVectorType>(DescType);
    return VecTy->getNumElements() == 4 &&
           VecTy->getElementType()->isIntegerTy(32);
  }

  /// Check if this is a tensor descriptor (<8 x i32>)
  bool isTensorDescriptor() const {
    auto *VecTy = cast<FixedVectorType>(DescType);
    return VecTy->getNumElements() == 8 &&
           VecTy->getElementType()->isIntegerTy(32);
  }
};

/// Groups similar descriptor patterns for optimization
struct DescriptorGroup {
  SmallVector<DescriptorPattern, 4> Patterns;
  Type *SharedType;
  Value *SharedBase; ///< Common base value (if any)
  SmallVector<unsigned, 8> SharedConstantFields;

  /// Calculate total optimization benefit
  unsigned getOptimizationBenefit() const {
    unsigned benefit = 0;
    for (const auto &pattern : Patterns) {
      // Base benefit from field reuse
      benefit += pattern.ConstantFields.size() * 2;

      // Extra benefit for loop patterns
      if (pattern.ContainingLoop)
        benefit *= 5;
    }
    return benefit;
  }
};

//===----------------------------------------------------------------------===//
// AMDGPUTDMOptimization Pass
//===----------------------------------------------------------------------===//

class AMDGPUTDMOptimization : public FunctionPass {
private:
  LoopInfo *LI = nullptr;

  /// Detected patterns in the function
  SmallVector<DescriptorPattern, 16> DetectedPatterns;

  /// Groups of optimizable patterns
  SmallVector<DescriptorGroup, 8> OptimizationGroups;

public:
  static char ID;

  AMDGPUTDMOptimization() : FunctionPass(ID) {
    initializeAMDGPUTDMOptimizationPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  /// Main optimization phases
  bool detectDescriptorPatterns(Function &F);
  void groupSimilarPatterns();
  bool transformPatterns(Function &F);

  /// Pattern detection helpers
  bool isDescriptorType(Type *Ty) const;
  DescriptorPattern analyzeInsertChain(InsertElementInst *FinalInsert);
  Value *extractBaseValue(const DescriptorPattern &Pattern);

  /// Transformation helpers
  bool transformDescriptorGroup(DescriptorGroup &Group, Function &F);
  Value *createSharedStorage(DescriptorGroup &Group, IRBuilder<> &Builder);
  void transformSinglePattern(DescriptorPattern &Pattern, Value *SharedStorage,
                              IRBuilder<> &Builder);

  /// Utility functions
  Loop *getContainingLoop(BasicBlock *BB);
  bool arePatternsSimilar(const DescriptorPattern &A,
                          const DescriptorPattern &B);
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

bool AMDGPUTDMOptimization::runOnFunction(Function &F) {
  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();

  LLVM_DEBUG(dbgs() << "Running TDM optimization on function: " << F.getName()
                    << "\n");

  // Phase 1: Detect descriptor patterns
  if (!detectDescriptorPatterns(F)) {
    LLVM_DEBUG(dbgs() << "No descriptor patterns found\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << "Found " << DetectedPatterns.size()
                    << " descriptor patterns\n");

  // Phase 2: Group similar patterns for optimization
  groupSimilarPatterns();

  LLVM_DEBUG(dbgs() << "Created " << OptimizationGroups.size()
                    << " optimization groups\n");

  // Phase 3: Transform patterns
  bool Changed = transformPatterns(F);

  // Cleanup for next function
  DetectedPatterns.clear();
  OptimizationGroups.clear();

  return Changed;
}

void AMDGPUTDMOptimization::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LoopInfoWrapperPass>();
  AU.setPreservesCFG();
}

//===----------------------------------------------------------------------===//
// Pattern Detection
//===----------------------------------------------------------------------===//

bool AMDGPUTDMOptimization::detectDescriptorPatterns(Function &F) {
  bool FoundPatterns = false;

  // Scan function for insertelement instructions that create descriptors
  for (auto &BB : F) {
    for (auto &I : BB) {
      auto *IE = dyn_cast<InsertElementInst>(&I);
      if (!IE || !isDescriptorType(IE->getType()))
        continue;

      // Check if this is the final insert in a descriptor creation chain
      // Only optimize chains that end at call instructions (descriptor consumers)
      if (!IE->hasOneUse())
        continue;

      User *SingleUser = *IE->user_begin();
      // Skip if user is another insertelement (chain continues)
      // Skip if user is NOT a call instruction (e.g., shufflevector is intermediate)
      if (isa<InsertElementInst>(SingleUser) || !isa<CallInst>(SingleUser))
        continue;

      // Analyze the complete chain
      DescriptorPattern Pattern = analyzeInsertChain(IE);
      if (Pattern.Chain.empty())
        continue;

      // Check if worth optimizing
      if (!Pattern.isWorthOptimizing()) {
        LLVM_DEBUG(
            dbgs() << "Pattern not worth optimizing: field reuse ratio = "
                   << Pattern.getFieldReuseRatio() << "\n");
        continue;
      }

      LLVM_DEBUG(
          dbgs() << "Found optimizable pattern: "
                 << (Pattern.isAddressDescriptor() ? "Address" : "Tensor")
                 << " descriptor with " << Pattern.ConstantFields.size()
                 << " constant fields\n");

      DetectedPatterns.push_back(std::move(Pattern));
      FoundPatterns = true;
    }
  }

  return FoundPatterns;
}

bool AMDGPUTDMOptimization::isDescriptorType(Type *Ty) const {
  auto *VecTy = dyn_cast<FixedVectorType>(Ty);
  if (!VecTy || !VecTy->getElementType()->isIntegerTy(32))
    return false;

  unsigned NumElements = VecTy->getNumElements();
  return NumElements == 4 || NumElements == 8; // Address or tensor descriptors
}

DescriptorPattern
AMDGPUTDMOptimization::analyzeInsertChain(InsertElementInst *FinalInsert) {
  DescriptorPattern Pattern;
  Pattern.DescType = FinalInsert->getType();
  Pattern.Location = FinalInsert->getParent();
  Pattern.ContainingLoop = getContainingLoop(Pattern.Location);

  // Trace back the insertelement chain
  SmallVector<InsertElementInst *, 8> Chain;
  Value *CurrentVal = FinalInsert;

  while (auto *IE = dyn_cast<InsertElementInst>(CurrentVal)) {
    Chain.push_back(IE);
    CurrentVal = IE->getOperand(0); // Vector being inserted into
  }

  // Reverse to get forward order
  std::reverse(Chain.begin(), Chain.end());
  Pattern.Chain = Chain;

  // Extract base value (the initial vector)
  Pattern.BaseValue = extractBaseValue(Pattern);

  // Analyze which fields are constant vs variable
  unsigned NumElements =
      cast<FixedVectorType>(Pattern.DescType)->getNumElements();
  SmallBitVector FieldSet(NumElements, false);

  for (auto *IE : Chain) {
    if (auto *CI = dyn_cast<ConstantInt>(IE->getOperand(2))) {
      unsigned Idx = CI->getZExtValue();
      if (Idx < NumElements) {
        FieldSet.set(Idx);
        Pattern.VariableFields.push_back(Idx);
      }
    }
  }

  // Fields not in chain are constant
  for (unsigned i = 0; i < NumElements; ++i) {
    if (!FieldSet[i])
      Pattern.ConstantFields.push_back(i);
  }

  return Pattern;
}

Value *
AMDGPUTDMOptimization::extractBaseValue(const DescriptorPattern &Pattern) {
  if (Pattern.Chain.empty())
    return nullptr;

  // Get the vector being inserted into by the first insert
  Value *Base = Pattern.Chain[0]->getOperand(0);

  // If base is a constant vector or another recognizable pattern, return it
  if (isa<Constant>(Base))
    return Base;

  // For shufflevector results, we might want to trace further back
  if (auto *SV = dyn_cast<ShuffleVectorInst>(Base))
    return SV; // Keep shufflevector as base for now

  return Base;
}

Loop *AMDGPUTDMOptimization::getContainingLoop(BasicBlock *BB) {
  return LI ? LI->getLoopFor(BB) : nullptr;
}

//===----------------------------------------------------------------------===//
// Pattern Grouping
//===----------------------------------------------------------------------===//

void AMDGPUTDMOptimization::groupSimilarPatterns() {
  // Simple grouping strategy: group by type and base similarity
  for (auto &Pattern : DetectedPatterns) {
    bool Added = false;

    // Try to add to existing group
    for (auto &Group : OptimizationGroups) {
      if (Group.SharedType == Pattern.DescType &&
          arePatternsSimilar(Group.Patterns[0], Pattern)) {
        Group.Patterns.push_back(Pattern);
        Added = true;
        break;
      }
    }

    // Create new group if needed
    if (!Added) {
      DescriptorGroup NewGroup;
      NewGroup.SharedType = Pattern.DescType;
      NewGroup.SharedBase = Pattern.BaseValue;
      NewGroup.Patterns.push_back(Pattern);
      OptimizationGroups.push_back(std::move(NewGroup));
    }
  }

  // Remove groups that don't meet optimization criteria
  OptimizationGroups.erase(
      std::remove_if(OptimizationGroups.begin(), OptimizationGroups.end(),
                     [](const DescriptorGroup &Group) {
                       // Skip groups that don't meet benefit threshold
                       if (Group.getOptimizationBenefit() < TDMOptBenefitThreshold)
                         return true;

                       // CRITICAL: Skip groups where the base is a non-constant SSA value.
                       // Non-constant bases (like shufflevector results) are already
                       // available as SSA values that all descriptors can share directly.
                       // Chaining would only create unnecessary data dependencies between
                       // descriptors without saving any instructions.
                       //
                       // We ONLY benefit from chaining when the base is a constant that
                       // needs to be materialized repeatedly (REG_SEQUENCE from constants).
                       if (Group.SharedBase && !isa<Constant>(Group.SharedBase)) {
                         LLVM_DEBUG(dbgs() << "Skipping group with non-constant SSA base - "
                                           << "no materialization savings\n");
                         return true;
                       }

                       // Skip groups with only ONE pattern in a loop.
                       // The optimization creates alloca + store/load which SROA
                       // converts to a phi node. For a single pattern, this is
                       // counterproductive - it creates loop-carried dependencies
                       // for constant fields that should stay in SGPRs.
                       // We only benefit when there are MULTIPLE patterns to chain.
                       if (Group.Patterns.size() == 1 &&
                           Group.Patterns[0].ContainingLoop) {
                         LLVM_DEBUG(dbgs() << "Skipping single pattern in loop - "
                                           << "no chaining benefit\n");
                         return true;
                       }

                       return false;
                     }),
      OptimizationGroups.end());
}

bool AMDGPUTDMOptimization::arePatternsSimilar(const DescriptorPattern &A,
                                               const DescriptorPattern &B) {
  // Patterns are similar if they have same type and similar field usage
  if (A.DescType != B.DescType)
    return false;

  // Check if constant fields overlap significantly
  SmallBitVector AConstants(
      cast<FixedVectorType>(A.DescType)->getNumElements());
  SmallBitVector BConstants(
      cast<FixedVectorType>(B.DescType)->getNumElements());

  for (unsigned Field : A.ConstantFields)
    AConstants.set(Field);
  for (unsigned Field : B.ConstantFields)
    BConstants.set(Field);

  // Count overlapping constant fields
  auto Intersection = AConstants & BConstants;
  unsigned OverlapCount = Intersection.count();
  unsigned TotalConstants = std::max(AConstants.count(), BConstants.count());

  return TotalConstants > 0 && (float)OverlapCount / TotalConstants >= 0.5f;
}

//===----------------------------------------------------------------------===//
// Pattern Transformation
//===----------------------------------------------------------------------===//

bool AMDGPUTDMOptimization::transformPatterns(Function &F) {
  bool Changed = false;

  for (auto &Group : OptimizationGroups) {
    LLVM_DEBUG(dbgs() << "Transforming group with " << Group.Patterns.size()
                      << " patterns, benefit = "
                      << Group.getOptimizationBenefit() << "\n");

    if (transformDescriptorGroup(Group, F))
      Changed = true;
  }

  return Changed;
}

bool AMDGPUTDMOptimization::transformDescriptorGroup(DescriptorGroup &Group,
                                                     Function &F) {
  if (Group.Patterns.empty())
    return false;

  // Find the best location to place shared storage
  BasicBlock *StorageLocation = Group.Patterns[0].Location;

  // If patterns are in a loop, try to hoist storage outside loop
  if (auto *Loop = Group.Patterns[0].ContainingLoop) {
    if (auto *Preheader = Loop->getLoopPreheader()) {
      StorageLocation = Preheader;
      LLVM_DEBUG(dbgs() << "Hoisting storage outside loop\n");
    }
  }

  // Create shared storage at the beginning of the storage block
  IRBuilder<> Builder(&StorageLocation->front());
  Value *SharedStorage = createSharedStorage(Group, Builder);

  if (!SharedStorage)
    return false;

  // If base is non-constant (e.g., shufflevector result), store it just before
  // the first pattern's chain. This ensures the base value is defined before
  // we try to store it.
  if (Group.SharedBase && !isa<Constant>(Group.SharedBase)) {
    // Insert the store just before the first insertelement in the first pattern
    IRBuilder<> BaseBuilder(Group.Patterns[0].Chain.front());
    BaseBuilder.CreateStore(Group.SharedBase, SharedStorage);
    LLVM_DEBUG(dbgs() << "Stored non-constant base before first pattern\n");
  }

  // Transform each pattern in the group
  for (auto &Pattern : Group.Patterns) {
    IRBuilder<> PatternBuilder(Pattern.Chain.back());
    transformSinglePattern(Pattern, SharedStorage, PatternBuilder);
  }

  return true;
}

Value *AMDGPUTDMOptimization::createSharedStorage(DescriptorGroup &Group,
                                                  IRBuilder<> &Builder) {
  // Create alloca in address space 5 (AMDGPU private memory)
  auto *StorageType = Group.SharedType;
  auto *Storage = Builder.CreateAlloca(
      StorageType, /*AddrSpace=*/5, /*ArraySize=*/nullptr, "tdm_desc_storage");

  // Initialize with base template if available
  if (Group.SharedBase) {
    auto *BaseConstant = dyn_cast<Constant>(Group.SharedBase);
    if (BaseConstant) {
      Builder.CreateStore(BaseConstant, Storage);
      LLVM_DEBUG(dbgs() << "Initialized storage with constant base\n");
    }
  }

  return Storage;
}

void AMDGPUTDMOptimization::transformSinglePattern(DescriptorPattern &Pattern,
                                                   Value *SharedStorage,
                                                   IRBuilder<> &Builder) {
  // Create field pointers for variable fields
  SmallVector<Value *, 8> FieldPointers;
  for (unsigned FieldIdx : Pattern.VariableFields) {
    Value *FieldPtr =
        Builder.CreateGEP(Pattern.DescType, SharedStorage,
                          {Builder.getInt32(0), Builder.getInt32(FieldIdx)},
                          "tdm_field_" + Twine(FieldIdx) + "_ptr");
    FieldPointers.push_back(FieldPtr);
  }

  // Update variable fields with values from the original chain
  for (unsigned i = 0;
       i < Pattern.VariableFields.size() && i < Pattern.Chain.size(); ++i) {
    auto *InsertInst = Pattern.Chain[i];
    Value *NewValue = InsertInst->getOperand(1); // Value being inserted
    Builder.CreateStore(NewValue, FieldPointers[i]);
  }

  // Replace final result with load from shared storage
  Value *OptimizedDescriptor =
      Builder.CreateLoad(Pattern.DescType, SharedStorage, "tdm_optimized_desc");

  // Replace all uses of the final insert with the load
  Pattern.Chain.back()->replaceAllUsesWith(OptimizedDescriptor);

  // Clean up the now-dead insertelement chain in reverse order
  // (each instruction uses the previous one, so delete from end to start)
  for (auto It = Pattern.Chain.rbegin(); It != Pattern.Chain.rend(); ++It) {
    InsertElementInst *IE = *It;
    if (IE->use_empty()) {
      IE->eraseFromParent();
    }
  }

  LLVM_DEBUG(dbgs() << "Transformed pattern with "
                    << Pattern.VariableFields.size() << " variable fields\n");
}

} // end anonymous namespace

char AMDGPUTDMOptimization::ID = 0;

INITIALIZE_PASS_BEGIN(AMDGPUTDMOptimization, DEBUG_TYPE,
                      "AMDGPU TDM Descriptor Optimization", false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(AMDGPUTDMOptimization, DEBUG_TYPE,
                    "AMDGPU TDM Descriptor Optimization", false, false)

namespace llvm {
FunctionPass *createAMDGPUTDMOptimizationPass() {
  return new AMDGPUTDMOptimization();
}
} // namespace llvm