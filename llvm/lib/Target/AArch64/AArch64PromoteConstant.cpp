//=- AArch64PromoteConstant.cpp --- Promote constant to global for AArch64 -==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the AArch64PromoteConstant pass which promotes constants
// to global variables when this is likely to be more efficient. Currently only
// types related to constant vector (i.e., constant vector, array of constant
// vectors, constant structure with a constant vector field, etc.) are promoted
// to global variables. Constant vectors are likely to be lowered in target
// constant pool during instruction selection already; therefore, the access
// will remain the same (memory load), but the structure types are not split
// into different constant pool accesses for each field. A bonus side effect is
// that created globals may be merged by the global merge pass.
//
// FIXME: This pass may be useful for other targets too.
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-promote-const"

// Stress testing mode - disable heuristics.
static cl::opt<bool> Stress("aarch64-stress-promote-const", cl::Hidden,
                            cl::desc("Promote all vector constants"));

STATISTIC(NumPromoted, "Number of promoted constants");
STATISTIC(NumPromotedUses, "Number of promoted constants uses");

//===----------------------------------------------------------------------===//
//                       AArch64PromoteConstant
//===----------------------------------------------------------------------===//

namespace {
/// Promotes interesting constant into global variables.
/// The motivating example is:
/// static const uint16_t TableA[32] = {
///   41944, 40330, 38837, 37450, 36158, 34953, 33826, 32768,
///   31776, 30841, 29960, 29128, 28340, 27595, 26887, 26215,
///   25576, 24967, 24386, 23832, 23302, 22796, 22311, 21846,
///   21400, 20972, 20561, 20165, 19785, 19419, 19066, 18725,
/// };
///
/// uint8x16x4_t LoadStatic(void) {
///   uint8x16x4_t ret;
///   ret.val[0] = vld1q_u16(TableA +  0);
///   ret.val[1] = vld1q_u16(TableA +  8);
///   ret.val[2] = vld1q_u16(TableA + 16);
///   ret.val[3] = vld1q_u16(TableA + 24);
///   return ret;
/// }
///
/// The constants in this example are folded into the uses. Thus, 4 different
/// constants are created.
///
/// As their type is vector the cheapest way to create them is to load them
/// for the memory.
///
/// Therefore the final assembly final has 4 different loads. With this pass
/// enabled, only one load is issued for the constants.
class AArch64PromoteConstant : public ModulePass {

public:
  static char ID;
  AArch64PromoteConstant() : ModulePass(ID) {}

  const char *getPassName() const override { return "AArch64 Promote Constant"; }

  /// Iterate over the functions and promote the interesting constants into
  /// global variables with module scope.
  bool runOnModule(Module &M) override {
    DEBUG(dbgs() << getPassName() << '\n');
    bool Changed = false;
    for (auto &MF : M) {
      Changed |= runOnFunction(MF);
    }
    return Changed;
  }

private:
  /// Look for interesting constants used within the given function.
  /// Promote them into global variables, load these global variables within
  /// the related function, so that the number of inserted load is minimal.
  bool runOnFunction(Function &F);

  // This transformation requires dominator info
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
  }

  /// Type to store a list of Uses.
  typedef SmallVector<Use *, 4> Uses;
  /// Map an insertion point to all the uses it dominates.
  typedef DenseMap<Instruction *, Uses> InsertionPoints;
  /// Map a function to the required insertion point of load for a
  /// global variable.
  typedef DenseMap<Function *, InsertionPoints> InsertionPointsPerFunc;

  /// Find the closest point that dominates the given Use.
  Instruction *findInsertionPoint(Use &Use);

  /// Check if the given insertion point is dominated by an existing
  /// insertion point.
  /// If true, the given use is added to the list of dominated uses for
  /// the related existing point.
  /// \param NewPt the insertion point to be checked
  /// \param Use the use to be added into the list of dominated uses
  /// \param InsertPts existing insertion points
  /// \pre NewPt and all instruction in InsertPts belong to the same function
  /// \return true if one of the insertion point in InsertPts dominates NewPt,
  ///         false otherwise
  bool isDominated(Instruction *NewPt, Use &Use, InsertionPoints &InsertPts);

  /// Check if the given insertion point can be merged with an existing
  /// insertion point in a common dominator.
  /// If true, the given use is added to the list of the created insertion
  /// point.
  /// \param NewPt the insertion point to be checked
  /// \param Use the use to be added into the list of dominated uses
  /// \param InsertPts existing insertion points
  /// \pre NewPt and all instruction in InsertPts belong to the same function
  /// \pre isDominated returns false for the exact same parameters.
  /// \return true if it exists an insertion point in InsertPts that could
  ///         have been merged with NewPt in a common dominator,
  ///         false otherwise
  bool tryAndMerge(Instruction *NewPt, Use &Use, InsertionPoints &InsertPts);

  /// Compute the minimal insertion points to dominates all the interesting
  /// uses of value.
  /// Insertion points are group per function and each insertion point
  /// contains a list of all the uses it dominates within the related function
  /// \param Val constant to be examined
  /// \param[out] InsPtsPerFunc output storage of the analysis
  void computeInsertionPoints(Constant *Val,
                              InsertionPointsPerFunc &InsPtsPerFunc);

  /// Insert a definition of a new global variable at each point contained in
  /// InsPtsPerFunc and update the related uses (also contained in
  /// InsPtsPerFunc).
  bool insertDefinitions(Constant *Cst, InsertionPointsPerFunc &InsPtsPerFunc);

  /// Compute the minimal insertion points to dominate all the interesting
  /// uses of Val and insert a definition of a new global variable
  /// at these points.
  /// Also update the uses of Val accordingly.
  /// Currently a use of Val is considered interesting if:
  /// - Val is not UndefValue
  /// - Val is not zeroinitialized
  /// - Replacing Val per a load of a global variable is valid.
  /// \see shouldConvert for more details
  bool computeAndInsertDefinitions(Constant *Val);

  /// Promote the given constant into a global variable if it is expected to
  /// be profitable.
  /// \return true if Cst has been promoted
  bool promoteConstant(Constant *Cst);

  /// Transfer the list of dominated uses of IPI to NewPt in InsertPts.
  /// Append Use to this list and delete the entry of IPI in InsertPts.
  static void appendAndTransferDominatedUses(Instruction *NewPt, Use &Use,
                                             InsertionPoints::iterator &IPI,
                                             InsertionPoints &InsertPts) {
    // Record the dominated use.
    IPI->second.push_back(&Use);
    // Transfer the dominated uses of IPI to NewPt
    // Inserting into the DenseMap may invalidate existing iterator.
    // Keep a copy of the key to find the iterator to erase.  Keep a copy of the
    // value so that we don't have to dereference IPI->second.
    Instruction *OldInstr = IPI->first;
    Uses OldUses = std::move(IPI->second);
    InsertPts[NewPt] = std::move(OldUses);
    // Erase IPI.
    InsertPts.erase(OldInstr);
  }
};
} // end anonymous namespace

char AArch64PromoteConstant::ID = 0;

namespace llvm {
void initializeAArch64PromoteConstantPass(PassRegistry &);
}

INITIALIZE_PASS_BEGIN(AArch64PromoteConstant, "aarch64-promote-const",
                      "AArch64 Promote Constant Pass", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(AArch64PromoteConstant, "aarch64-promote-const",
                    "AArch64 Promote Constant Pass", false, false)

ModulePass *llvm::createAArch64PromoteConstantPass() {
  return new AArch64PromoteConstant();
}

/// Check if the given type uses a vector type.
static bool isConstantUsingVectorTy(const Type *CstTy) {
  if (CstTy->isVectorTy())
    return true;
  if (CstTy->isStructTy()) {
    for (unsigned EltIdx = 0, EndEltIdx = CstTy->getStructNumElements();
         EltIdx < EndEltIdx; ++EltIdx)
      if (isConstantUsingVectorTy(CstTy->getStructElementType(EltIdx)))
        return true;
  } else if (CstTy->isArrayTy())
    return isConstantUsingVectorTy(CstTy->getArrayElementType());
  return false;
}

/// Check if the given use (Instruction + OpIdx) of Cst should be converted into
/// a load of a global variable initialized with Cst.
/// A use should be converted if it is legal to do so.
/// For instance, it is not legal to turn the mask operand of a shuffle vector
/// into a load of a global variable.
static bool shouldConvertUse(const Constant *Cst, const Instruction *Instr,
                             unsigned OpIdx) {
  // shufflevector instruction expects a const for the mask argument, i.e., the
  // third argument. Do not promote this use in that case.
  if (isa<const ShuffleVectorInst>(Instr) && OpIdx == 2)
    return false;

  // extractvalue instruction expects a const idx.
  if (isa<const ExtractValueInst>(Instr) && OpIdx > 0)
    return false;

  // extractvalue instruction expects a const idx.
  if (isa<const InsertValueInst>(Instr) && OpIdx > 1)
    return false;

  if (isa<const AllocaInst>(Instr) && OpIdx > 0)
    return false;

  // Alignment argument must be constant.
  if (isa<const LoadInst>(Instr) && OpIdx > 0)
    return false;

  // Alignment argument must be constant.
  if (isa<const StoreInst>(Instr) && OpIdx > 1)
    return false;

  // Index must be constant.
  if (isa<const GetElementPtrInst>(Instr) && OpIdx > 0)
    return false;

  // Personality function and filters must be constant.
  // Give up on that instruction.
  if (isa<const LandingPadInst>(Instr))
    return false;

  // Switch instruction expects constants to compare to.
  if (isa<const SwitchInst>(Instr))
    return false;

  // Expected address must be a constant.
  if (isa<const IndirectBrInst>(Instr))
    return false;

  // Do not mess with intrinsics.
  if (isa<const IntrinsicInst>(Instr))
    return false;

  // Do not mess with inline asm.
  const CallInst *CI = dyn_cast<const CallInst>(Instr);
  if (CI && isa<const InlineAsm>(CI->getCalledValue()))
    return false;

  return true;
}

/// Check if the given Cst should be converted into
/// a load of a global variable initialized with Cst.
/// A constant should be converted if it is likely that the materialization of
/// the constant will be tricky. Thus, we give up on zero or undef values.
///
/// \todo Currently, accept only vector related types.
/// Also we give up on all simple vector type to keep the existing
/// behavior. Otherwise, we should push here all the check of the lowering of
/// BUILD_VECTOR. By giving up, we lose the potential benefit of merging
/// constant via global merge and the fact that the same constant is stored
/// only once with this method (versus, as many function that uses the constant
/// for the regular approach, even for float).
/// Again, the simplest solution would be to promote every
/// constant and rematerialize them when they are actually cheap to create.
static bool shouldConvert(const Constant *Cst) {
  if (isa<const UndefValue>(Cst))
    return false;

  // FIXME: In some cases, it may be interesting to promote in memory
  // a zero initialized constant.
  // E.g., when the type of Cst require more instructions than the
  // adrp/add/load sequence or when this sequence can be shared by several
  // instances of Cst.
  // Ideally, we could promote this into a global and rematerialize the constant
  // when it was a bad idea.
  if (Cst->isZeroValue())
    return false;

  if (Stress)
    return true;

  // FIXME: see function \todo
  if (Cst->getType()->isVectorTy())
    return false;
  return isConstantUsingVectorTy(Cst->getType());
}

Instruction *AArch64PromoteConstant::findInsertionPoint(Use &Use) {
  Instruction *User = cast<Instruction>(Use.getUser());

  // If this user is a phi, the insertion point is in the related
  // incoming basic block.
  if (PHINode *PhiInst = dyn_cast<PHINode>(User))
    return PhiInst->getIncomingBlock(Use.getOperandNo())->getTerminator();

  return User;
}

bool AArch64PromoteConstant::isDominated(Instruction *NewPt, Use &Use,
                                         InsertionPoints &InsertPts) {

  DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>(
      *NewPt->getParent()->getParent()).getDomTree();

  // Traverse all the existing insertion points and check if one is dominating
  // NewPt. If it is, remember that.
  for (auto &IPI : InsertPts) {
    if (NewPt == IPI.first || DT.dominates(IPI.first, NewPt) ||
        // When IPI.first is a terminator instruction, DT may think that
        // the result is defined on the edge.
        // Here we are testing the insertion point, not the definition.
        (IPI.first->getParent() != NewPt->getParent() &&
         DT.dominates(IPI.first->getParent(), NewPt->getParent()))) {
      // No need to insert this point. Just record the dominated use.
      DEBUG(dbgs() << "Insertion point dominated by:\n");
      DEBUG(IPI.first->print(dbgs()));
      DEBUG(dbgs() << '\n');
      IPI.second.push_back(&Use);
      return true;
    }
  }
  return false;
}

bool AArch64PromoteConstant::tryAndMerge(Instruction *NewPt, Use &Use,
                                         InsertionPoints &InsertPts) {
  DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>(
      *NewPt->getParent()->getParent()).getDomTree();
  BasicBlock *NewBB = NewPt->getParent();

  // Traverse all the existing insertion point and check if one is dominated by
  // NewPt and thus useless or can be combined with NewPt into a common
  // dominator.
  for (InsertionPoints::iterator IPI = InsertPts.begin(),
                                 EndIPI = InsertPts.end();
       IPI != EndIPI; ++IPI) {
    BasicBlock *CurBB = IPI->first->getParent();
    if (NewBB == CurBB) {
      // Instructions are in the same block.
      // By construction, NewPt is dominating the other.
      // Indeed, isDominated returned false with the exact same arguments.
      DEBUG(dbgs() << "Merge insertion point with:\n");
      DEBUG(IPI->first->print(dbgs()));
      DEBUG(dbgs() << "\nat considered insertion point.\n");
      appendAndTransferDominatedUses(NewPt, Use, IPI, InsertPts);
      return true;
    }

    // Look for a common dominator
    BasicBlock *CommonDominator = DT.findNearestCommonDominator(NewBB, CurBB);
    // If none exists, we cannot merge these two points.
    if (!CommonDominator)
      continue;

    if (CommonDominator != NewBB) {
      // By construction, the CommonDominator cannot be CurBB.
      assert(CommonDominator != CurBB &&
             "Instruction has not been rejected during isDominated check!");
      // Take the last instruction of the CommonDominator as insertion point
      NewPt = CommonDominator->getTerminator();
    }
    // else, CommonDominator is the block of NewBB, hence NewBB is the last
    // possible insertion point in that block.
    DEBUG(dbgs() << "Merge insertion point with:\n");
    DEBUG(IPI->first->print(dbgs()));
    DEBUG(dbgs() << '\n');
    DEBUG(NewPt->print(dbgs()));
    DEBUG(dbgs() << '\n');
    appendAndTransferDominatedUses(NewPt, Use, IPI, InsertPts);
    return true;
  }
  return false;
}

void AArch64PromoteConstant::computeInsertionPoints(
    Constant *Val, InsertionPointsPerFunc &InsPtsPerFunc) {
  DEBUG(dbgs() << "** Compute insertion points **\n");
  for (Use &Use : Val->uses()) {
    Instruction *User = dyn_cast<Instruction>(Use.getUser());

    // If the user is not an Instruction, we cannot modify it.
    if (!User)
      continue;

    // Filter out uses that should not be converted.
    if (!shouldConvertUse(Val, User, Use.getOperandNo()))
      continue;

    DEBUG(dbgs() << "Considered use, opidx " << Use.getOperandNo() << ":\n");
    DEBUG(User->print(dbgs()));
    DEBUG(dbgs() << '\n');

    Instruction *InsertionPoint = findInsertionPoint(Use);

    DEBUG(dbgs() << "Considered insertion point:\n");
    DEBUG(InsertionPoint->print(dbgs()));
    DEBUG(dbgs() << '\n');

    // Check if the current insertion point is useless, i.e., it is dominated
    // by another one.
    InsertionPoints &InsertPts =
        InsPtsPerFunc[InsertionPoint->getParent()->getParent()];
    if (isDominated(InsertionPoint, Use, InsertPts))
      continue;
    // This insertion point is useful, check if we can merge some insertion
    // point in a common dominator or if NewPt dominates an existing one.
    if (tryAndMerge(InsertionPoint, Use, InsertPts))
      continue;

    DEBUG(dbgs() << "Keep considered insertion point\n");

    // It is definitely useful by its own
    InsertPts[InsertionPoint].push_back(&Use);
  }
}

bool AArch64PromoteConstant::insertDefinitions(
    Constant *Cst, InsertionPointsPerFunc &InsPtsPerFunc) {
  // We will create one global variable per Module.
  DenseMap<Module *, GlobalVariable *> ModuleToMergedGV;
  bool HasChanged = false;

  // Traverse all insertion points in all the function.
  for (const auto &FctToInstPtsIt : InsPtsPerFunc) {
    const InsertionPoints &InsertPts = FctToInstPtsIt.second;
// Do more checking for debug purposes.
#ifndef NDEBUG
    DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>(
                            *FctToInstPtsIt.first).getDomTree();
#endif
    assert(!InsertPts.empty() && "Empty uses does not need a definition");

    Module *M = FctToInstPtsIt.first->getParent();
    GlobalVariable *&PromotedGV = ModuleToMergedGV[M];
    if (!PromotedGV) {
      PromotedGV = new GlobalVariable(
          *M, Cst->getType(), true, GlobalValue::InternalLinkage, nullptr,
          "_PromotedConst", nullptr, GlobalVariable::NotThreadLocal);
      PromotedGV->setInitializer(Cst);
      DEBUG(dbgs() << "Global replacement: ");
      DEBUG(PromotedGV->print(dbgs()));
      DEBUG(dbgs() << '\n');
      ++NumPromoted;
      HasChanged = true;
    }

    for (const auto &IPI : InsertPts) {
      // Create the load of the global variable.
      IRBuilder<> Builder(IPI.first);
      LoadInst *LoadedCst = Builder.CreateLoad(PromotedGV);
      DEBUG(dbgs() << "**********\n");
      DEBUG(dbgs() << "New def: ");
      DEBUG(LoadedCst->print(dbgs()));
      DEBUG(dbgs() << '\n');

      // Update the dominated uses.
      for (Use *Use : IPI.second) {
#ifndef NDEBUG
        assert(DT.dominates(LoadedCst, findInsertionPoint(*Use)) &&
               "Inserted definition does not dominate all its uses!");
#endif
        DEBUG(dbgs() << "Use to update " << Use->getOperandNo() << ":");
        DEBUG(Use->getUser()->print(dbgs()));
        DEBUG(dbgs() << '\n');
        Use->set(LoadedCst);
        ++NumPromotedUses;
      }
    }
  }
  return HasChanged;
}

bool AArch64PromoteConstant::computeAndInsertDefinitions(Constant *Val) {
  InsertionPointsPerFunc InsertPtsPerFunc;
  computeInsertionPoints(Val, InsertPtsPerFunc);
  return insertDefinitions(Val, InsertPtsPerFunc);
}

bool AArch64PromoteConstant::promoteConstant(Constant *Cst) {
  assert(Cst && "Given variable is not a valid constant.");

  if (!shouldConvert(Cst))
    return false;

  DEBUG(dbgs() << "******************************\n");
  DEBUG(dbgs() << "Candidate constant: ");
  DEBUG(Cst->print(dbgs()));
  DEBUG(dbgs() << '\n');

  return computeAndInsertDefinitions(Cst);
}

bool AArch64PromoteConstant::runOnFunction(Function &F) {
  // Look for instructions using constant vector. Promote that constant to a
  // global variable. Create as few loads of this variable as possible and
  // update the uses accordingly.
  bool LocalChange = false;
  SmallPtrSet<Constant *, 8> AlreadyChecked;

  for (Instruction &I : instructions(&F)) {
    // Traverse the operand, looking for constant vectors. Replace them by a
    // load of a global variable of constant vector type.
    for (Value *Op : I.operand_values()) {
      Constant *Cst = dyn_cast<Constant>(Op);
      // There is no point in promoting global values as they are already
      // global. Do not promote constant expressions either, as they may
      // require some code expansion.
      if (Cst && !isa<GlobalValue>(Cst) && !isa<ConstantExpr>(Cst) &&
          AlreadyChecked.insert(Cst).second)
        LocalChange |= promoteConstant(Cst);
    }
  }
  return LocalChange;
}
