//===-- GCStrategy.cpp - Garbage collection infrastructure -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements target- and collector-independent garbage collection
// infrastructure.
//
// GCMachineCodeAnalysis identifies the GC safe points in the machine code.
// Roots are identified in SelectionDAGISel.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GCStrategy.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Module.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/DominatorInternals.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

  /// LowerIntrinsics - This pass rewrites calls to the llvm.gcread or
  /// llvm.gcwrite intrinsics, replacing them with simple loads and stores as
  /// directed by the GCStrategy. It also performs automatic root initialization
  /// and custom intrinsic lowering.
  class LowerIntrinsics : public FunctionPass {
    static bool NeedsDefaultLoweringPass(const GCStrategy &C);
    static bool NeedsCustomLoweringPass(const GCStrategy &C);
    static bool CouldBecomeSafePoint(Instruction *I);
    bool PerformDefaultLowering(Function &F, GCStrategy &Coll);
    static bool InsertRootInitializers(Function &F,
                                       AllocaInst **Roots, unsigned Count);

  public:
    static char ID;

    LowerIntrinsics();
    const char *getPassName() const;
    void getAnalysisUsage(AnalysisUsage &AU) const;

    bool doInitialization(Module &M);
    bool runOnFunction(Function &F);
  };


  /// GCMachineCodeAnalysis - This is a target-independent pass over the machine
  /// function representation to identify safe points for the garbage collector
  /// in the machine code. It inserts labels at safe points and populates a
  /// GCMetadata record for each function.
  class GCMachineCodeAnalysis : public MachineFunctionPass {
    const TargetMachine *TM;
    GCFunctionInfo *FI;
    MachineModuleInfo *MMI;
    const TargetInstrInfo *TII;

    void FindSafePoints(MachineFunction &MF);
    void VisitCallPoint(MachineBasicBlock::iterator MI);
    MCSymbol *InsertLabel(MachineBasicBlock &MBB,
                          MachineBasicBlock::iterator MI,
                          DebugLoc DL) const;

    void FindStackOffsets(MachineFunction &MF);

  public:
    static char ID;

    GCMachineCodeAnalysis();
    void getAnalysisUsage(AnalysisUsage &AU) const;

    bool runOnMachineFunction(MachineFunction &MF);
  };

}

// -----------------------------------------------------------------------------

GCStrategy::GCStrategy() :
  NeededSafePoints(0),
  CustomReadBarriers(false),
  CustomWriteBarriers(false),
  CustomRoots(false),
  CustomSafePoints(false),
  InitRoots(true),
  UsesMetadata(false)
{}

GCStrategy::~GCStrategy() {
  for (iterator I = begin(), E = end(); I != E; ++I)
    delete *I;

  Functions.clear();
}

bool GCStrategy::initializeCustomLowering(Module &M) { return false; }

bool GCStrategy::performCustomLowering(Function &F) {
  dbgs() << "gc " << getName() << " must override performCustomLowering.\n";
  llvm_unreachable("must override performCustomLowering");
}


bool GCStrategy::findCustomSafePoints(GCFunctionInfo& FI, MachineFunction &F) {
  dbgs() << "gc " << getName() << " must override findCustomSafePoints.\n";
  llvm_unreachable(0);
}


GCFunctionInfo *GCStrategy::insertFunctionInfo(const Function &F) {
  GCFunctionInfo *FI = new GCFunctionInfo(F, *this);
  Functions.push_back(FI);
  return FI;
}

// -----------------------------------------------------------------------------

INITIALIZE_PASS_BEGIN(LowerIntrinsics, "gc-lowering", "GC Lowering",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(GCModuleInfo)
INITIALIZE_PASS_END(LowerIntrinsics, "gc-lowering", "GC Lowering", false, false)

FunctionPass *llvm::createGCLoweringPass() {
  return new LowerIntrinsics();
}

char LowerIntrinsics::ID = 0;

LowerIntrinsics::LowerIntrinsics()
  : FunctionPass(ID) {
    initializeLowerIntrinsicsPass(*PassRegistry::getPassRegistry());
  }

const char *LowerIntrinsics::getPassName() const {
  return "Lower Garbage Collection Instructions";
}

void LowerIntrinsics::getAnalysisUsage(AnalysisUsage &AU) const {
  FunctionPass::getAnalysisUsage(AU);
  AU.addRequired<GCModuleInfo>();
  AU.addPreserved<DominatorTree>();
}

/// doInitialization - If this module uses the GC intrinsics, find them now.
bool LowerIntrinsics::doInitialization(Module &M) {
  // FIXME: This is rather antisocial in the context of a JIT since it performs
  //        work against the entire module. But this cannot be done at
  //        runFunction time (initializeCustomLowering likely needs to change
  //        the module).
  GCModuleInfo *MI = getAnalysisIfAvailable<GCModuleInfo>();
  assert(MI && "LowerIntrinsics didn't require GCModuleInfo!?");
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (!I->isDeclaration() && I->hasGC())
      MI->getFunctionInfo(*I); // Instantiate the GC strategy.

  bool MadeChange = false;
  for (GCModuleInfo::iterator I = MI->begin(), E = MI->end(); I != E; ++I)
    if (NeedsCustomLoweringPass(**I))
      if ((*I)->initializeCustomLowering(M))
        MadeChange = true;

  return MadeChange;
}

bool LowerIntrinsics::InsertRootInitializers(Function &F, AllocaInst **Roots,
                                                          unsigned Count) {
  // Scroll past alloca instructions.
  BasicBlock::iterator IP = F.getEntryBlock().begin();
  while (isa<AllocaInst>(IP)) ++IP;

  // Search for initializers in the initial BB.
  SmallPtrSet<AllocaInst*,16> InitedRoots;
  for (; !CouldBecomeSafePoint(IP); ++IP)
    if (StoreInst *SI = dyn_cast<StoreInst>(IP))
      if (AllocaInst *AI =
          dyn_cast<AllocaInst>(SI->getOperand(1)->stripPointerCasts()))
        InitedRoots.insert(AI);

  // Add root initializers.
  bool MadeChange = false;

  for (AllocaInst **I = Roots, **E = Roots + Count; I != E; ++I)
    if (!InitedRoots.count(*I)) {
      StoreInst* SI = new StoreInst(ConstantPointerNull::get(cast<PointerType>(
                        cast<PointerType>((*I)->getType())->getElementType())),
                        *I);
      SI->insertAfter(*I);
      MadeChange = true;
    }

  return MadeChange;
}

bool LowerIntrinsics::NeedsDefaultLoweringPass(const GCStrategy &C) {
  // Default lowering is necessary only if read or write barriers have a default
  // action. The default for roots is no action.
  return !C.customWriteBarrier()
      || !C.customReadBarrier()
      || C.initializeRoots();
}

bool LowerIntrinsics::NeedsCustomLoweringPass(const GCStrategy &C) {
  // Custom lowering is only necessary if enabled for some action.
  return C.customWriteBarrier()
      || C.customReadBarrier()
      || C.customRoots();
}

/// CouldBecomeSafePoint - Predicate to conservatively determine whether the
/// instruction could introduce a safe point.
bool LowerIntrinsics::CouldBecomeSafePoint(Instruction *I) {
  // The natural definition of instructions which could introduce safe points
  // are:
  //
  //   - call, invoke (AfterCall, BeforeCall)
  //   - phis (Loops)
  //   - invoke, ret, unwind (Exit)
  //
  // However, instructions as seemingly inoccuous as arithmetic can become
  // libcalls upon lowering (e.g., div i64 on a 32-bit platform), so instead
  // it is necessary to take a conservative approach.

  if (isa<AllocaInst>(I) || isa<GetElementPtrInst>(I) ||
      isa<StoreInst>(I) || isa<LoadInst>(I))
    return false;

  // llvm.gcroot is safe because it doesn't do anything at runtime.
  if (CallInst *CI = dyn_cast<CallInst>(I))
    if (Function *F = CI->getCalledFunction())
      if (unsigned IID = F->getIntrinsicID())
        if (IID == Intrinsic::gcroot)
          return false;

  return true;
}

/// runOnFunction - Replace gcread/gcwrite intrinsics with loads and stores.
/// Leave gcroot intrinsics; the code generator needs to see those.
bool LowerIntrinsics::runOnFunction(Function &F) {
  // Quick exit for functions that do not use GC.
  if (!F.hasGC())
    return false;

  GCFunctionInfo &FI = getAnalysis<GCModuleInfo>().getFunctionInfo(F);
  GCStrategy &S = FI.getStrategy();

  bool MadeChange = false;

  if (NeedsDefaultLoweringPass(S))
    MadeChange |= PerformDefaultLowering(F, S);

  bool UseCustomLoweringPass = NeedsCustomLoweringPass(S);
  if (UseCustomLoweringPass)
    MadeChange |= S.performCustomLowering(F);

  // Custom lowering may modify the CFG, so dominators must be recomputed.
  if (UseCustomLoweringPass) {
    if (DominatorTree *DT = getAnalysisIfAvailable<DominatorTree>())
      DT->DT->recalculate(F);
  }

  return MadeChange;
}

bool LowerIntrinsics::PerformDefaultLowering(Function &F, GCStrategy &S) {
  bool LowerWr = !S.customWriteBarrier();
  bool LowerRd = !S.customReadBarrier();
  bool InitRoots = S.initializeRoots();

  SmallVector<AllocaInst*, 32> Roots;

  bool MadeChange = false;
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
    for (BasicBlock::iterator II = BB->begin(), E = BB->end(); II != E;) {
      if (IntrinsicInst *CI = dyn_cast<IntrinsicInst>(II++)) {
        Function *F = CI->getCalledFunction();
        switch (F->getIntrinsicID()) {
        case Intrinsic::gcwrite:
          if (LowerWr) {
            // Replace a write barrier with a simple store.
            Value *St = new StoreInst(CI->getArgOperand(0),
                                      CI->getArgOperand(2), CI);
            CI->replaceAllUsesWith(St);
            CI->eraseFromParent();
          }
          break;
        case Intrinsic::gcread:
          if (LowerRd) {
            // Replace a read barrier with a simple load.
            Value *Ld = new LoadInst(CI->getArgOperand(1), "", CI);
            Ld->takeName(CI);
            CI->replaceAllUsesWith(Ld);
            CI->eraseFromParent();
          }
          break;
        case Intrinsic::gcroot:
          if (InitRoots) {
            // Initialize the GC root, but do not delete the intrinsic. The
            // backend needs the intrinsic to flag the stack slot.
            Roots.push_back(cast<AllocaInst>(
                              CI->getArgOperand(0)->stripPointerCasts()));
          }
          break;
        default:
          continue;
        }

        MadeChange = true;
      }
    }
  }

  if (Roots.size())
    MadeChange |= InsertRootInitializers(F, Roots.begin(), Roots.size());

  return MadeChange;
}

// -----------------------------------------------------------------------------

char GCMachineCodeAnalysis::ID = 0;
char &llvm::GCMachineCodeAnalysisID = GCMachineCodeAnalysis::ID;

INITIALIZE_PASS(GCMachineCodeAnalysis, "gc-analysis",
                "Analyze Machine Code For Garbage Collection", false, false)

GCMachineCodeAnalysis::GCMachineCodeAnalysis()
  : MachineFunctionPass(ID) {}

void GCMachineCodeAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  MachineFunctionPass::getAnalysisUsage(AU);
  AU.setPreservesAll();
  AU.addRequired<MachineModuleInfo>();
  AU.addRequired<GCModuleInfo>();
}

MCSymbol *GCMachineCodeAnalysis::InsertLabel(MachineBasicBlock &MBB,
                                             MachineBasicBlock::iterator MI,
                                             DebugLoc DL) const {
  MCSymbol *Label = MBB.getParent()->getContext().CreateTempSymbol();
  BuildMI(MBB, MI, DL, TII->get(TargetOpcode::GC_LABEL)).addSym(Label);
  return Label;
}

void GCMachineCodeAnalysis::VisitCallPoint(MachineBasicBlock::iterator CI) {
  // Find the return address (next instruction), too, so as to bracket the call
  // instruction.
  MachineBasicBlock::iterator RAI = CI;
  ++RAI;

  if (FI->getStrategy().needsSafePoint(GC::PreCall)) {
    MCSymbol* Label = InsertLabel(*CI->getParent(), CI, CI->getDebugLoc());
    FI->addSafePoint(GC::PreCall, Label, CI->getDebugLoc());
  }

  if (FI->getStrategy().needsSafePoint(GC::PostCall)) {
    MCSymbol* Label = InsertLabel(*CI->getParent(), RAI, CI->getDebugLoc());
    FI->addSafePoint(GC::PostCall, Label, CI->getDebugLoc());
  }
}

void GCMachineCodeAnalysis::FindSafePoints(MachineFunction &MF) {
  for (MachineFunction::iterator BBI = MF.begin(),
                                 BBE = MF.end(); BBI != BBE; ++BBI)
    for (MachineBasicBlock::iterator MI = BBI->begin(),
                                     ME = BBI->end(); MI != ME; ++MI)
      if (MI->isCall())
        VisitCallPoint(MI);
}

void GCMachineCodeAnalysis::FindStackOffsets(MachineFunction &MF) {
  const TargetFrameLowering *TFI = TM->getFrameLowering();
  assert(TFI && "TargetRegisterInfo not available!");

  for (GCFunctionInfo::roots_iterator RI = FI->roots_begin();
       RI != FI->roots_end();) {
    // If the root references a dead object, no need to keep it.
    if (MF.getFrameInfo()->isDeadObjectIndex(RI->Num)) {
      RI = FI->removeStackRoot(RI);
    } else {
      RI->StackOffset = TFI->getFrameIndexOffset(MF, RI->Num);
      ++RI;
    }
  }
}

bool GCMachineCodeAnalysis::runOnMachineFunction(MachineFunction &MF) {
  // Quick exit for functions that do not use GC.
  if (!MF.getFunction()->hasGC())
    return false;

  FI = &getAnalysis<GCModuleInfo>().getFunctionInfo(*MF.getFunction());
  if (!FI->getStrategy().needsSafePoints())
    return false;

  TM = &MF.getTarget();
  MMI = &getAnalysis<MachineModuleInfo>();
  TII = TM->getInstrInfo();

  // Find the size of the stack frame.
  FI->setFrameSize(MF.getFrameInfo()->getStackSize());

  // Find all safe points.
  if (FI->getStrategy().customSafePoints()) {
    FI->getStrategy().findCustomSafePoints(*FI, MF);
  } else {
    FindSafePoints(MF);
  }

  // Find the stack offsets for all roots.
  FindStackOffsets(MF);

  return false;
}
