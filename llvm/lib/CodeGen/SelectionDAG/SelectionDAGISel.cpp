//===-- SelectionDAGISel.cpp - Implement the SelectionDAGISel class -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the SelectionDAGISel class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "isel"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "SelectionDAGBuild.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Constants.h"
#include "llvm/CallingConv.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/InlineAsm.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/CodeGen/FastISel.h"
#include "llvm/CodeGen/GCStrategy.h"
#include "llvm/CodeGen/GCMetadata.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/ScheduleDAGSDNodes.h"
#include "llvm/CodeGen/ScheduleHazardRecognizer.h"
#include "llvm/CodeGen/SchedulerRegistry.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/DwarfWriter.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Timer.h"
#include <algorithm>
using namespace llvm;

static cl::opt<bool>
EnableValueProp("enable-value-prop", cl::Hidden);
static cl::opt<bool>
DisableLegalizeTypes("disable-legalize-types", cl::Hidden);
#ifndef NDEBUG
static cl::opt<bool>
EnableFastISelVerbose("fast-isel-verbose", cl::Hidden,
          cl::desc("Enable verbose messages in the \"fast\" "
                   "instruction selector"));
static cl::opt<bool>
EnableFastISelAbort("fast-isel-abort", cl::Hidden,
          cl::desc("Enable abort calls when \"fast\" instruction fails"));
#else
static const bool EnableFastISelVerbose = false,
                  EnableFastISelAbort = false;
#endif
static cl::opt<bool>
SchedLiveInCopies("schedule-livein-copies",
                  cl::desc("Schedule copies of livein registers"),
                  cl::init(false));

#ifndef NDEBUG
static cl::opt<bool>
ViewDAGCombine1("view-dag-combine1-dags", cl::Hidden,
          cl::desc("Pop up a window to show dags before the first "
                   "dag combine pass"));
static cl::opt<bool>
ViewLegalizeTypesDAGs("view-legalize-types-dags", cl::Hidden,
          cl::desc("Pop up a window to show dags before legalize types"));
static cl::opt<bool>
ViewLegalizeDAGs("view-legalize-dags", cl::Hidden,
          cl::desc("Pop up a window to show dags before legalize"));
static cl::opt<bool>
ViewDAGCombine2("view-dag-combine2-dags", cl::Hidden,
          cl::desc("Pop up a window to show dags before the second "
                   "dag combine pass"));
static cl::opt<bool>
ViewDAGCombineLT("view-dag-combine-lt-dags", cl::Hidden,
          cl::desc("Pop up a window to show dags before the post legalize types"
                   " dag combine pass"));
static cl::opt<bool>
ViewISelDAGs("view-isel-dags", cl::Hidden,
          cl::desc("Pop up a window to show isel dags as they are selected"));
static cl::opt<bool>
ViewSchedDAGs("view-sched-dags", cl::Hidden,
          cl::desc("Pop up a window to show sched dags as they are processed"));
static cl::opt<bool>
ViewSUnitDAGs("view-sunit-dags", cl::Hidden,
      cl::desc("Pop up a window to show SUnit dags after they are processed"));
#else
static const bool ViewDAGCombine1 = false,
                  ViewLegalizeTypesDAGs = false, ViewLegalizeDAGs = false,
                  ViewDAGCombine2 = false,
                  ViewDAGCombineLT = false,
                  ViewISelDAGs = false, ViewSchedDAGs = false,
                  ViewSUnitDAGs = false;
#endif

//===---------------------------------------------------------------------===//
///
/// RegisterScheduler class - Track the registration of instruction schedulers.
///
//===---------------------------------------------------------------------===//
MachinePassRegistry RegisterScheduler::Registry;

//===---------------------------------------------------------------------===//
///
/// ISHeuristic command line option for instruction schedulers.
///
//===---------------------------------------------------------------------===//
static cl::opt<RegisterScheduler::FunctionPassCtor, false,
               RegisterPassParser<RegisterScheduler> >
ISHeuristic("pre-RA-sched",
            cl::init(&createDefaultScheduler),
            cl::desc("Instruction schedulers available (before register"
                     " allocation):"));

static RegisterScheduler
defaultListDAGScheduler("default", "Best scheduler for the target",
                        createDefaultScheduler);

namespace llvm {
  //===--------------------------------------------------------------------===//
  /// createDefaultScheduler - This creates an instruction scheduler appropriate
  /// for the target.
  ScheduleDAG* createDefaultScheduler(SelectionDAGISel *IS,
                                      bool Fast) {
    const TargetLowering &TLI = IS->getTargetLowering();

    if (Fast)
      return createFastDAGScheduler(IS, Fast);
    if (TLI.getSchedulingPreference() == TargetLowering::SchedulingForLatency)
      return createTDListDAGScheduler(IS, Fast);
    assert(TLI.getSchedulingPreference() ==
         TargetLowering::SchedulingForRegPressure && "Unknown sched type!");
    return createBURRListDAGScheduler(IS, Fast);
  }
}

// EmitInstrWithCustomInserter - This method should be implemented by targets
// that mark instructions with the 'usesCustomDAGSchedInserter' flag.  These
// instructions are special in various ways, which require special support to
// insert.  The specified MachineInstr is created but not inserted into any
// basic blocks, and the scheduler passes ownership of it to this method.
MachineBasicBlock *TargetLowering::EmitInstrWithCustomInserter(MachineInstr *MI,
                                                       MachineBasicBlock *MBB) {
  cerr << "If a target marks an instruction with "
       << "'usesCustomDAGSchedInserter', it must implement "
       << "TargetLowering::EmitInstrWithCustomInserter!\n";
  abort();
  return 0;  
}

/// EmitLiveInCopy - Emit a copy for a live in physical register. If the
/// physical register has only a single copy use, then coalesced the copy
/// if possible.
static void EmitLiveInCopy(MachineBasicBlock *MBB,
                           MachineBasicBlock::iterator &InsertPos,
                           unsigned VirtReg, unsigned PhysReg,
                           const TargetRegisterClass *RC,
                           DenseMap<MachineInstr*, unsigned> &CopyRegMap,
                           const MachineRegisterInfo &MRI,
                           const TargetRegisterInfo &TRI,
                           const TargetInstrInfo &TII) {
  unsigned NumUses = 0;
  MachineInstr *UseMI = NULL;
  for (MachineRegisterInfo::use_iterator UI = MRI.use_begin(VirtReg),
         UE = MRI.use_end(); UI != UE; ++UI) {
    UseMI = &*UI;
    if (++NumUses > 1)
      break;
  }

  // If the number of uses is not one, or the use is not a move instruction,
  // don't coalesce. Also, only coalesce away a virtual register to virtual
  // register copy.
  bool Coalesced = false;
  unsigned SrcReg, DstReg, SrcSubReg, DstSubReg;
  if (NumUses == 1 &&
      TII.isMoveInstr(*UseMI, SrcReg, DstReg, SrcSubReg, DstSubReg) &&
      TargetRegisterInfo::isVirtualRegister(DstReg)) {
    VirtReg = DstReg;
    Coalesced = true;
  }

  // Now find an ideal location to insert the copy.
  MachineBasicBlock::iterator Pos = InsertPos;
  while (Pos != MBB->begin()) {
    MachineInstr *PrevMI = prior(Pos);
    DenseMap<MachineInstr*, unsigned>::iterator RI = CopyRegMap.find(PrevMI);
    // copyRegToReg might emit multiple instructions to do a copy.
    unsigned CopyDstReg = (RI == CopyRegMap.end()) ? 0 : RI->second;
    if (CopyDstReg && !TRI.regsOverlap(CopyDstReg, PhysReg))
      // This is what the BB looks like right now:
      // r1024 = mov r0
      // ...
      // r1    = mov r1024
      //
      // We want to insert "r1025 = mov r1". Inserting this copy below the
      // move to r1024 makes it impossible for that move to be coalesced.
      //
      // r1025 = mov r1
      // r1024 = mov r0
      // ...
      // r1    = mov 1024
      // r2    = mov 1025
      break; // Woot! Found a good location.
    --Pos;
  }

  TII.copyRegToReg(*MBB, Pos, VirtReg, PhysReg, RC, RC);
  CopyRegMap.insert(std::make_pair(prior(Pos), VirtReg));
  if (Coalesced) {
    if (&*InsertPos == UseMI) ++InsertPos;
    MBB->erase(UseMI);
  }
}

/// EmitLiveInCopies - If this is the first basic block in the function,
/// and if it has live ins that need to be copied into vregs, emit the
/// copies into the block.
static void EmitLiveInCopies(MachineBasicBlock *EntryMBB,
                             const MachineRegisterInfo &MRI,
                             const TargetRegisterInfo &TRI,
                             const TargetInstrInfo &TII) {
  if (SchedLiveInCopies) {
    // Emit the copies at a heuristically-determined location in the block.
    DenseMap<MachineInstr*, unsigned> CopyRegMap;
    MachineBasicBlock::iterator InsertPos = EntryMBB->begin();
    for (MachineRegisterInfo::livein_iterator LI = MRI.livein_begin(),
           E = MRI.livein_end(); LI != E; ++LI)
      if (LI->second) {
        const TargetRegisterClass *RC = MRI.getRegClass(LI->second);
        EmitLiveInCopy(EntryMBB, InsertPos, LI->second, LI->first,
                       RC, CopyRegMap, MRI, TRI, TII);
      }
  } else {
    // Emit the copies into the top of the block.
    for (MachineRegisterInfo::livein_iterator LI = MRI.livein_begin(),
           E = MRI.livein_end(); LI != E; ++LI)
      if (LI->second) {
        const TargetRegisterClass *RC = MRI.getRegClass(LI->second);
        TII.copyRegToReg(*EntryMBB, EntryMBB->begin(),
                         LI->second, LI->first, RC, RC);
      }
  }
}

//===----------------------------------------------------------------------===//
// SelectionDAGISel code
//===----------------------------------------------------------------------===//

SelectionDAGISel::SelectionDAGISel(TargetMachine &tm, bool fast) :
  FunctionPass(&ID), TM(tm), TLI(*tm.getTargetLowering()),
  FuncInfo(new FunctionLoweringInfo(TLI)),
  CurDAG(new SelectionDAG(TLI, *FuncInfo)),
  SDL(new SelectionDAGLowering(*CurDAG, TLI, *FuncInfo)),
  GFI(),
  Fast(fast),
  DAGSize(0)
{}

SelectionDAGISel::~SelectionDAGISel() {
  delete SDL;
  delete CurDAG;
  delete FuncInfo;
}

unsigned SelectionDAGISel::MakeReg(MVT VT) {
  return RegInfo->createVirtualRegister(TLI.getRegClassFor(VT));
}

void SelectionDAGISel::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<AliasAnalysis>();
  AU.addRequired<GCModuleInfo>();
  AU.addRequired<DwarfWriter>();
  AU.setPreservesAll();
}

bool SelectionDAGISel::runOnFunction(Function &Fn) {
  // Do some sanity-checking on the command-line options.
  assert((!EnableFastISelVerbose || EnableFastISel) &&
         "-fast-isel-verbose requires -fast-isel");
  assert((!EnableFastISelAbort || EnableFastISel) &&
         "-fast-isel-abort requires -fast-isel");

  // Get alias analysis for load/store combining.
  AA = &getAnalysis<AliasAnalysis>();

  TargetMachine &TM = TLI.getTargetMachine();
  MF = &MachineFunction::construct(&Fn, TM);
  const TargetInstrInfo &TII = *TM.getInstrInfo();
  const TargetRegisterInfo &TRI = *TM.getRegisterInfo();

  if (MF->getFunction()->hasGC())
    GFI = &getAnalysis<GCModuleInfo>().getFunctionInfo(*MF->getFunction());
  else
    GFI = 0;
  RegInfo = &MF->getRegInfo();
  DOUT << "\n\n\n=== " << Fn.getName() << "\n";

  MachineModuleInfo *MMI = getAnalysisIfAvailable<MachineModuleInfo>();
  DwarfWriter *DW = getAnalysisIfAvailable<DwarfWriter>();
  CurDAG->init(*MF, MMI, DW);
  FuncInfo->set(Fn, *MF, *CurDAG, EnableFastISel);
  SDL->init(GFI, *AA);

  for (Function::iterator I = Fn.begin(), E = Fn.end(); I != E; ++I)
    if (InvokeInst *Invoke = dyn_cast<InvokeInst>(I->getTerminator()))
      // Mark landing pad.
      FuncInfo->MBBMap[Invoke->getSuccessor(1)]->setIsLandingPad();

  SelectAllBasicBlocks(Fn, *MF, MMI, DW, TII);

  // If the first basic block in the function has live ins that need to be
  // copied into vregs, emit the copies into the top of the block before
  // emitting the code for the block.
  EmitLiveInCopies(MF->begin(), *RegInfo, TRI, TII);

  // Add function live-ins to entry block live-in set.
  for (MachineRegisterInfo::livein_iterator I = RegInfo->livein_begin(),
         E = RegInfo->livein_end(); I != E; ++I)
    MF->begin()->addLiveIn(I->first);

#ifndef NDEBUG
  assert(FuncInfo->CatchInfoFound.size() == FuncInfo->CatchInfoLost.size() &&
         "Not all catch info was assigned to a landing pad!");
#endif

  FuncInfo->clear();

  return true;
}

static void copyCatchInfo(BasicBlock *SrcBB, BasicBlock *DestBB,
                          MachineModuleInfo *MMI, FunctionLoweringInfo &FLI) {
  for (BasicBlock::iterator I = SrcBB->begin(), E = --SrcBB->end(); I != E; ++I)
    if (EHSelectorInst *EHSel = dyn_cast<EHSelectorInst>(I)) {
      // Apply the catch info to DestBB.
      AddCatchInfo(*EHSel, MMI, FLI.MBBMap[DestBB]);
#ifndef NDEBUG
      if (!FLI.MBBMap[SrcBB]->isLandingPad())
        FLI.CatchInfoFound.insert(EHSel);
#endif
    }
}

/// IsFixedFrameObjectWithPosOffset - Check if object is a fixed frame object and
/// whether object offset >= 0.
static bool
IsFixedFrameObjectWithPosOffset(MachineFrameInfo *MFI, SDValue Op) {
  if (!isa<FrameIndexSDNode>(Op)) return false;

  FrameIndexSDNode * FrameIdxNode = dyn_cast<FrameIndexSDNode>(Op);
  int FrameIdx =  FrameIdxNode->getIndex();
  return MFI->isFixedObjectIndex(FrameIdx) &&
    MFI->getObjectOffset(FrameIdx) >= 0;
}

/// IsPossiblyOverwrittenArgumentOfTailCall - Check if the operand could
/// possibly be overwritten when lowering the outgoing arguments in a tail
/// call. Currently the implementation of this call is very conservative and
/// assumes all arguments sourcing from FORMAL_ARGUMENTS or a CopyFromReg with
/// virtual registers would be overwritten by direct lowering.
static bool IsPossiblyOverwrittenArgumentOfTailCall(SDValue Op,
                                                    MachineFrameInfo *MFI) {
  RegisterSDNode * OpReg = NULL;
  if (Op.getOpcode() == ISD::FORMAL_ARGUMENTS ||
      (Op.getOpcode()== ISD::CopyFromReg &&
       (OpReg = dyn_cast<RegisterSDNode>(Op.getOperand(1))) &&
       (OpReg->getReg() >= TargetRegisterInfo::FirstVirtualRegister)) ||
      (Op.getOpcode() == ISD::LOAD &&
       IsFixedFrameObjectWithPosOffset(MFI, Op.getOperand(1))) ||
      (Op.getOpcode() == ISD::MERGE_VALUES &&
       Op.getOperand(Op.getResNo()).getOpcode() == ISD::LOAD &&
       IsFixedFrameObjectWithPosOffset(MFI, Op.getOperand(Op.getResNo()).
                                       getOperand(1))))
    return true;
  return false;
}

/// CheckDAGForTailCallsAndFixThem - This Function looks for CALL nodes in the
/// DAG and fixes their tailcall attribute operand.
static void CheckDAGForTailCallsAndFixThem(SelectionDAG &DAG, 
                                           const TargetLowering& TLI) {
  SDNode * Ret = NULL;
  SDValue Terminator = DAG.getRoot();

  // Find RET node.
  if (Terminator.getOpcode() == ISD::RET) {
    Ret = Terminator.getNode();
  }
 
  // Fix tail call attribute of CALL nodes.
  for (SelectionDAG::allnodes_iterator BE = DAG.allnodes_begin(),
         BI = DAG.allnodes_end(); BI != BE; ) {
    --BI;
    if (CallSDNode *TheCall = dyn_cast<CallSDNode>(BI)) {
      SDValue OpRet(Ret, 0);
      SDValue OpCall(BI, 0);
      bool isMarkedTailCall = TheCall->isTailCall();
      // If CALL node has tail call attribute set to true and the call is not
      // eligible (no RET or the target rejects) the attribute is fixed to
      // false. The TargetLowering::IsEligibleForTailCallOptimization function
      // must correctly identify tail call optimizable calls.
      if (!isMarkedTailCall) continue;
      if (Ret==NULL ||
          !TLI.IsEligibleForTailCallOptimization(TheCall, OpRet, DAG)) {
        // Not eligible. Mark CALL node as non tail call. Note that we
        // can modify the call node in place since calls are not CSE'd.
        TheCall->setNotTailCall();
      } else {
        // Look for tail call clobbered arguments. Emit a series of
        // copyto/copyfrom virtual register nodes to protect them.
        SmallVector<SDValue, 32> Ops;
        SDValue Chain = TheCall->getChain(), InFlag;
        Ops.push_back(Chain);
        Ops.push_back(TheCall->getCallee());
        for (unsigned i = 0, e = TheCall->getNumArgs(); i != e; ++i) {
          SDValue Arg = TheCall->getArg(i);
          bool isByVal = TheCall->getArgFlags(i).isByVal();
          MachineFunction &MF = DAG.getMachineFunction();
          MachineFrameInfo *MFI = MF.getFrameInfo();
          if (!isByVal &&
              IsPossiblyOverwrittenArgumentOfTailCall(Arg, MFI)) {
            MVT VT = Arg.getValueType();
            unsigned VReg = MF.getRegInfo().
              createVirtualRegister(TLI.getRegClassFor(VT));
            Chain = DAG.getCopyToReg(Chain, VReg, Arg, InFlag);
            InFlag = Chain.getValue(1);
            Arg = DAG.getCopyFromReg(Chain, VReg, VT, InFlag);
            Chain = Arg.getValue(1);
            InFlag = Arg.getValue(2);
          }
          Ops.push_back(Arg);
          Ops.push_back(TheCall->getArgFlagsVal(i));
        }
        // Link in chain of CopyTo/CopyFromReg.
        Ops[0] = Chain;
        DAG.UpdateNodeOperands(OpCall, Ops.begin(), Ops.size());
      }
    }
  }
}

void SelectionDAGISel::SelectBasicBlock(BasicBlock *LLVMBB,
                                        BasicBlock::iterator Begin,
                                        BasicBlock::iterator End) {
  SDL->setCurrentBasicBlock(BB);

  // Lower all of the non-terminator instructions.
  for (BasicBlock::iterator I = Begin; I != End; ++I)
    if (!isa<TerminatorInst>(I))
      SDL->visit(*I);

  // Ensure that all instructions which are used outside of their defining
  // blocks are available as virtual registers.  Invoke is handled elsewhere.
  for (BasicBlock::iterator I = Begin; I != End; ++I)
    if (!I->use_empty() && !isa<PHINode>(I) && !isa<InvokeInst>(I)) {
      DenseMap<const Value*,unsigned>::iterator VMI =FuncInfo->ValueMap.find(I);
      if (VMI != FuncInfo->ValueMap.end())
        SDL->CopyValueToVirtualRegister(I, VMI->second);
    }

  // Handle PHI nodes in successor blocks.
  if (End == LLVMBB->end()) {
    HandlePHINodesInSuccessorBlocks(LLVMBB);

    // Lower the terminator after the copies are emitted.
    SDL->visit(*LLVMBB->getTerminator());
  }
    
  // Make sure the root of the DAG is up-to-date.
  CurDAG->setRoot(SDL->getControlRoot());

  // Check whether calls in this block are real tail calls. Fix up CALL nodes
  // with correct tailcall attribute so that the target can rely on the tailcall
  // attribute indicating whether the call is really eligible for tail call
  // optimization.
  if (PerformTailCallOpt)
    CheckDAGForTailCallsAndFixThem(*CurDAG, TLI);

  // Final step, emit the lowered DAG as machine code.
  CodeGenAndEmitDAG();
  SDL->clear();
}

void SelectionDAGISel::ComputeLiveOutVRegInfo() {
  SmallPtrSet<SDNode*, 128> VisitedNodes;
  SmallVector<SDNode*, 128> Worklist;
  
  Worklist.push_back(CurDAG->getRoot().getNode());
  
  APInt Mask;
  APInt KnownZero;
  APInt KnownOne;
  
  while (!Worklist.empty()) {
    SDNode *N = Worklist.back();
    Worklist.pop_back();
    
    // If we've already seen this node, ignore it.
    if (!VisitedNodes.insert(N))
      continue;
    
    // Otherwise, add all chain operands to the worklist.
    for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i)
      if (N->getOperand(i).getValueType() == MVT::Other)
        Worklist.push_back(N->getOperand(i).getNode());
    
    // If this is a CopyToReg with a vreg dest, process it.
    if (N->getOpcode() != ISD::CopyToReg)
      continue;
    
    unsigned DestReg = cast<RegisterSDNode>(N->getOperand(1))->getReg();
    if (!TargetRegisterInfo::isVirtualRegister(DestReg))
      continue;
    
    // Ignore non-scalar or non-integer values.
    SDValue Src = N->getOperand(2);
    MVT SrcVT = Src.getValueType();
    if (!SrcVT.isInteger() || SrcVT.isVector())
      continue;
    
    unsigned NumSignBits = CurDAG->ComputeNumSignBits(Src);
    Mask = APInt::getAllOnesValue(SrcVT.getSizeInBits());
    CurDAG->ComputeMaskedBits(Src, Mask, KnownZero, KnownOne);
    
    // Only install this information if it tells us something.
    if (NumSignBits != 1 || KnownZero != 0 || KnownOne != 0) {
      DestReg -= TargetRegisterInfo::FirstVirtualRegister;
      FunctionLoweringInfo &FLI = CurDAG->getFunctionLoweringInfo();
      if (DestReg >= FLI.LiveOutRegInfo.size())
        FLI.LiveOutRegInfo.resize(DestReg+1);
      FunctionLoweringInfo::LiveOutInfo &LOI = FLI.LiveOutRegInfo[DestReg];
      LOI.NumSignBits = NumSignBits;
      LOI.KnownOne = NumSignBits;
      LOI.KnownZero = NumSignBits;
    }
  }
}

void SelectionDAGISel::CodeGenAndEmitDAG() {
  std::string GroupName;
  if (TimePassesIsEnabled)
    GroupName = "Instruction Selection and Scheduling";
  std::string BlockName;
  if (ViewDAGCombine1 || ViewLegalizeTypesDAGs || ViewLegalizeDAGs ||
      ViewDAGCombine2 || ViewDAGCombineLT || ViewISelDAGs || ViewSchedDAGs ||
      ViewSUnitDAGs)
    BlockName = CurDAG->getMachineFunction().getFunction()->getName() + ':' +
                BB->getBasicBlock()->getName();

  DOUT << "Initial selection DAG:\n";
  DEBUG(CurDAG->dump());

  if (ViewDAGCombine1) CurDAG->viewGraph("dag-combine1 input for " + BlockName);

  // Run the DAG combiner in pre-legalize mode.
  if (TimePassesIsEnabled) {
    NamedRegionTimer T("DAG Combining 1", GroupName);
    CurDAG->Combine(Unrestricted, *AA, Fast);
  } else {
    CurDAG->Combine(Unrestricted, *AA, Fast);
  }
  
  DOUT << "Optimized lowered selection DAG:\n";
  DEBUG(CurDAG->dump());
  
  // Second step, hack on the DAG until it only uses operations and types that
  // the target supports.
  if (!DisableLegalizeTypes) {
    if (ViewLegalizeTypesDAGs) CurDAG->viewGraph("legalize-types input for " +
                                                 BlockName);

    bool Changed;
    if (TimePassesIsEnabled) {
      NamedRegionTimer T("Type Legalization", GroupName);
      Changed = CurDAG->LegalizeTypes();
    } else {
      Changed = CurDAG->LegalizeTypes();
    }

    DOUT << "Type-legalized selection DAG:\n";
    DEBUG(CurDAG->dump());

    if (Changed) {
      if (ViewDAGCombineLT)
        CurDAG->viewGraph("dag-combine-lt input for " + BlockName);

      // Run the DAG combiner in post-type-legalize mode.
      if (TimePassesIsEnabled) {
        NamedRegionTimer T("DAG Combining after legalize types", GroupName);
        CurDAG->Combine(NoIllegalTypes, *AA, Fast);
      } else {
        CurDAG->Combine(NoIllegalTypes, *AA, Fast);
      }

      DOUT << "Optimized type-legalized selection DAG:\n";
      DEBUG(CurDAG->dump());
    }
  }
  
  if (ViewLegalizeDAGs) CurDAG->viewGraph("legalize input for " + BlockName);

  if (TimePassesIsEnabled) {
    NamedRegionTimer T("DAG Legalization", GroupName);
    CurDAG->Legalize(DisableLegalizeTypes);
  } else {
    CurDAG->Legalize(DisableLegalizeTypes);
  }
  
  DOUT << "Legalized selection DAG:\n";
  DEBUG(CurDAG->dump());
  
  if (ViewDAGCombine2) CurDAG->viewGraph("dag-combine2 input for " + BlockName);

  // Run the DAG combiner in post-legalize mode.
  if (TimePassesIsEnabled) {
    NamedRegionTimer T("DAG Combining 2", GroupName);
    CurDAG->Combine(NoIllegalOperations, *AA, Fast);
  } else {
    CurDAG->Combine(NoIllegalOperations, *AA, Fast);
  }
  
  DOUT << "Optimized legalized selection DAG:\n";
  DEBUG(CurDAG->dump());

  if (ViewISelDAGs) CurDAG->viewGraph("isel input for " + BlockName);
  
  if (!Fast && EnableValueProp)
    ComputeLiveOutVRegInfo();

  // Third, instruction select all of the operations to machine code, adding the
  // code to the MachineBasicBlock.
  if (TimePassesIsEnabled) {
    NamedRegionTimer T("Instruction Selection", GroupName);
    InstructionSelect();
  } else {
    InstructionSelect();
  }

  DOUT << "Selected selection DAG:\n";
  DEBUG(CurDAG->dump());

  if (ViewSchedDAGs) CurDAG->viewGraph("scheduler input for " + BlockName);

  // Schedule machine code.
  ScheduleDAG *Scheduler;
  if (TimePassesIsEnabled) {
    NamedRegionTimer T("Instruction Scheduling", GroupName);
    Scheduler = Schedule();
  } else {
    Scheduler = Schedule();
  }

  if (ViewSUnitDAGs) Scheduler->viewGraph();

  // Emit machine code to BB.  This can change 'BB' to the last block being 
  // inserted into.
  if (TimePassesIsEnabled) {
    NamedRegionTimer T("Instruction Creation", GroupName);
    BB = Scheduler->EmitSchedule();
  } else {
    BB = Scheduler->EmitSchedule();
  }

  // Free the scheduler state.
  if (TimePassesIsEnabled) {
    NamedRegionTimer T("Instruction Scheduling Cleanup", GroupName);
    delete Scheduler;
  } else {
    delete Scheduler;
  }

  DOUT << "Selected machine code:\n";
  DEBUG(BB->dump());
}  

void SelectionDAGISel::SelectAllBasicBlocks(Function &Fn,
                                            MachineFunction &MF,
                                            MachineModuleInfo *MMI,
                                            DwarfWriter *DW,
                                            const TargetInstrInfo &TII) {
  // Initialize the Fast-ISel state, if needed.
  FastISel *FastIS = 0;
  if (EnableFastISel)
    FastIS = TLI.createFastISel(MF, MMI, DW,
                                FuncInfo->ValueMap,
                                FuncInfo->MBBMap,
                                FuncInfo->StaticAllocaMap
#ifndef NDEBUG
                                , FuncInfo->CatchInfoLost
#endif
                                );

  // Iterate over all basic blocks in the function.
  for (Function::iterator I = Fn.begin(), E = Fn.end(); I != E; ++I) {
    BasicBlock *LLVMBB = &*I;
    BB = FuncInfo->MBBMap[LLVMBB];

    BasicBlock::iterator const Begin = LLVMBB->begin();
    BasicBlock::iterator const End = LLVMBB->end();
    BasicBlock::iterator BI = Begin;

    // Lower any arguments needed in this block if this is the entry block.
    bool SuppressFastISel = false;
    if (LLVMBB == &Fn.getEntryBlock()) {
      LowerArguments(LLVMBB);

      // If any of the arguments has the byval attribute, forgo
      // fast-isel in the entry block.
      if (FastIS) {
        unsigned j = 1;
        for (Function::arg_iterator I = Fn.arg_begin(), E = Fn.arg_end();
             I != E; ++I, ++j)
          if (Fn.paramHasAttr(j, Attribute::ByVal)) {
            if (EnableFastISelVerbose || EnableFastISelAbort)
              cerr << "FastISel skips entry block due to byval argument\n";
            SuppressFastISel = true;
            break;
          }
      }
    }

    if (MMI && BB->isLandingPad()) {
      // Add a label to mark the beginning of the landing pad.  Deletion of the
      // landing pad can thus be detected via the MachineModuleInfo.
      unsigned LabelID = MMI->addLandingPad(BB);

      const TargetInstrDesc &II = TII.get(TargetInstrInfo::EH_LABEL);
      BuildMI(BB, SDL->getCurDebugLoc(), II).addImm(LabelID);

      // Mark exception register as live in.
      unsigned Reg = TLI.getExceptionAddressRegister();
      if (Reg) BB->addLiveIn(Reg);

      // Mark exception selector register as live in.
      Reg = TLI.getExceptionSelectorRegister();
      if (Reg) BB->addLiveIn(Reg);

      // FIXME: Hack around an exception handling flaw (PR1508): the personality
      // function and list of typeids logically belong to the invoke (or, if you
      // like, the basic block containing the invoke), and need to be associated
      // with it in the dwarf exception handling tables.  Currently however the
      // information is provided by an intrinsic (eh.selector) that can be moved
      // to unexpected places by the optimizers: if the unwind edge is critical,
      // then breaking it can result in the intrinsics being in the successor of
      // the landing pad, not the landing pad itself.  This results in exceptions
      // not being caught because no typeids are associated with the invoke.
      // This may not be the only way things can go wrong, but it is the only way
      // we try to work around for the moment.
      BranchInst *Br = dyn_cast<BranchInst>(LLVMBB->getTerminator());

      if (Br && Br->isUnconditional()) { // Critical edge?
        BasicBlock::iterator I, E;
        for (I = LLVMBB->begin(), E = --LLVMBB->end(); I != E; ++I)
          if (isa<EHSelectorInst>(I))
            break;

        if (I == E)
          // No catch info found - try to extract some from the successor.
          copyCatchInfo(Br->getSuccessor(0), LLVMBB, MMI, *FuncInfo);
      }
    }

    // Before doing SelectionDAG ISel, see if FastISel has been requested.
    if (FastIS && !SuppressFastISel) {
      // Emit code for any incoming arguments. This must happen before
      // beginning FastISel on the entry block.
      if (LLVMBB == &Fn.getEntryBlock()) {
        CurDAG->setRoot(SDL->getControlRoot());
        CodeGenAndEmitDAG();
        SDL->clear();
      }
      FastIS->startNewBlock(BB);
      // Do FastISel on as many instructions as possible.
      for (; BI != End; ++BI) {
        // Just before the terminator instruction, insert instructions to
        // feed PHI nodes in successor blocks.
        if (isa<TerminatorInst>(BI))
          if (!HandlePHINodesInSuccessorBlocksFast(LLVMBB, FastIS)) {
            if (EnableFastISelVerbose || EnableFastISelAbort) {
              cerr << "FastISel miss: ";
              BI->dump();
            }
            if (EnableFastISelAbort)
              assert(0 && "FastISel didn't handle a PHI in a successor");
            break;
          }

        // First try normal tablegen-generated "fast" selection.
        if (FastIS->SelectInstruction(BI))
          continue;

        // Next, try calling the target to attempt to handle the instruction.
        if (FastIS->TargetSelectInstruction(BI))
          continue;

        // Then handle certain instructions as single-LLVM-Instruction blocks.
        if (isa<CallInst>(BI)) {
          if (EnableFastISelVerbose || EnableFastISelAbort) {
            cerr << "FastISel missed call: ";
            BI->dump();
          }

          if (BI->getType() != Type::VoidTy) {
            unsigned &R = FuncInfo->ValueMap[BI];
            if (!R)
              R = FuncInfo->CreateRegForValue(BI);
          }

          SelectBasicBlock(LLVMBB, BI, next(BI));
          // If the instruction was codegen'd with multiple blocks,
          // inform the FastISel object where to resume inserting.
          FastIS->setCurrentBlock(BB);
          continue;
        }

        // Otherwise, give up on FastISel for the rest of the block.
        // For now, be a little lenient about non-branch terminators.
        if (!isa<TerminatorInst>(BI) || isa<BranchInst>(BI)) {
          if (EnableFastISelVerbose || EnableFastISelAbort) {
            cerr << "FastISel miss: ";
            BI->dump();
          }
          if (EnableFastISelAbort)
            // The "fast" selector couldn't handle something and bailed.
            // For the purpose of debugging, just abort.
            assert(0 && "FastISel didn't select the entire block");
        }
        break;
      }
    }

    // Run SelectionDAG instruction selection on the remainder of the block
    // not handled by FastISel. If FastISel is not run, this is the entire
    // block.
    if (BI != End)
      SelectBasicBlock(LLVMBB, BI, End);

    FinishBasicBlock();
  }

  delete FastIS;
}

void
SelectionDAGISel::FinishBasicBlock() {

  DOUT << "Target-post-processed machine code:\n";
  DEBUG(BB->dump());

  DOUT << "Total amount of phi nodes to update: "
       << SDL->PHINodesToUpdate.size() << "\n";
  DEBUG(for (unsigned i = 0, e = SDL->PHINodesToUpdate.size(); i != e; ++i)
          DOUT << "Node " << i << " : (" << SDL->PHINodesToUpdate[i].first
               << ", " << SDL->PHINodesToUpdate[i].second << ")\n";);
  
  // Next, now that we know what the last MBB the LLVM BB expanded is, update
  // PHI nodes in successors.
  if (SDL->SwitchCases.empty() &&
      SDL->JTCases.empty() &&
      SDL->BitTestCases.empty()) {
    for (unsigned i = 0, e = SDL->PHINodesToUpdate.size(); i != e; ++i) {
      MachineInstr *PHI = SDL->PHINodesToUpdate[i].first;
      assert(PHI->getOpcode() == TargetInstrInfo::PHI &&
             "This is not a machine PHI node that we are updating!");
      PHI->addOperand(MachineOperand::CreateReg(SDL->PHINodesToUpdate[i].second,
                                                false));
      PHI->addOperand(MachineOperand::CreateMBB(BB));
    }
    SDL->PHINodesToUpdate.clear();
    return;
  }

  for (unsigned i = 0, e = SDL->BitTestCases.size(); i != e; ++i) {
    // Lower header first, if it wasn't already lowered
    if (!SDL->BitTestCases[i].Emitted) {
      // Set the current basic block to the mbb we wish to insert the code into
      BB = SDL->BitTestCases[i].Parent;
      SDL->setCurrentBasicBlock(BB);
      // Emit the code
      SDL->visitBitTestHeader(SDL->BitTestCases[i]);
      CurDAG->setRoot(SDL->getRoot());
      CodeGenAndEmitDAG();
      SDL->clear();
    }    

    for (unsigned j = 0, ej = SDL->BitTestCases[i].Cases.size(); j != ej; ++j) {
      // Set the current basic block to the mbb we wish to insert the code into
      BB = SDL->BitTestCases[i].Cases[j].ThisBB;
      SDL->setCurrentBasicBlock(BB);
      // Emit the code
      if (j+1 != ej)
        SDL->visitBitTestCase(SDL->BitTestCases[i].Cases[j+1].ThisBB,
                              SDL->BitTestCases[i].Reg,
                              SDL->BitTestCases[i].Cases[j]);
      else
        SDL->visitBitTestCase(SDL->BitTestCases[i].Default,
                              SDL->BitTestCases[i].Reg,
                              SDL->BitTestCases[i].Cases[j]);
        
        
      CurDAG->setRoot(SDL->getRoot());
      CodeGenAndEmitDAG();
      SDL->clear();
    }

    // Update PHI Nodes
    for (unsigned pi = 0, pe = SDL->PHINodesToUpdate.size(); pi != pe; ++pi) {
      MachineInstr *PHI = SDL->PHINodesToUpdate[pi].first;
      MachineBasicBlock *PHIBB = PHI->getParent();
      assert(PHI->getOpcode() == TargetInstrInfo::PHI &&
             "This is not a machine PHI node that we are updating!");
      // This is "default" BB. We have two jumps to it. From "header" BB and
      // from last "case" BB.
      if (PHIBB == SDL->BitTestCases[i].Default) {
        PHI->addOperand(MachineOperand::CreateReg(SDL->PHINodesToUpdate[pi].second,
                                                  false));
        PHI->addOperand(MachineOperand::CreateMBB(SDL->BitTestCases[i].Parent));
        PHI->addOperand(MachineOperand::CreateReg(SDL->PHINodesToUpdate[pi].second,
                                                  false));
        PHI->addOperand(MachineOperand::CreateMBB(SDL->BitTestCases[i].Cases.
                                                  back().ThisBB));
      }
      // One of "cases" BB.
      for (unsigned j = 0, ej = SDL->BitTestCases[i].Cases.size();
           j != ej; ++j) {
        MachineBasicBlock* cBB = SDL->BitTestCases[i].Cases[j].ThisBB;
        if (cBB->succ_end() !=
            std::find(cBB->succ_begin(),cBB->succ_end(), PHIBB)) {
          PHI->addOperand(MachineOperand::CreateReg(SDL->PHINodesToUpdate[pi].second,
                                                    false));
          PHI->addOperand(MachineOperand::CreateMBB(cBB));
        }
      }
    }
  }
  SDL->BitTestCases.clear();

  // If the JumpTable record is filled in, then we need to emit a jump table.
  // Updating the PHI nodes is tricky in this case, since we need to determine
  // whether the PHI is a successor of the range check MBB or the jump table MBB
  for (unsigned i = 0, e = SDL->JTCases.size(); i != e; ++i) {
    // Lower header first, if it wasn't already lowered
    if (!SDL->JTCases[i].first.Emitted) {
      // Set the current basic block to the mbb we wish to insert the code into
      BB = SDL->JTCases[i].first.HeaderBB;
      SDL->setCurrentBasicBlock(BB);
      // Emit the code
      SDL->visitJumpTableHeader(SDL->JTCases[i].second, SDL->JTCases[i].first);
      CurDAG->setRoot(SDL->getRoot());
      CodeGenAndEmitDAG();
      SDL->clear();
    }
    
    // Set the current basic block to the mbb we wish to insert the code into
    BB = SDL->JTCases[i].second.MBB;
    SDL->setCurrentBasicBlock(BB);
    // Emit the code
    SDL->visitJumpTable(SDL->JTCases[i].second);
    CurDAG->setRoot(SDL->getRoot());
    CodeGenAndEmitDAG();
    SDL->clear();
    
    // Update PHI Nodes
    for (unsigned pi = 0, pe = SDL->PHINodesToUpdate.size(); pi != pe; ++pi) {
      MachineInstr *PHI = SDL->PHINodesToUpdate[pi].first;
      MachineBasicBlock *PHIBB = PHI->getParent();
      assert(PHI->getOpcode() == TargetInstrInfo::PHI &&
             "This is not a machine PHI node that we are updating!");
      // "default" BB. We can go there only from header BB.
      if (PHIBB == SDL->JTCases[i].second.Default) {
        PHI->addOperand(MachineOperand::CreateReg(SDL->PHINodesToUpdate[pi].second,
                                                  false));
        PHI->addOperand(MachineOperand::CreateMBB(SDL->JTCases[i].first.HeaderBB));
      }
      // JT BB. Just iterate over successors here
      if (BB->succ_end() != std::find(BB->succ_begin(),BB->succ_end(), PHIBB)) {
        PHI->addOperand(MachineOperand::CreateReg(SDL->PHINodesToUpdate[pi].second,
                                                  false));
        PHI->addOperand(MachineOperand::CreateMBB(BB));
      }
    }
  }
  SDL->JTCases.clear();
  
  // If the switch block involved a branch to one of the actual successors, we
  // need to update PHI nodes in that block.
  for (unsigned i = 0, e = SDL->PHINodesToUpdate.size(); i != e; ++i) {
    MachineInstr *PHI = SDL->PHINodesToUpdate[i].first;
    assert(PHI->getOpcode() == TargetInstrInfo::PHI &&
           "This is not a machine PHI node that we are updating!");
    if (BB->isSuccessor(PHI->getParent())) {
      PHI->addOperand(MachineOperand::CreateReg(SDL->PHINodesToUpdate[i].second,
                                                false));
      PHI->addOperand(MachineOperand::CreateMBB(BB));
    }
  }
  
  // If we generated any switch lowering information, build and codegen any
  // additional DAGs necessary.
  for (unsigned i = 0, e = SDL->SwitchCases.size(); i != e; ++i) {
    // Set the current basic block to the mbb we wish to insert the code into
    BB = SDL->SwitchCases[i].ThisBB;
    SDL->setCurrentBasicBlock(BB);
    
    // Emit the code
    SDL->visitSwitchCase(SDL->SwitchCases[i]);
    CurDAG->setRoot(SDL->getRoot());
    CodeGenAndEmitDAG();
    SDL->clear();
    
    // Handle any PHI nodes in successors of this chunk, as if we were coming
    // from the original BB before switch expansion.  Note that PHI nodes can
    // occur multiple times in PHINodesToUpdate.  We have to be very careful to
    // handle them the right number of times.
    while ((BB = SDL->SwitchCases[i].TrueBB)) {  // Handle LHS and RHS.
      for (MachineBasicBlock::iterator Phi = BB->begin();
           Phi != BB->end() && Phi->getOpcode() == TargetInstrInfo::PHI; ++Phi){
        // This value for this PHI node is recorded in PHINodesToUpdate, get it.
        for (unsigned pn = 0; ; ++pn) {
          assert(pn != SDL->PHINodesToUpdate.size() &&
                 "Didn't find PHI entry!");
          if (SDL->PHINodesToUpdate[pn].first == Phi) {
            Phi->addOperand(MachineOperand::CreateReg(SDL->PHINodesToUpdate[pn].
                                                      second, false));
            Phi->addOperand(MachineOperand::CreateMBB(SDL->SwitchCases[i].ThisBB));
            break;
          }
        }
      }
      
      // Don't process RHS if same block as LHS.
      if (BB == SDL->SwitchCases[i].FalseBB)
        SDL->SwitchCases[i].FalseBB = 0;
      
      // If we haven't handled the RHS, do so now.  Otherwise, we're done.
      SDL->SwitchCases[i].TrueBB = SDL->SwitchCases[i].FalseBB;
      SDL->SwitchCases[i].FalseBB = 0;
    }
    assert(SDL->SwitchCases[i].TrueBB == 0 && SDL->SwitchCases[i].FalseBB == 0);
  }
  SDL->SwitchCases.clear();

  SDL->PHINodesToUpdate.clear();
}


/// Schedule - Pick a safe ordering for instructions for each
/// target node in the graph.
///
ScheduleDAG *SelectionDAGISel::Schedule() {
  RegisterScheduler::FunctionPassCtor Ctor = RegisterScheduler::getDefault();
  
  if (!Ctor) {
    Ctor = ISHeuristic;
    RegisterScheduler::setDefault(Ctor);
  }
  
  ScheduleDAG *Scheduler = Ctor(this, Fast);
  Scheduler->Run(CurDAG, BB, BB->end(), BB->end());

  return Scheduler;
}


ScheduleHazardRecognizer *SelectionDAGISel::CreateTargetHazardRecognizer() {
  return new ScheduleHazardRecognizer();
}

//===----------------------------------------------------------------------===//
// Helper functions used by the generated instruction selector.
//===----------------------------------------------------------------------===//
// Calls to these methods are generated by tblgen.

/// CheckAndMask - The isel is trying to match something like (and X, 255).  If
/// the dag combiner simplified the 255, we still want to match.  RHS is the
/// actual value in the DAG on the RHS of an AND, and DesiredMaskS is the value
/// specified in the .td file (e.g. 255).
bool SelectionDAGISel::CheckAndMask(SDValue LHS, ConstantSDNode *RHS, 
                                    int64_t DesiredMaskS) const {
  const APInt &ActualMask = RHS->getAPIntValue();
  const APInt &DesiredMask = APInt(LHS.getValueSizeInBits(), DesiredMaskS);
  
  // If the actual mask exactly matches, success!
  if (ActualMask == DesiredMask)
    return true;
  
  // If the actual AND mask is allowing unallowed bits, this doesn't match.
  if (ActualMask.intersects(~DesiredMask))
    return false;
  
  // Otherwise, the DAG Combiner may have proven that the value coming in is
  // either already zero or is not demanded.  Check for known zero input bits.
  APInt NeededMask = DesiredMask & ~ActualMask;
  if (CurDAG->MaskedValueIsZero(LHS, NeededMask))
    return true;
  
  // TODO: check to see if missing bits are just not demanded.

  // Otherwise, this pattern doesn't match.
  return false;
}

/// CheckOrMask - The isel is trying to match something like (or X, 255).  If
/// the dag combiner simplified the 255, we still want to match.  RHS is the
/// actual value in the DAG on the RHS of an OR, and DesiredMaskS is the value
/// specified in the .td file (e.g. 255).
bool SelectionDAGISel::CheckOrMask(SDValue LHS, ConstantSDNode *RHS, 
                                   int64_t DesiredMaskS) const {
  const APInt &ActualMask = RHS->getAPIntValue();
  const APInt &DesiredMask = APInt(LHS.getValueSizeInBits(), DesiredMaskS);
  
  // If the actual mask exactly matches, success!
  if (ActualMask == DesiredMask)
    return true;
  
  // If the actual AND mask is allowing unallowed bits, this doesn't match.
  if (ActualMask.intersects(~DesiredMask))
    return false;
  
  // Otherwise, the DAG Combiner may have proven that the value coming in is
  // either already zero or is not demanded.  Check for known zero input bits.
  APInt NeededMask = DesiredMask & ~ActualMask;
  
  APInt KnownZero, KnownOne;
  CurDAG->ComputeMaskedBits(LHS, NeededMask, KnownZero, KnownOne);
  
  // If all the missing bits in the or are already known to be set, match!
  if ((NeededMask & KnownOne) == NeededMask)
    return true;
  
  // TODO: check to see if missing bits are just not demanded.
  
  // Otherwise, this pattern doesn't match.
  return false;
}


/// SelectInlineAsmMemoryOperands - Calls to this are automatically generated
/// by tblgen.  Others should not call it.
void SelectionDAGISel::
SelectInlineAsmMemoryOperands(std::vector<SDValue> &Ops) {
  std::vector<SDValue> InOps;
  std::swap(InOps, Ops);

  Ops.push_back(InOps[0]);  // input chain.
  Ops.push_back(InOps[1]);  // input asm string.

  unsigned i = 2, e = InOps.size();
  if (InOps[e-1].getValueType() == MVT::Flag)
    --e;  // Don't process a flag operand if it is here.
  
  while (i != e) {
    unsigned Flags = cast<ConstantSDNode>(InOps[i])->getZExtValue();
    if ((Flags & 7) != 4 /*MEM*/) {
      // Just skip over this operand, copying the operands verbatim.
      Ops.insert(Ops.end(), InOps.begin()+i, InOps.begin()+i+(Flags >> 3) + 1);
      i += (Flags >> 3) + 1;
    } else {
      assert((Flags >> 3) == 1 && "Memory operand with multiple values?");
      // Otherwise, this is a memory operand.  Ask the target to select it.
      std::vector<SDValue> SelOps;
      if (SelectInlineAsmMemoryOperand(InOps[i+1], 'm', SelOps)) {
        cerr << "Could not match memory address.  Inline asm failure!\n";
        exit(1);
      }
      
      // Add this to the output node.
      MVT IntPtrTy = CurDAG->getTargetLoweringInfo().getPointerTy();
      Ops.push_back(CurDAG->getTargetConstant(4/*MEM*/ | (SelOps.size()<< 3),
                                              IntPtrTy));
      Ops.insert(Ops.end(), SelOps.begin(), SelOps.end());
      i += 2;
    }
  }
  
  // Add the flag input back if present.
  if (e != InOps.size())
    Ops.push_back(InOps.back());
}

char SelectionDAGISel::ID = 0;
