#if LLPC_BUILD_NPI
//===----------- AMDGPUPrivateObjectVGPRs.cpp - Private object VGPRs ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Add implicit use/def operands to V_LOAD/STORE_IDX pseudos for VGPRs
/// allocated to promoted private objects and thus prevent the register
/// allocator from using these VGPRs where the private objects are live.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUVGPRIndexingAnalysis.h"
#include "GCNSubtarget.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-private-object-vgprs"

static cl::opt<unsigned>
    RegChunkSizeInDWords("private-object-reg-chunk-size", cl::Hidden,
                         cl::desc("Number of 32-bit VGPRs per register chunk "
                                  "for promoted private objects"),
                         cl::init(1));

using ObjectRegs = SmallVector<MCPhysReg, 50>;

namespace {

class AMDGPUPrivateObjectVGPRs : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUPrivateObjectVGPRs() : MachineFunctionPass(ID) {
    initializeAMDGPUPrivateObjectVGPRsPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Def/use private object VGPRs";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AMDGPUIndexingInfoWrapper>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  // Private/shared indexing analysis
  const AMDGPUIndexingInfo *SII;
  const SIRegisterInfo *TRI;
  const SIInstrInfo *TII;
  // Store computed object regs with the associated PO Info, (un)known precise
  // reg use information determines the computed regs.
  std::unordered_map<const AMDGPUPrivateObjectIdxInfo *, ObjectRegs>
      FilteredObjectRegs;
  // Store a more generic set of regs that are computed always with no precise
  // use information.
  std::unordered_map<const MDNode *, ObjectRegs> MDNodeObjectRegs;

  ObjectRegs &getObjectRegs(const MDNode &Obj);
  ObjectRegs &getObjectRegs(const AMDGPUPrivateObjectIdxInfo &PO);

  void updateObjectRegs(
      ObjectRegs &Regs, unsigned Offset, unsigned Size,
      std::optional<std::pair<unsigned, unsigned>> RelativeUsedRegs);

  void insertObjectDef(MachineBasicBlock::instr_iterator DefPt,
                       const MDNode &Obj, MachineBasicBlock &MBB);

  void addUseDefOperands(MachineInstr &MI,
                         const AMDGPUPrivateObjectIdxInfo &PO);

  void addLiveInRegs(MachineBasicBlock &MBB, const MDNode &Obj);
};

} // End anonymous namespace.

INITIALIZE_PASS(AMDGPUPrivateObjectVGPRs, DEBUG_TYPE,
                "AMDGPU Add defs/uses for private object VGPRs", false, false)

char AMDGPUPrivateObjectVGPRs::ID = 0;

char &llvm::AMDGPUPrivateObjectVGPRsID = AMDGPUPrivateObjectVGPRs::ID;

FunctionPass *llvm::createAMDGPUPrivateObjectVGPRsPass() {
  return new AMDGPUPrivateObjectVGPRs();
}

void AMDGPUPrivateObjectVGPRs::updateObjectRegs(
    ObjectRegs &Regs, unsigned Offset, unsigned Size,
    std::optional<std::pair<unsigned, unsigned>> RelativeUsedRegs = {}) {
  assert(Offset % 4 == 0 && Size % 4 == 0);
  unsigned RegWidth = RegChunkSizeInDWords * 32;
  const TargetRegisterClass *BaseRegRC =
      TRI->getAnyVGPRClassForBitWidth(RegWidth);
  if (!BaseRegRC)
    report_fatal_error("Invalid VGPR width " + Twine(RegWidth));
  unsigned BaseRegIdx = Offset / 4;
  MCPhysReg BaseReg = BaseRegRC->getRegister(BaseRegIdx);
  unsigned NumRegs = Size / (RegChunkSizeInDWords * 4);
  for (unsigned I : seq(NumRegs)) {
    MCPhysReg Reg = BaseReg + I * RegChunkSizeInDWords;
    if (!RelativeUsedRegs || (Reg >= BaseReg + RelativeUsedRegs->first &&
                              Reg < BaseReg + RelativeUsedRegs->second))
      Regs.push_back(Reg);
  }

  if (unsigned LastChunkSize = Size % (RegChunkSizeInDWords * 4)) {
    unsigned LastRegWidth = LastChunkSize * 8;
    const TargetRegisterClass *LastRegRC =
        TRI->getAnyVGPRClassForBitWidth(LastRegWidth);
    if (!LastRegRC)
      report_fatal_error("Invalid VGPR width " + Twine(LastRegWidth));
    unsigned LastRegIdx = BaseRegIdx + RegChunkSizeInDWords * NumRegs;
    MCPhysReg Reg = LastRegRC->getRegister(LastRegIdx);
    MCPhysReg ChunkReg = BaseRegRC->getRegister(LastRegIdx);
    // Filter using the larger chunk size, as that's covered by the object.
    if (!RelativeUsedRegs || (ChunkReg >= BaseReg + RelativeUsedRegs->first &&
                              ChunkReg < BaseReg + RelativeUsedRegs->second))
      Regs.push_back(Reg);
  }
}

ObjectRegs &
AMDGPUPrivateObjectVGPRs::getObjectRegs(const AMDGPUPrivateObjectIdxInfo &PO) {
  auto &Regs = FilteredObjectRegs[&PO];
  if (!Regs.empty())
    return Regs;
  // The (un)known regs used by this PO relative to itself (as opposed to the
  // base of laneshared space).
  std::optional<std::pair<unsigned, unsigned>> RelativeUsedRegs;
  if (PO.UsedRegs.has_value()) {
    std::pair<unsigned, unsigned> Result = *PO.UsedRegs;
    // Shift interval to only include the immediate offset.
    Result.first -= SII->getLaneSharedSize() + PO.Offset / 4;
    Result.second -= SII->getLaneSharedSize() + PO.Offset / 4;
    // Widen the interval to the nearest chunk.
    Result.first =
        alignDown(Result.first, static_cast<unsigned>(RegChunkSizeInDWords));
    Result.second =
        alignTo(Result.second, static_cast<unsigned>(RegChunkSizeInDWords));
    RelativeUsedRegs = Result;
  }
  updateObjectRegs(Regs, PO.Offset, PO.Size, RelativeUsedRegs);
  return Regs;
}

ObjectRegs &AMDGPUPrivateObjectVGPRs::getObjectRegs(const MDNode &Obj) {
  auto &Regs = MDNodeObjectRegs[&Obj];
  if (Regs.empty()) {
    auto [Offset, Size] = getAMDGPUPrivateObjectNodeInfo(&Obj);
    updateObjectRegs(Regs, Offset, Size);
  }
  return Regs;
}

void AMDGPUPrivateObjectVGPRs::insertObjectDef(
    MachineBasicBlock::instr_iterator DefPt, const MDNode &Obj,
    MachineBasicBlock &MBB) {
  ObjectRegs &Regs = getObjectRegs(Obj);

  if (DefPt != MBB.instr_end()) {
    while (DefPt->isBundledWithPred())
      --DefPt;
  }

  for (MCPhysReg Reg : Regs)
    BuildMI(MBB, DefPt, DebugLoc(), TII->get(TargetOpcode::IMPLICIT_DEF), Reg);
}

void AMDGPUPrivateObjectVGPRs::addUseDefOperands(
    MachineInstr &MI, const AMDGPUPrivateObjectIdxInfo &PO) {
  assert(MI.getOpcode() == AMDGPU::V_LOAD_IDX ||
         MI.getOpcode() == AMDGPU::V_STORE_IDX);
  ObjectRegs &Regs = getObjectRegs(PO);

  MachineInstr *Bundle = nullptr;
  if (MI.isBundled())
    Bundle = &*getBundleStart(MI.getIterator());

  for (MCPhysReg Reg : Regs) {
    // In the case where no private object interval is available,
    // we conservatively assume V_LOAD_IDX loads all registers assigned to the
    // object and V_STORE_IDX store only some of them, meaning V_STORE_IDX have
    // to have both defs and uses for all the registers. When interval is
    // available V_STORE_IDX only needs defs on relevant registers.
    if ((MI.getOpcode() == AMDGPU::V_LOAD_IDX) || !PO.UsedRegs.has_value()) {
      MachineOperand UseOp =
          MachineOperand::CreateReg(Reg, /*isDef=*/false, /*isImp=*/true);
      MI.addOperand(UseOp);
      if (Bundle && !Bundle->hasRegisterImplicitUseOperand(Reg))
        Bundle->addOperand(UseOp);
    }

    if (MI.getOpcode() == AMDGPU::V_STORE_IDX) {
      MachineOperand StoreOp =
          MachineOperand::CreateReg(Reg, /*isDef=*/true, /*isImp=*/true);
      MI.addOperand(StoreOp);
      if (Bundle && (Bundle->findRegisterDefOperandIdx(Reg, TRI) == -1))
        Bundle->addOperand(StoreOp);
    }
  }
}

void AMDGPUPrivateObjectVGPRs::addLiveInRegs(MachineBasicBlock &MBB,
                                             const MDNode &Obj) {
  ObjectRegs &Regs = getObjectRegs(Obj);

  for (MCPhysReg Reg : Regs)
    MBB.addLiveIn(Reg);
}

bool AMDGPUPrivateObjectVGPRs::runOnMachineFunction(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!ST.hasVGPRIndexingRegisters())
    return false;
  SII = &getAnalysis<AMDGPUIndexingInfoWrapper>().getIndexingInfo();
  TII = ST.getInstrInfo();
  TRI = ST.getRegisterInfo();
  FilteredObjectRegs.clear();
  MDNodeObjectRegs.clear();

  // Compute reachable objects. Use SmallVector to avoid wasting memory
  // storing lots of empty or nearly empty sets.
  DenseMap<const MachineBasicBlock *, SmallVector<const MDNode *, 4>>
      ReachableObjs;
  SmallSet<MachineBasicBlock *, 16> Worklist;
  for (MachineBasicBlock &MBB : MF)
    Worklist.insert(&MBB);
  while (!Worklist.empty()) {
    MachineBasicBlock &MBB = **Worklist.begin();
    Worklist.erase(&MBB);
    SmallVectorImpl<const MDNode *> &BlockReachable = ReachableObjs[&MBB];
    bool Updated = false;
    for (MachineInstr &MI : MBB.instrs()) {
      if (auto Info = SII->getPrivateObjectIdxInfo(&MI);
          Info && !is_contained(BlockReachable, Info->get().Obj)) {
        BlockReachable.push_back(Info->get().Obj);
        Updated = true;
      }
    }
    for (MachineBasicBlock *Succ : MBB.successors()) {
      for (const MDNode *Obj : ReachableObjs[Succ]) {
        if (!is_contained(BlockReachable, Obj)) {
          BlockReachable.push_back(Obj);
          Updated = true;
        }
      }
    }
    if (Updated)
      Worklist.insert(MBB.predecessors().begin(), MBB.predecessors().end());
  }

  // Collect promoted private objects that must be live at the
  // beginnings of basic blocks (live-ins). All live-in objects of a
  // block and all objects referred to in that block and live-ins
  // of any successors of that block must also be live-ins of all its
  // successors, unless they are unreachable there.
  DenseMap<const MachineBasicBlock *, SmallVector<const MDNode *, 4>> LiveIns;
  for (MachineBasicBlock &MBB : MF)
    Worklist.insert(&MBB);
  while (!Worklist.empty()) {
    // Gather objects that are live at the end of the block.
    MachineBasicBlock &MBB = **Worklist.begin();
    Worklist.erase(&MBB);
    SmallVector<const MDNode *, 4> LiveObjs = LiveIns[&MBB];
    for (MachineInstr &MI : MBB.instrs()) {
      if (auto Info = SII->getPrivateObjectIdxInfo(&MI);
          Info && !is_contained(LiveObjs, Info->get().Obj))
        LiveObjs.push_back(Info->get().Obj);
    }

    // Add objects that must be defined at the beginnings of successors.
    for (MachineBasicBlock *Succ : MBB.successors()) {
      for (const MDNode *Obj : LiveIns[Succ]) {
        if (!is_contained(LiveObjs, Obj))
          LiveObjs.push_back(Obj);
      }
    }

    // Propagate all objects that are defined in this block to successors.
    for (MachineBasicBlock *Succ : MBB.successors()) {
      SmallVectorImpl<const MDNode *> &SuccLiveIns = LiveIns[Succ];
      const SmallVectorImpl<const MDNode *> &Reachables = ReachableObjs[Succ];
      for (const MDNode *Obj : LiveObjs) {
        if (!is_contained(SuccLiveIns, Obj) && is_contained(Reachables, Obj)) {
          SuccLiveIns.push_back(Obj);
          Worklist.insert(Succ);
          for (MachineBasicBlock *Pred : Succ->predecessors()) {
            if (Pred != &MBB)
              Worklist.insert(Pred);
          }
        }
      }
    }
  }

  // Add use/def operands and live-in registers, and insert object
  // definitions. Object definitions are inserted where control
  // transitions from where objects don't have to be live to where they
  // must be live. For all objects that are not live-ins of a block but
  // used in it we insert a definition right before the corresponding
  // V_LOAD/STORE_IDX pseudo. Also, if an object must be defined in a
  // successor of a block but is not a live-in of that block nor it is
  // referred to within that block, we insert a definition for that
  // object just before the first terminator of the block.
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    SmallVector<const MDNode *, 4> LiveObjs = LiveIns[&MBB];
    for (const MDNode *Obj : LiveObjs)
      addLiveInRegs(MBB, *Obj);

    for (MachineInstr &MI : MBB.instrs()) {
      if (auto Info = SII->getPrivateObjectIdxInfo(&MI)) {
        const AMDGPUPrivateObjectIdxInfo &PO = Info->get();
        addUseDefOperands(MI, PO);
        if (!is_contained(LiveObjs, PO.Obj)) {
          LiveObjs.push_back(PO.Obj);
          insertObjectDef(MI.getIterator(), *PO.Obj, MBB);
        }
        Changed = true;
      }
    }

    for (MachineBasicBlock *Succ : MBB.successors()) {
      for (const MDNode *Obj : LiveIns[Succ]) {
        if (!is_contained(LiveObjs, Obj)) {
          LiveObjs.push_back(Obj);
          insertObjectDef(MBB.getFirstInstrTerminator(), *Obj, MBB);
          Changed = true;
        }
      }
    }
  }

  return Changed;
}
#endif /* LLPC_BUILD_NPI */
