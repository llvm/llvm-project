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
#include "GCNSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-private-object-vgprs"

static cl::opt<unsigned>
    RegChunkSizeInDWords("private-object-reg-chunk-size", cl::Hidden,
                         cl::desc("Number of 32-bit VGPRs per register chunk "
                                  "for promoted private objects"),
                         cl::init(1));

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
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS(AMDGPUPrivateObjectVGPRs, DEBUG_TYPE,
                "AMDGPU Add defs/uses for private object VGPRs", false, false)

char AMDGPUPrivateObjectVGPRs::ID = 0;

char &llvm::AMDGPUPrivateObjectVGPRsID = AMDGPUPrivateObjectVGPRs::ID;

FunctionPass *llvm::createAMDGPUPrivateObjectVGPRsPass() {
  return new AMDGPUPrivateObjectVGPRs();
}

// Return metadata describing the object the passed instruction refers to,
// if any, or nullptr otherwise.
static const MDNode *getPromotedPrivateObject(const MachineInstr &MI) {
  if (MI.getOpcode() != AMDGPU::V_LOAD_IDX &&
      MI.getOpcode() != AMDGPU::V_STORE_IDX)
    return nullptr;

  const MachineMemOperand *MMO = *MI.memoperands_begin();
  assert(MMO);
  const Value *Ptr = MMO->getValue();
  if (!Ptr)
    return nullptr;

  if (auto *GEP = dyn_cast<GEPOperator>(Ptr))
    Ptr = GEP->getPointerOperand();
  const auto *Alloca = dyn_cast<AllocaInst>(Ptr);
  if (!Alloca)
    return nullptr;

  const MDNode *Obj = Alloca->getMetadata("amdgpu.allocated.vgprs");
  if (!Obj)
    return nullptr;

  return Obj;
}

static void getObjectRegs(const MDNode *Obj, SmallVectorImpl<MCPhysReg> &Regs,
                          const SIRegisterInfo &TRI) {
  unsigned Offset =
      cast<ConstantInt>(
          cast<ConstantAsMetadata>(Obj->getOperand(0))->getValue())
          ->getZExtValue();
  unsigned Size = cast<ConstantInt>(
                      cast<ConstantAsMetadata>(Obj->getOperand(1))->getValue())
                      ->getZExtValue();

  assert(Offset % 4 == 0 && Size % 4 == 0);
  unsigned RegWidth = RegChunkSizeInDWords * 32;
  const TargetRegisterClass *BaseRegRC =
      TRI.getAnyVGPRClassForBitWidth(RegWidth);
  if (!BaseRegRC)
    report_fatal_error("Invalid VGPR width " + Twine(RegWidth));
  unsigned BaseRegIdx = Offset / 4;
  MCPhysReg BaseReg = BaseRegRC->getRegister(BaseRegIdx);
  unsigned NumRegs = Size / (RegChunkSizeInDWords * 4);
  for (unsigned I : seq(NumRegs))
    Regs.push_back(BaseReg + I * RegChunkSizeInDWords);

  if (unsigned LastChunkSize = Size % (RegChunkSizeInDWords * 4)) {
    unsigned LastRegWidth = LastChunkSize * 8;
    const TargetRegisterClass *LastRegRC =
        TRI.getAnyVGPRClassForBitWidth(LastRegWidth);
    if (!LastRegRC)
      report_fatal_error("Invalid VGPR width " + Twine(LastRegWidth));
    unsigned LastRegIdx = BaseRegIdx + RegChunkSizeInDWords * NumRegs;
    Regs.push_back(LastRegRC->getRegister(LastRegIdx));
  }
}

using RegVector = SmallVector<MCPhysReg, 50>;

static void insertObjectDef(MachineInstr &MI, const MDNode *Obj,
                            const SIInstrInfo &TII, const SIRegisterInfo &TRI) {
  RegVector Regs;
  getObjectRegs(Obj, Regs, TRI);

  MachineBasicBlock::instr_iterator DefPt = MI.getIterator();
  while (DefPt->isBundledWithPred())
    --DefPt;

  for (MCPhysReg Reg : Regs) {
    BuildMI(*MI.getParent(), DefPt, DebugLoc(),
            TII.get(TargetOpcode::IMPLICIT_DEF), Reg);
  }
}

static void addUseDefOperands(MachineInstr &MI, const MDNode *Obj,
                              const SIRegisterInfo &TRI) {
  assert(MI.getOpcode() == AMDGPU::V_LOAD_IDX ||
         MI.getOpcode() == AMDGPU::V_STORE_IDX);

  RegVector Regs;
  getObjectRegs(Obj, Regs, TRI);

  for (MCPhysReg Reg : Regs) {
    // In general case, we don't know which VGPRs are read or written, so
    // we conservatively assume V_LOAD_IDX pseudos load all of them and
    // V_STORE_IDX store only some of them, meaning V_STORE_IDX have to
    // have both defs and uses for all the registers.
    // TODO: In cases with constant GEPs where we can realiably determine
    // the accessed VGPRs we don't need to add defs/uses for all registers
    // and V_STORE_IDX don't need to have implicit uses.
    MI.addOperand(
        MachineOperand::CreateReg(Reg, /*isDef=*/false, /*isImp=*/true));

    if (MI.getOpcode() == AMDGPU::V_STORE_IDX) {
      MI.addOperand(
          MachineOperand::CreateReg(Reg, /*isDef=*/true, /*isImp=*/true));
    }
  }
}

static void addLiveInRegs(MachineBasicBlock &MBB, const MDNode *Obj,
                          const SIRegisterInfo &TRI) {
  RegVector Regs;
  getObjectRegs(Obj, Regs, TRI);

  for (MCPhysReg Reg : Regs)
    MBB.addLiveIn(Reg);
}

bool AMDGPUPrivateObjectVGPRs::runOnMachineFunction(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!ST.hasVGPRIndexingRegisters())
    return false;

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
      if (const MDNode *Obj = getPromotedPrivateObject(MI);
          Obj && !is_contained(BlockReachable, Obj)) {
        BlockReachable.push_back(Obj);
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
      if (const MDNode *Obj = getPromotedPrivateObject(MI);
          Obj && !is_contained(LiveObjs, Obj))
        LiveObjs.push_back(Obj);
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
  const SIInstrInfo &TII = *ST.getInstrInfo();
  const SIRegisterInfo &TRI = *ST.getRegisterInfo();
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    SmallVector<const MDNode *, 4> LiveObjs = LiveIns[&MBB];
    for (const MDNode *Obj : LiveObjs)
      addLiveInRegs(MBB, Obj, TRI);

    for (MachineInstr &MI : MBB.instrs()) {
      if (const MDNode *Obj = getPromotedPrivateObject(MI)) {
        addUseDefOperands(MI, Obj, TRI);
        if (!is_contained(LiveObjs, Obj)) {
          LiveObjs.push_back(Obj);
          insertObjectDef(MI, Obj, TII, TRI);
        }
        Changed = true;
      }
    }

    for (MachineBasicBlock *Succ : MBB.successors()) {
      for (const MDNode *Obj : LiveIns[Succ]) {
        if (!is_contained(LiveObjs, Obj)) {
          LiveObjs.push_back(Obj);
          insertObjectDef(*MBB.getFirstTerminator(), Obj, TII, TRI);
          Changed = true;
        }
      }
    }
  }

  return Changed;
}
