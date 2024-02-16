//===-- HexagonPostIncOpt.cpp - Hexagon Post Increment Optimization Pass --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Convert post-inc addressing mode into base-offset addressing mode.
// Ex:
// original loop:
// v1 = phi(v0, v3)
// v2,v3 = post_load v1, 4

// Often, unroller creates below form of post-increments:
// v1 = phi(v0, v3')
// v2,v3  = post_load v1, 4
// v2',v3'= post_load v3, 4

// This can be optimized in two ways

// 1.
// v1 = phi(v0, v3')
// v2,v3' = post_load v1, 8
// v2' = load v3', -4
//
// 2.
// v1 = phi(v0, v3')
// v2,v3' = post_load v1, 8
// v2' = load v1, 4
//
// Option 2 is favored as we can packetize two memory operations in a single
// packet. However, this is not always favorable due to memory dependences
// and in cases where we form a bigger chain of post-increment ops that will
// create more spills as we can not execute post-increment ops with out
// executing base-offset instructions.
//===----------------------------------------------------------------------===//
#include "HexagonInstrInfo.h"
#include "HexagonSubtarget.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "hexagon-postincopt"

static cl::opt<unsigned> PostIncChainThreshold(
    "post-inc-chain-threshold", cl::Hidden, cl::init(4),
    cl::desc("Limit the number of post-inc instructions in a chain."));

static cl::opt<bool> PreferPostIncStore(
    "prefer-post-inc-store", cl::Hidden, cl::init(true),
    cl::desc("Prefer post-inc store in a list of loads and stores."));

namespace llvm {
void initializeHexagonPostIncOptPass(PassRegistry &);
FunctionPass *createHexagonPostIncOpt();
} // namespace llvm

namespace {

class HexagonPostIncOpt : public MachineFunctionPass {
  MachineLoopInfo *MLI = nullptr;
  const HexagonInstrInfo *HII = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  const MachineRegisterInfo *MRI = nullptr;
  const HexagonSubtarget *HST = nullptr;

public:
  static char ID;

  HexagonPostIncOpt() : MachineFunctionPass(ID) {
    initializeHexagonPostIncOptPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AAResultsWrapperPass>();
    AU.addPreserved<AAResultsWrapperPass>();
    AU.addRequired<MachineLoopInfo>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return "Hexagon Post-Inc-Opt Pass"; }

  bool runOnMachineFunction(MachineFunction &Fn) override;

private:
  bool translatePostIncsInLoop(MachineBasicBlock &MBB);
  void replacePostIncWithBaseOffset(MachineBasicBlock &MBB) const;
  void replacePostIncWithBaseOffset(MachineInstr &MI) const;
  bool isPostIncInsn(MachineInstr &MI) const;
  void foldAdds(MachineBasicBlock &MBB) const;
  void updateBaseAndOffset(MachineInstr &MI, MachineInstr &AddMI) const;
  void removeDeadInstructions(MachineBasicBlock &MBB) const;

  void generatePostInc(MachineBasicBlock &MBB);
  bool canReplaceWithPostInc(MachineInstr *MI, MachineInstr *AddMI) const;
  void replaceWithPostInc(MachineInstr *MI, MachineInstr *AddMI) const;

  bool isValidOffset(const MachineInstr &MI, int64_t Offset) const;
  bool isValidPostIncValue(const MachineInstr &MI, int IncVal) const;
};

class HexagonPostIncOptSchedDAG : public ScheduleDAGInstrs {
  HexagonPostIncOpt &Pass;

public:
  HexagonPostIncOptSchedDAG(HexagonPostIncOpt &P, MachineFunction &MF,
                            MachineLoopInfo *MLI)
      : ScheduleDAGInstrs(MF, MLI, false), Pass(P){};
  void schedule() override;
  ScheduleDAGTopologicalSort &getTopo() { return Topo; };
};

} // End anonymous namespace.

char HexagonPostIncOpt::ID = 0;

INITIALIZE_PASS_BEGIN(HexagonPostIncOpt, DEBUG_TYPE,
                      "Hexagon Post-Inc-Opt Pass", false, false)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_END(HexagonPostIncOpt, DEBUG_TYPE, "Hexagon Post-Inc-Opt Pass",
                    false, false)

/// Return true if MIA dominates MIB.
static bool dominates(MachineInstr *MIA, MachineInstr *MIB) {
  if (MIA->getParent() != MIB->getParent())
    return false; // Don't know since machine dominator tree is out of date.

  MachineBasicBlock *MBB = MIA->getParent();
  MachineBasicBlock::iterator I = MBB->instr_begin();
  // Iterate over the basic block until MIA or MIB is found.
  for (; &*I != MIA && &*I != MIB; ++I)
    ;

  // MIA dominates MIB if MIA is found first.
  return &*I == MIA;
}

// Return the Phi register value that comes from the loop block.
static unsigned getLoopPhiReg(MachineInstr *Phi, MachineBasicBlock *LoopBB) {
  for (unsigned i = 1, e = Phi->getNumOperands(); i != e; i += 2)
    if (Phi->getOperand(i + 1).getMBB() == LoopBB)
      return Phi->getOperand(i).getReg();
  return UINT_MAX;
}

static bool isAddWithImmValue(const MachineInstr &MI) {
  // FIXME: For now, only deal with adds that have strict immediate values.
  // Some A2_addi instructions can be of the form.
  // %338:intregs = A2_addi %7:intregs, @_ZL7phs_tbl + 16
  return MI.getOpcode() == Hexagon::A2_addi && MI.getOperand(2).isImm();
}

// Compute the number of 'real' instructions in the basic block by
// ignoring terminators.
static unsigned getBasicBlockSize(MachineBasicBlock &MBB) {
  unsigned size = 0;
  for (auto &I : make_range(MBB.begin(), MBB.getFirstTerminator()))
    if (!I.isDebugInstr())
      size++;
  return size;
}

// Setup Post increment Schedule DAG.
static void initPISchedDAG(HexagonPostIncOptSchedDAG &PIDAG,
                           MachineBasicBlock &MBB) {
  PIDAG.startBlock(&MBB);
  PIDAG.enterRegion(&MBB, MBB.begin(), MBB.getFirstTerminator(),
                    getBasicBlockSize(MBB));
  // Build the graph.
  PIDAG.schedule();
  // exitRegion() is an empty function in base class. So, safe to call it here.
  PIDAG.exitRegion();
}

// Check if post-increment candidate has any memory dependence on any
// instruction in the chain.
static bool hasMemoryDependency(SUnit *PostIncSU,
                                SmallVector<MachineInstr *, 4> &UseList) {

  // FIXME: Fine tune the order dependence. Probably can only consider memory
  // related OrderKind.
  for (auto &Dep : PostIncSU->Succs)
    if (Dep.getKind() == SDep::Order)
      if (std::find(UseList.begin(), UseList.end(),
                    Dep.getSUnit()->getInstr()) != UseList.end())
        return true;

  return false;
}

// Fold an add with immediate into either an add or a load or a store.
void HexagonPostIncOpt::foldAdds(MachineBasicBlock &MBB) const {
  LLVM_DEBUG(dbgs() << "#Fold add instructions in this block.\n");
  for (auto &MI : make_range(MBB.getFirstNonPHI(), MBB.getFirstTerminator())) {
    if (!isAddWithImmValue(MI))
      continue;
    unsigned DefReg = MI.getOperand(0).getReg();
    unsigned AddReg = MI.getOperand(1).getReg();
    int64_t AddImm = MI.getOperand(2).getImm();

    SmallVector<MachineInstr *, 4> UseList;
    // Gather the uses of add instruction's def reg.
    for (auto &MO : make_range(MRI->use_begin(DefReg), MRI->use_end())) {
      MachineInstr *UseMI = MO.getParent();
      // Deal with only the instuctions that belong to this block.
      // If we cross this block, the generation of post-increment logic
      // will not be able to transform to post-inc due to dominance.
      if (UseMI->getParent() == &MBB)
        UseList.push_back(UseMI);
    }

    if (UseList.empty())
      continue;

    LLVM_DEBUG({
      dbgs() << "Current instruction considered for folding \n";
      MI.dump();
    });

    for (auto UseMI : UseList) {
      if (isAddWithImmValue(*UseMI)) {
        int64_t NewImm = AddImm + UseMI->getOperand(2).getImm();
        // Fold if the new immediate is with in the range.
        if (HII->isValidOffset(UseMI->getOpcode(), NewImm, TRI, false)) {
          LLVM_DEBUG({
            UseMI->dump();
            dbgs() << "\t is folded in to \n";
          });
          UseMI->getOperand(1).setReg(AddReg);
          UseMI->getOperand(2).setImm(NewImm);
          LLVM_DEBUG(UseMI->dump());
        }
      } else if (HII->isBaseImmOffset(*UseMI)) {
        LLVM_DEBUG({
          UseMI->dump();
          dbgs() << "\t is folded in to \n";
        });
        updateBaseAndOffset(*UseMI, MI);
        LLVM_DEBUG(UseMI->dump());
      }
      LLVM_DEBUG(dbgs() << "\n");
    }
  }
  removeDeadInstructions(MBB);
  LLVM_DEBUG(dbgs() << "#End of the fold instructions logic.\n");
}

void HexagonPostIncOpt::updateBaseAndOffset(MachineInstr &MI,
                                            MachineInstr &AddMI) const {
  assert(HII->isBaseImmOffset(MI));
  unsigned BasePos, OffsetPos;
  if (!HII->getBaseAndOffsetPosition(MI, BasePos, OffsetPos))
    return;

  MachineOperand &OffsetOp = MI.getOperand(OffsetPos);
  MachineOperand &BaseOp = MI.getOperand(BasePos);

  if (BaseOp.getReg() != AddMI.getOperand(0).getReg())
    return;

  unsigned IncBase = AddMI.getOperand(1).getReg();
  int64_t IncValue = AddMI.getOperand(2).getImm();

  int64_t NewOffset = OffsetOp.getImm() + IncValue;
  if (!isValidOffset(MI, NewOffset))
    return;

  OffsetOp.setImm(NewOffset);
  BaseOp.setReg(IncBase);
}

void HexagonPostIncOpt::removeDeadInstructions(MachineBasicBlock &MBB) const {
  // For MBB, check that the value defined by each instruction is used.
  // If not, delete it.
  for (MachineBasicBlock::reverse_instr_iterator MI = MBB.instr_rbegin(),
                                                 ME = MBB.instr_rend();
       MI != ME;) {
    // From DeadMachineInstructionElem. Don't delete inline assembly.
    if (MI->isInlineAsm()) {
      ++MI;
      continue;
    }
    bool SawStore = false;
    // Check if it's safe to remove the instruction due to side effects.
    if (!MI->isSafeToMove(nullptr, SawStore)) {
      ++MI;
      continue;
    }
    unsigned Uses = 0;
    for (MachineInstr::mop_iterator MOI = MI->operands_begin(),
                                    MOE = MI->operands_end();
         MOI != MOE; ++MOI) {
      if (!MOI->isReg() || !MOI->isDef())
        continue;
      unsigned reg = MOI->getReg();
      // Assume physical registers are used.
      if (Register::isPhysicalRegister(reg)) {
        Uses++;
        continue;
      }
      if (MRI->use_begin(reg) != MRI->use_end())
        Uses++;
    }
    if (!Uses) {
      MI++->eraseFromParent();
      continue;
    }
    ++MI;
  }
}

bool HexagonPostIncOpt::isPostIncInsn(MachineInstr &MI) const {
  // Predicated post-increments are not yet handled. (ISel is not generating
  // them yet). Circular buffer instructions should not be handled.
  return (HII->isPostIncWithImmOffset(MI) && !HII->isPredicated(MI) &&
          !HII->isCircBufferInstr(MI));
}

/// For instructions with a base and offset, return true if the new Offset
/// is a valid value with the correct alignment.
bool HexagonPostIncOpt::isValidOffset(const MachineInstr &MI,
                                      int64_t Offset) const {
  if (!HII->isValidOffset(MI.getOpcode(), Offset, TRI, false))
    return false;
  unsigned AlignMask = HII->getMemAccessSize(MI) - 1;
  return (Offset & AlignMask) == 0;
}

bool HexagonPostIncOpt::isValidPostIncValue(const MachineInstr &MI,
                                            int IncVal) const {
  unsigned AlignMask = HII->getMemAccessSize(MI) - 1;
  if ((IncVal & AlignMask) != 0)
    return false;

  // Number of total bits in the instruction used to encode Inc value.
  unsigned IncBits = 4;
  // For HVX instructions, the offset is 3.
  if (HexagonII::isCVI(MI.getDesc()))
    IncBits = 3;

  IncBits += Log2_32(HII->getMemAccessSize(MI));
  if (HII->getMemAccessSize(MI) > 8)
    IncBits = 16;

  int MinValidVal = -1U << (IncBits - 1);
  int MaxValidVal = ~(-1U << (IncBits - 1));
  return (IncVal >= MinValidVal && IncVal <= MaxValidVal);
}

void HexagonPostIncOptSchedDAG::schedule() {
  AliasAnalysis *AA = &Pass.getAnalysis<AAResultsWrapperPass>().getAAResults();
  buildSchedGraph(AA);
}

// Replace post-increment operations with base+offset counterpart.
void HexagonPostIncOpt::replacePostIncWithBaseOffset(
    MachineBasicBlock &MBB) const {
  LLVM_DEBUG(dbgs() << "#Replacing post-increment instructions with "
                       "base+offset counterparts.\n");

  SmallVector<MachineInstr *, 4> MIList;
  for (auto &MI : make_range(MBB.getFirstNonPHI(), MBB.getFirstTerminator())) {
    // Check for eligible post-inc candidates.
    if (!isPostIncInsn(MI))
      continue;
    MIList.push_back(&MI);
  }

  for (auto MI : MIList)
    replacePostIncWithBaseOffset(*MI);

  LLVM_DEBUG(dbgs() << "#Done with replacing post-increment instructions.\n");
}

void HexagonPostIncOpt::replacePostIncWithBaseOffset(MachineInstr &MI) const {
  short NewOpcode = HII->changeAddrMode_pi_io(MI.getOpcode());
  if (NewOpcode < 0)
    return;

  unsigned BasePos = 0, OffsetPos = 0;
  if (!HII->getBaseAndOffsetPosition(MI, BasePos, OffsetPos))
    return;
  const MachineOperand &PostIncOffset = MI.getOperand(OffsetPos);
  const MachineOperand &PostIncBase = MI.getOperand(BasePos);

  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = MI.getDebugLoc();
  MachineOperand *PostIncDest;
  MachineInstrBuilder MIB;
  if (MI.mayLoad()) {
    PostIncDest = &MI.getOperand(1);
    const MachineOperand &LDValue = MI.getOperand(0);
    MIB = BuildMI(MBB, MI, DL, HII->get(NewOpcode));
    MIB.add(LDValue).add(PostIncBase).addImm(0);
  } else {
    PostIncDest = &MI.getOperand(0);
    const MachineOperand &STValue = MI.getOperand(3);
    MIB = BuildMI(MBB, MI, DL, HII->get(NewOpcode));
    MIB.add(PostIncBase).addImm(0).add(STValue);
  }

  // Transfer memoperands.
  MIB->cloneMemRefs(*MBB.getParent(), MI);

  // Create an add instruction for the post-inc addition of offset.
  MachineInstrBuilder MIBA = BuildMI(MBB, MI, DL, HII->get(Hexagon::A2_addi));
  MIBA.add(*PostIncDest).add(PostIncBase).add(PostIncOffset);

  LLVM_DEBUG({
    dbgs() << "\n";
    MI.dump();
    dbgs() << "\tis tranformed to \n";
    MIB->dump();
    MIBA->dump();
    dbgs() << "\n\n";
  });

  MI.eraseFromParent();
}

void HexagonPostIncOpt::generatePostInc(MachineBasicBlock &MBB) {
  LLVM_DEBUG(dbgs() << "# Generate Post-inc and update uses if needed.\n");
  MachineBasicBlock::iterator MII = MBB.getFirstNonPHI();
  MachineBasicBlock::iterator MIE = MBB.instr_begin();
  bool isOK = true;
  while (MII != MIE) {
    MachineInstr *Phi = &*std::prev(MII);
    MII = std::prev(MII);
    unsigned LoopVal = getLoopPhiReg(Phi, &MBB);
    if (LoopVal == UINT_MAX)
      continue;
    MachineInstr *LoopInst = MRI->getVRegDef(LoopVal);
    if (!isAddWithImmValue(*LoopInst))
      continue;

    if (LoopInst->getOpcode() != Hexagon::A2_addi)
      continue;

    unsigned AddReg = LoopInst->getOperand(1).getReg();
    int64_t AddImm = LoopInst->getOperand(2).getImm();
    SmallVector<MachineInstr *, 4> UseList;
    MachineInstr *PostIncCandidate = nullptr;

    // Find the probable candidates for Post-increment instruction.
    SmallVector<MachineInstr *, 4> CandList;
    for (auto &MO : make_range(MRI->use_begin(AddReg), MRI->use_end())) {
      MachineInstr *UseMI = MO.getParent();

      if (UseMI == LoopInst)
        continue;

      if (!dominates(UseMI, LoopInst)) {
        isOK = false;
        break;
      }
      const MachineOperand *BaseOp = nullptr;
      int64_t Offset;
      bool OffsetIsScalable;
      if (!HII->isBaseImmOffset(*UseMI) ||
          !HII->getMemOperandWithOffset(*UseMI, BaseOp, Offset,
                                        OffsetIsScalable, TRI)) {
        isOK = false;
        break;
      }
      int64_t NewOffset = Offset - AddImm;
      if (!isValidOffset(*UseMI, NewOffset) || !BaseOp->isReg() ||
          BaseOp->getReg() != AddReg) {
        isOK = false;
        break;
      }
      if (OffsetIsScalable) {
        isOK = false;
        break;
      }
      if (Offset == 0) {
        // If you have stores in the chain, make sure they are in the beginning
        // of the list. Eg: LD, LD, ST, ST will end up as LD, LD, PostInc_ST,
        // ST.
        if (UseMI->mayStore() && PreferPostIncStore)
          CandList.insert(CandList.begin(), UseMI);
        else
          CandList.push_back(UseMI);
        continue;
      }
      UseList.push_back(UseMI);
    }

    if (!isOK)
      continue;

    for (auto MI : CandList) {
      if (!PostIncCandidate)
        PostIncCandidate = MI;
      // Push the rest of the list for updation.
      else
        UseList.push_back(MI);
    }

    // If a candidate is found, replace it with the post-inc instruction.
    // Also, adjust offset for other uses as needed.
    if (!PostIncCandidate || !canReplaceWithPostInc(PostIncCandidate, LoopInst))
      continue;

    // Logic to determine what the base register to be.
    // There are two choices:
    //   1. New address register after we updated the post-increment candidate.
    //      v2,v3 = post_load v1, 4
    //      v3 is the choice here.
    //   2. The base register we used in post-increment candidate.
    //      v2,v3 = post_load v1, 4
    //      v1 is the choice here.
    // Use v3  if there is a memory dependence between post-inc instruction and
    // any other instruction in the chain.
    // FIXME: We can do some complex DAG analysis based off height and depth and
    // selectively update other instructions in the chain. Use v3 if there are
    // more instructions in the chain, otherwise we will end up increasing the
    // height of the DAG resulting in more spills. By default we have a
    // threshold controlled by the option "post-inc-chain-threshold" which is
    // set to 4. v1 is preferred as we can packetize two memory operations in a
    // single packet in scalar core. But it heavily depends on the structure of
    // DAG.
    bool UpdateBaseToNew = false;

    // Do not bother to build a DAG and analyze if the Use list is empty.
    if (!UseList.empty()) {
      MachineFunction *MF = MBB.getParent();
      // Setup the Post-inc schedule DAG.
      HexagonPostIncOptSchedDAG PIDAG(*this, *MF, MLI);
      initPISchedDAG(PIDAG, MBB);
      SUnit *SU = PIDAG.getSUnit(PostIncCandidate);
      if (hasMemoryDependency(SU, UseList) ||
          UseList.size() >= PostIncChainThreshold)
        UpdateBaseToNew = true;
    }

    if (UpdateBaseToNew) {
      LLVM_DEBUG(dbgs() << "The heuristic determines to update the uses of the "
                           "base register of post-increment\n");
      for (auto UseMI : UseList) {
        if (!dominates(PostIncCandidate, UseMI))
          continue;
        unsigned BasePos, OffsetPos;
        if (HII->getBaseAndOffsetPosition(*UseMI, BasePos, OffsetPos)) {
          // New offset has already been validated; no need to do it again.
          LLVM_DEBUG({
            UseMI->dump();
            dbgs() << "\t is transformed to \n";
          });
          int64_t NewOffset = UseMI->getOperand(OffsetPos).getImm() - AddImm;
          UseMI->getOperand(OffsetPos).setImm(NewOffset);
          UseMI->getOperand(BasePos).setReg(LoopVal);
          LLVM_DEBUG(UseMI->dump());
        }
      }
    }
    replaceWithPostInc(PostIncCandidate, LoopInst);
  }
  LLVM_DEBUG(dbgs() << "# End of generation of Post-inc.\n");
}

bool HexagonPostIncOpt::canReplaceWithPostInc(MachineInstr *MI,
                                              MachineInstr *AddMI) const {
  if (HII->changeAddrMode_io_pi(MI->getOpcode()) < 0)
    return false;
  assert(AddMI->getOpcode() == Hexagon::A2_addi);
  return isValidPostIncValue(*MI, AddMI->getOperand(2).getImm());
}

void HexagonPostIncOpt::replaceWithPostInc(MachineInstr *MI,
                                           MachineInstr *AddMI) const {
  short NewOpcode = HII->changeAddrMode_io_pi(MI->getOpcode());
  assert(NewOpcode >= 0 &&
         "Couldn't change base offset to post-increment form");

  MachineBasicBlock &MBB = *MI->getParent();
  DebugLoc DL = MI->getDebugLoc();
  const MachineOperand &IncDest = AddMI->getOperand(0);
  const MachineOperand &IncBase = AddMI->getOperand(1);
  const MachineOperand &IncValue = AddMI->getOperand(2);
  MachineInstrBuilder MIB;
  LLVM_DEBUG({
    dbgs() << "\n\n";
    MI->dump();
    dbgs() << "\t is tranformed to post-inc form of \n";
  });

  if (MI->mayLoad()) {
    const MachineOperand &LDValue = MI->getOperand(0);
    MIB = BuildMI(MBB, *MI, DL, HII->get(NewOpcode));
    MIB.add(LDValue).add(IncDest).add(IncBase).add(IncValue);
  } else {
    const MachineOperand &STValue = MI->getOperand(2);
    MIB = BuildMI(MBB, *MI, DL, HII->get(NewOpcode));
    MIB.add(IncDest).add(IncBase).add(IncValue).add(STValue);
  }

  // Transfer memoperands.
  MIB->cloneMemRefs(*MBB.getParent(), *MI);

  LLVM_DEBUG({
    MIB->dump();
    dbgs() << "As a result this add instruction is erased.\n";
    AddMI->dump();
  });

  MI->eraseFromParent();
  AddMI->eraseFromParent();
}

bool HexagonPostIncOpt::translatePostIncsInLoop(MachineBasicBlock &MBB) {
  // Algorithm:
  // 1. Replace all the post-inc instructions with Base+Offset instruction and
  // an add instruction in this block.
  // 2. Fold all the adds in to respective uses.
  // 3. Generate post-increment instructions and update the uses of the base
  // register if needed based on constraints.

  replacePostIncWithBaseOffset(MBB);
  foldAdds(MBB);
  generatePostInc(MBB);
  return true;
}

bool HexagonPostIncOpt::runOnMachineFunction(MachineFunction &MF) {

  // Skip pass if requested.
  if (skipFunction(MF.getFunction()))
    return false;

  // Get Target Information.
  MLI = &getAnalysis<MachineLoopInfo>();
  HST = &MF.getSubtarget<HexagonSubtarget>();
  TRI = HST->getRegisterInfo();
  MRI = &MF.getRegInfo();
  HII = HST->getInstrInfo();

  // Skip this pass for TinyCore.
  // Tiny core allwos partial post increment operations - This constraint can
  // be imposed inside the pass. In a chain of post-increments, the first can
  // be post-increment, rest can be adjusted to base+offset (these are
  // inexpensive in most of the cases);
  if (HST->isTinyCore())
    return false;

  LLVM_DEBUG({
    dbgs() << "Begin: Hexagon Post-Inc-Opt Pass.\n";
    dbgs() << "Function: " << MF.getName() << "\n";
  });
  bool Change = false;
  std::vector<MachineBasicBlock *> MLBB;
  for (auto &BB : MF) {
    // Check if this Basic Block belongs to any loop.
    auto *LI = MLI->getLoopFor(&BB);
    // We only deal with inner-most loops that has one block.
    if (LI && LI->getBlocks().size() == 1) {
      MachineBasicBlock *MBB = LI->getHeader();
      // Do not traverse blocks that are already visited.
      if (std::find(MLBB.begin(), MLBB.end(), MBB) != MLBB.end())
        continue;

      MLBB.push_back(MBB);

      LLVM_DEBUG(dbgs() << "\n\t Basic Block: " << MBB->getName() << "\n");
      Change |= translatePostIncsInLoop(*MBB);
    }
  }
  LLVM_DEBUG(dbgs() << "End: Hexagon Post-Inc-Opt Pass\n");
  return Change;
}

FunctionPass *llvm::createHexagonPostIncOpt() {
  return new HexagonPostIncOpt();
}
