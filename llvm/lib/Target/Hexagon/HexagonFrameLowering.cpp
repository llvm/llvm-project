//===-- HexagonFrameLowering.cpp - Define frame lowering ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "hexagon-pei"

#include "HexagonFrameLowering.h"
#include "Hexagon.h"
#include "HexagonInstrInfo.h"
#include "HexagonMachineFunctionInfo.h"
#include "HexagonRegisterInfo.h"
#include "HexagonSubtarget.h"
#include "HexagonTargetMachine.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

// Hexagon stack frame layout as defined by the ABI:
//
//                                                       Incoming arguments
//                                                       passed via stack
//                                                                      |
//                                                                      |
//        SP during function's                 FP during function's     |
//    +-- runtime (top of stack)               runtime (bottom) --+     |
//    |                                                           |     |
// --++---------------------+------------------+-----------------++-+-------
//   |  parameter area for  |  variable-size   |   fixed-size    |LR|  arg
//   |   called functions   |  local objects   |  local objects  |FP|
// --+----------------------+------------------+-----------------+--+-------
//    <-    size known    -> <- size unknown -> <- size known  ->
//
// Low address                                                 High address
//
// <--- stack growth
//
//
// - In any circumstances, the outgoing function arguments are always accessi-
//   ble using the SP, and the incoming arguments are accessible using the FP.
// - If the local objects are not aligned, they can always be accessed using
//   the FP.
// - If there are no variable-sized objects, the local objects can always be
//   accessed using the SP, regardless whether they are aligned or not. (The
//   alignment padding will be at the bottom of the stack (highest address),
//   and so the offset with respect to the SP will be known at the compile-
//   -time.)
//
// The only complication occurs if there are both, local aligned objects, and
// dynamically allocated (variable-sized) objects. The alignment pad will be
// placed between the FP and the local objects, thus preventing the use of the
// FP to access the local objects. At the same time, the variable-sized objects
// will be between the SP and the local objects, thus introducing an unknown
// distance from the SP to the locals.
//
// To avoid this problem, a new register is created that holds the aligned
// address of the bottom of the stack, referred in the sources as AP (aligned
// pointer). The AP will be equal to "FP-p", where "p" is the smallest pad
// that aligns AP to the required boundary (a maximum of the alignments of
// all stack objects, fixed- and variable-sized). All local objects[1] will
// then use AP as the base pointer.
// [1] The exception is with "fixed" stack objects. "Fixed" stack objects get
// their name from being allocated at fixed locations on the stack, relative
// to the FP. In the presence of dynamic allocation and local alignment, such
// objects can only be accessed through the FP.
//
// Illustration of the AP:
//                                                                FP --+
//                                                                     |
// ---------------+---------------------+-----+-----------------------++-+--
//   Rest of the  | Local stack objects | Pad |  Fixed stack objects  |LR|
//   stack frame  | (aligned)           |     |  (CSR, spills, etc.)  |FP|
// ---------------+---------------------+-----+-----------------+-----+--+--
//                                      |<-- Multiple of the -->|
//                                           stack alignment    +-- AP
//
// The AP is set up at the beginning of the function. Since it is not a dedi-
// cated (reserved) register, it needs to be kept live throughout the function
// to be available as the base register for local object accesses.
// Normally, an address of a stack objects is obtained by a pseudo-instruction
// TFR_FI. To access local objects with the AP register present, a different
// pseudo-instruction needs to be used: TFR_FIA. The TFR_FIA takes one extra
// argument compared to TFR_FI: the first input register is the AP register.
// This keeps the register live between its definition and its uses.

// The AP register is originally set up using pseudo-instruction ALIGNA:
//   AP = ALIGNA A
// where
//   A  - required stack alignment
// The alignment value must be the maximum of all alignments required by
// any stack object.

// The dynamic allocation uses a pseudo-instruction ALLOCA:
//   Rd = ALLOCA Rs, A
// where
//   Rd - address of the allocated space
//   Rs - minimum size (the actual allocated can be larger to accommodate
//        alignment)
//   A  - required alignment


using namespace llvm;

static cl::opt<bool> DisableDeallocRet("disable-hexagon-dealloc-ret",
    cl::Hidden, cl::desc("Disable Dealloc Return for Hexagon target"));


static cl::opt<int> NumberScavengerSlots("number-scavenger-slots",
    cl::Hidden, cl::desc("Set the number of scavenger slots"), cl::init(2),
    cl::ZeroOrMore);

static cl::opt<int> SpillFuncThreshold("spill-func-threshold",
    cl::Hidden, cl::desc("Specify O2(not Os) spill func threshold"),
    cl::init(6), cl::ZeroOrMore);

static cl::opt<int> SpillFuncThresholdOs("spill-func-threshold-Os",
    cl::Hidden, cl::desc("Specify Os spill func threshold"),
    cl::init(1), cl::ZeroOrMore);

static cl::opt<bool> EnableShrinkWrapping("hexagon-shrink-frame",
    cl::init(true), cl::Hidden, cl::ZeroOrMore,
    cl::desc("Enable stack frame shrink wrapping"));

static cl::opt<unsigned> ShrinkLimit("shrink-frame-limit", cl::init(UINT_MAX),
    cl::Hidden, cl::ZeroOrMore, cl::desc("Max count of stack frame "
    "shrink-wraps"));

static cl::opt<bool> UseAllocframe("use-allocframe", cl::init(true),
    cl::Hidden, cl::desc("Use allocframe more conservatively"));


namespace llvm {
  void initializeHexagonCallFrameInformationPass(PassRegistry&);
  FunctionPass *createHexagonCallFrameInformation();
}

namespace {
  class HexagonCallFrameInformation : public MachineFunctionPass {
  public:
    static char ID;
    HexagonCallFrameInformation() : MachineFunctionPass(ID) {
      PassRegistry &PR = *PassRegistry::getPassRegistry();
      initializeHexagonCallFrameInformationPass(PR);
    }
    bool runOnMachineFunction(MachineFunction &MF) override;
  };

  char HexagonCallFrameInformation::ID = 0;
}

bool HexagonCallFrameInformation::runOnMachineFunction(MachineFunction &MF) {
  auto &HFI = *MF.getSubtarget<HexagonSubtarget>().getFrameLowering();
  bool NeedCFI = MF.getMMI().hasDebugInfo() ||
                 MF.getFunction()->needsUnwindTableEntry();

  if (!NeedCFI)
    return false;
  HFI.insertCFIInstructions(MF);
  return true;
}

INITIALIZE_PASS(HexagonCallFrameInformation, "hexagon-cfi",
                "Hexagon call frame information", false, false)

FunctionPass *llvm::createHexagonCallFrameInformation() {
  return new HexagonCallFrameInformation();
}


namespace {
  /// Map a register pair Reg to the subregister that has the greater "number",
  /// i.e. D3 (aka R7:6) will be mapped to R7, etc.
  unsigned getMax32BitSubRegister(unsigned Reg, const TargetRegisterInfo &TRI,
                                  bool hireg = true) {
    if (Reg < Hexagon::D0 || Reg > Hexagon::D15)
      return Reg;

    unsigned RegNo = 0;
    for (MCSubRegIterator SubRegs(Reg, &TRI); SubRegs.isValid(); ++SubRegs) {
      if (hireg) {
        if (*SubRegs > RegNo)
          RegNo = *SubRegs;
      } else {
        if (!RegNo || *SubRegs < RegNo)
          RegNo = *SubRegs;
      }
    }
    return RegNo;
  }

  /// Returns the callee saved register with the largest id in the vector.
  unsigned getMaxCalleeSavedReg(const std::vector<CalleeSavedInfo> &CSI,
                                const TargetRegisterInfo &TRI) {
    assert(Hexagon::R1 > 0 &&
           "Assume physical registers are encoded as positive integers");
    if (CSI.empty())
      return 0;

    unsigned Max = getMax32BitSubRegister(CSI[0].getReg(), TRI);
    for (unsigned I = 1, E = CSI.size(); I < E; ++I) {
      unsigned Reg = getMax32BitSubRegister(CSI[I].getReg(), TRI);
      if (Reg > Max)
        Max = Reg;
    }
    return Max;
  }

  /// Checks if the basic block contains any instruction that needs a stack
  /// frame to be already in place.
  bool needsStackFrame(const MachineBasicBlock &MBB, const BitVector &CSR) {
    for (auto &I : MBB) {
      const MachineInstr *MI = &I;
      if (MI->isCall())
        return true;
      unsigned Opc = MI->getOpcode();
      switch (Opc) {
        case Hexagon::ALLOCA:
        case Hexagon::ALIGNA:
          return true;
        default:
          break;
      }
      // Check individual operands.
      for (const MachineOperand &MO : MI->operands()) {
        // While the presence of a frame index does not prove that a stack
        // frame will be required, all frame indexes should be within alloc-
        // frame/deallocframe. Otherwise, the code that translates a frame
        // index into an offset would have to be aware of the placement of
        // the frame creation/destruction instructions.
        if (MO.isFI())
          return true;
        if (!MO.isReg())
          continue;
        unsigned R = MO.getReg();
        // Virtual registers will need scavenging, which then may require
        // a stack slot.
        if (TargetRegisterInfo::isVirtualRegister(R))
          return true;
        if (CSR[R])
          return true;
      }
    }
    return false;
  }

  /// Returns true if MBB has a machine instructions that indicates a tail call
  /// in the block.
  bool hasTailCall(const MachineBasicBlock &MBB) {
    MachineBasicBlock::const_iterator I = MBB.getLastNonDebugInstr();
    unsigned RetOpc = I->getOpcode();
    return RetOpc == Hexagon::TCRETURNi || RetOpc == Hexagon::TCRETURNr;
  }

  /// Returns true if MBB contains an instruction that returns.
  bool hasReturn(const MachineBasicBlock &MBB) {
    for (auto I = MBB.getFirstTerminator(), E = MBB.end(); I != E; ++I)
      if (I->isReturn())
        return true;
    return false;
  }
}


/// Implements shrink-wrapping of the stack frame. By default, stack frame
/// is created in the function entry block, and is cleaned up in every block
/// that returns. This function finds alternate blocks: one for the frame
/// setup (prolog) and one for the cleanup (epilog).
void HexagonFrameLowering::findShrunkPrologEpilog(MachineFunction &MF,
      MachineBasicBlock *&PrologB, MachineBasicBlock *&EpilogB) const {
  static unsigned ShrinkCounter = 0;

  if (ShrinkLimit.getPosition()) {
    if (ShrinkCounter >= ShrinkLimit)
      return;
    ShrinkCounter++;
  }

  auto &HST = static_cast<const HexagonSubtarget&>(MF.getSubtarget());
  auto &HRI = *HST.getRegisterInfo();

  MachineDominatorTree MDT;
  MDT.runOnMachineFunction(MF);
  MachinePostDominatorTree MPT;
  MPT.runOnMachineFunction(MF);

  typedef DenseMap<unsigned,unsigned> UnsignedMap;
  UnsignedMap RPO;
  typedef ReversePostOrderTraversal<const MachineFunction*> RPOTType;
  RPOTType RPOT(&MF);
  unsigned RPON = 0;
  for (RPOTType::rpo_iterator I = RPOT.begin(), E = RPOT.end(); I != E; ++I)
    RPO[(*I)->getNumber()] = RPON++;

  // Don't process functions that have loops, at least for now. Placement
  // of prolog and epilog must take loop structure into account. For simpli-
  // city don't do it right now.
  for (auto &I : MF) {
    unsigned BN = RPO[I.getNumber()];
    for (auto SI = I.succ_begin(), SE = I.succ_end(); SI != SE; ++SI) {
      // If found a back-edge, return.
      if (RPO[(*SI)->getNumber()] <= BN)
        return;
    }
  }

  // Collect the set of blocks that need a stack frame to execute. Scan
  // each block for uses/defs of callee-saved registers, calls, etc.
  SmallVector<MachineBasicBlock*,16> SFBlocks;
  BitVector CSR(Hexagon::NUM_TARGET_REGS);
  for (const MCPhysReg *P = HRI.getCalleeSavedRegs(&MF); *P; ++P)
    CSR[*P] = true;

  for (auto &I : MF)
    if (needsStackFrame(I, CSR))
      SFBlocks.push_back(&I);

  DEBUG({
    dbgs() << "Blocks needing SF: {";
    for (auto &B : SFBlocks)
      dbgs() << " BB#" << B->getNumber();
    dbgs() << " }\n";
  });
  // No frame needed?
  if (SFBlocks.empty())
    return;

  // Pick a common dominator and a common post-dominator.
  MachineBasicBlock *DomB = SFBlocks[0];
  for (unsigned i = 1, n = SFBlocks.size(); i < n; ++i) {
    DomB = MDT.findNearestCommonDominator(DomB, SFBlocks[i]);
    if (!DomB)
      break;
  }
  MachineBasicBlock *PDomB = SFBlocks[0];
  for (unsigned i = 1, n = SFBlocks.size(); i < n; ++i) {
    PDomB = MPT.findNearestCommonDominator(PDomB, SFBlocks[i]);
    if (!PDomB)
      break;
  }
  DEBUG({
    dbgs() << "Computed dom block: BB#";
    if (DomB) dbgs() << DomB->getNumber();
    else      dbgs() << "<null>";
    dbgs() << ", computed pdom block: BB#";
    if (PDomB) dbgs() << PDomB->getNumber();
    else       dbgs() << "<null>";
    dbgs() << "\n";
  });
  if (!DomB || !PDomB)
    return;

  // Make sure that DomB dominates PDomB and PDomB post-dominates DomB.
  if (!MDT.dominates(DomB, PDomB)) {
    DEBUG(dbgs() << "Dom block does not dominate pdom block\n");
    return;
  }
  if (!MPT.dominates(PDomB, DomB)) {
    DEBUG(dbgs() << "PDom block does not post-dominate dom block\n");
    return;
  }

  // Finally, everything seems right.
  PrologB = DomB;
  EpilogB = PDomB;
}

/// Perform most of the PEI work here:
/// - saving/restoring of the callee-saved registers,
/// - stack frame creation and destruction.
/// Normally, this work is distributed among various functions, but doing it
/// in one place allows shrink-wrapping of the stack frame.
void HexagonFrameLowering::emitPrologue(MachineFunction &MF,
                                        MachineBasicBlock &MBB) const {
  auto &HST = static_cast<const HexagonSubtarget&>(MF.getSubtarget());
  auto &HRI = *HST.getRegisterInfo();

  assert(&MF.front() == &MBB && "Shrink-wrapping not yet supported");
  MachineFrameInfo *MFI = MF.getFrameInfo();
  const std::vector<CalleeSavedInfo> &CSI = MFI->getCalleeSavedInfo();

  MachineBasicBlock *PrologB = &MF.front(), *EpilogB = nullptr;
  if (EnableShrinkWrapping)
    findShrunkPrologEpilog(MF, PrologB, EpilogB);

  insertCSRSpillsInBlock(*PrologB, CSI, HRI);
  insertPrologueInBlock(*PrologB);

  if (EpilogB) {
    insertCSRRestoresInBlock(*EpilogB, CSI, HRI);
    insertEpilogueInBlock(*EpilogB);
  } else {
    for (auto &B : MF)
      if (B.isReturnBlock())
        insertCSRRestoresInBlock(B, CSI, HRI);

    for (auto &B : MF)
      if (B.isReturnBlock())
        insertEpilogueInBlock(B);
  }
}


void HexagonFrameLowering::insertPrologueInBlock(MachineBasicBlock &MBB) const {
  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  auto &HST = MF.getSubtarget<HexagonSubtarget>();
  auto &HII = *HST.getInstrInfo();
  auto &HRI = *HST.getRegisterInfo();
  DebugLoc dl;

  unsigned MaxAlign = std::max(MFI->getMaxAlignment(), getStackAlignment());

  // Calculate the total stack frame size.
  // Get the number of bytes to allocate from the FrameInfo.
  unsigned FrameSize = MFI->getStackSize();
  // Round up the max call frame size to the max alignment on the stack.
  unsigned MaxCFA = RoundUpToAlignment(MFI->getMaxCallFrameSize(), MaxAlign);
  MFI->setMaxCallFrameSize(MaxCFA);

  FrameSize = MaxCFA + RoundUpToAlignment(FrameSize, MaxAlign);
  MFI->setStackSize(FrameSize);

  bool AlignStack = (MaxAlign > getStackAlignment());

  // Get the number of bytes to allocate from the FrameInfo.
  unsigned NumBytes = MFI->getStackSize();
  unsigned SP = HRI.getStackRegister();
  unsigned MaxCF = MFI->getMaxCallFrameSize();
  MachineBasicBlock::iterator InsertPt = MBB.begin();

  auto *FuncInfo = MF.getInfo<HexagonMachineFunctionInfo>();
  auto &AdjustRegs = FuncInfo->getAllocaAdjustInsts();

  for (auto MI : AdjustRegs) {
    assert((MI->getOpcode() == Hexagon::ALLOCA) && "Expected alloca");
    expandAlloca(MI, HII, SP, MaxCF);
    MI->eraseFromParent();
  }

  if (!hasFP(MF))
    return;

  // Check for overflow.
  // Hexagon_TODO: Ugh! hardcoding. Is there an API that can be used?
  const unsigned int ALLOCFRAME_MAX = 16384;

  // Create a dummy memory operand to avoid allocframe from being treated as
  // a volatile memory reference.
  MachineMemOperand *MMO =
    MF.getMachineMemOperand(MachinePointerInfo(), MachineMemOperand::MOStore,
                            4, 4);

  if (NumBytes >= ALLOCFRAME_MAX) {
    // Emit allocframe(#0).
    BuildMI(MBB, InsertPt, dl, HII.get(Hexagon::S2_allocframe))
      .addImm(0)
      .addMemOperand(MMO);

    // Subtract offset from frame pointer.
    // We use a caller-saved non-parameter register for that.
    unsigned CallerSavedReg = HRI.getFirstCallerSavedNonParamReg();
    BuildMI(MBB, InsertPt, dl, HII.get(Hexagon::CONST32_Int_Real),
            CallerSavedReg).addImm(NumBytes);
    BuildMI(MBB, InsertPt, dl, HII.get(Hexagon::A2_sub), SP)
      .addReg(SP)
      .addReg(CallerSavedReg);
  } else {
    BuildMI(MBB, InsertPt, dl, HII.get(Hexagon::S2_allocframe))
      .addImm(NumBytes)
      .addMemOperand(MMO);
  }

  if (AlignStack) {
    BuildMI(MBB, InsertPt, dl, HII.get(Hexagon::A2_andir), SP)
        .addReg(SP)
        .addImm(-int64_t(MaxAlign));
  }
}

void HexagonFrameLowering::insertEpilogueInBlock(MachineBasicBlock &MBB) const {
  MachineFunction &MF = *MBB.getParent();
  if (!hasFP(MF))
    return;

  auto &HST = static_cast<const HexagonSubtarget&>(MF.getSubtarget());
  auto &HII = *HST.getInstrInfo();
  auto &HRI = *HST.getRegisterInfo();
  unsigned SP = HRI.getStackRegister();

  MachineInstr *RetI = nullptr;
  for (auto &I : MBB) {
    if (!I.isReturn())
      continue;
    RetI = &I;
    break;
  }
  unsigned RetOpc = RetI ? RetI->getOpcode() : 0;

  MachineBasicBlock::iterator InsertPt = MBB.getFirstTerminator();
  DebugLoc DL;
  if (InsertPt != MBB.end())
    DL = InsertPt->getDebugLoc();
  else if (!MBB.empty())
    DL = std::prev(MBB.end())->getDebugLoc();

  // Handle EH_RETURN.
  if (RetOpc == Hexagon::EH_RETURN_JMPR) {
    BuildMI(MBB, InsertPt, DL, HII.get(Hexagon::L2_deallocframe));
    BuildMI(MBB, InsertPt, DL, HII.get(Hexagon::A2_add), SP)
        .addReg(SP)
        .addReg(Hexagon::R28);
    return;
  }

  // Check for RESTORE_DEALLOC_RET* tail call. Don't emit an extra dealloc-
  // frame instruction if we encounter it.
  if (RetOpc == Hexagon::RESTORE_DEALLOC_RET_JMP_V4) {
    MachineBasicBlock::iterator It = RetI;
    ++It;
    // Delete all instructions after the RESTORE (except labels).
    while (It != MBB.end()) {
      if (!It->isLabel())
        It = MBB.erase(It);
      else
        ++It;
    }
    return;
  }

  // It is possible that the restoring code is a call to a library function.
  // All of the restore* functions include "deallocframe", so we need to make
  // sure that we don't add an extra one.
  bool NeedsDeallocframe = true;
  if (!MBB.empty() && InsertPt != MBB.begin()) {
    MachineBasicBlock::iterator PrevIt = std::prev(InsertPt);
    unsigned COpc = PrevIt->getOpcode();
    if (COpc == Hexagon::RESTORE_DEALLOC_BEFORE_TAILCALL_V4)
      NeedsDeallocframe = false;
  }

  if (!NeedsDeallocframe)
    return;
  // If the returning instruction is JMPret, replace it with dealloc_return,
  // otherwise just add deallocframe. The function could be returning via a
  // tail call.
  if (RetOpc != Hexagon::JMPret || DisableDeallocRet) {
    BuildMI(MBB, InsertPt, DL, HII.get(Hexagon::L2_deallocframe));
    return;
  }
  unsigned NewOpc = Hexagon::L4_return;
  MachineInstr *NewI = BuildMI(MBB, RetI, DL, HII.get(NewOpc));
  // Transfer the function live-out registers.
  NewI->copyImplicitOps(MF, RetI);
  MBB.erase(RetI);
}


namespace {
  bool IsAllocFrame(MachineBasicBlock::const_iterator It) {
    if (!It->isBundle())
      return It->getOpcode() == Hexagon::S2_allocframe;
    auto End = It->getParent()->instr_end();
    MachineBasicBlock::const_instr_iterator I = It.getInstrIterator();
    while (++I != End && I->isBundled())
      if (I->getOpcode() == Hexagon::S2_allocframe)
        return true;
    return false;
  }

  MachineBasicBlock::iterator FindAllocFrame(MachineBasicBlock &B) {
    for (auto &I : B)
      if (IsAllocFrame(I))
        return I;
    return B.end();
  }
}


void HexagonFrameLowering::insertCFIInstructions(MachineFunction &MF) const {
  for (auto &B : MF) {
    auto AF = FindAllocFrame(B);
    if (AF == B.end())
      continue;
    insertCFIInstructionsAt(B, ++AF);
  }
}


void HexagonFrameLowering::insertCFIInstructionsAt(MachineBasicBlock &MBB,
      MachineBasicBlock::iterator At) const {
  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineModuleInfo &MMI = MF.getMMI();
  auto &HST = MF.getSubtarget<HexagonSubtarget>();
  auto &HII = *HST.getInstrInfo();
  auto &HRI = *HST.getRegisterInfo();

  // If CFI instructions have debug information attached, something goes
  // wrong with the final assembly generation: the prolog_end is placed
  // in a wrong location.
  DebugLoc DL;
  const MCInstrDesc &CFID = HII.get(TargetOpcode::CFI_INSTRUCTION);

  MCSymbol *FrameLabel = MMI.getContext().createTempSymbol();

  if (hasFP(MF)) {
    unsigned DwFPReg = HRI.getDwarfRegNum(HRI.getFrameRegister(), true);
    unsigned DwRAReg = HRI.getDwarfRegNum(HRI.getRARegister(), true);

    // Define CFA via an offset from the value of FP.
    //
    //  -8   -4    0 (SP)
    // --+----+----+---------------------
    //   | FP | LR |          increasing addresses -->
    // --+----+----+---------------------
    //   |         +-- Old SP (before allocframe)
    //   +-- New FP (after allocframe)
    //
    // MCCFIInstruction::createDefCfa subtracts the offset from the register.
    // MCCFIInstruction::createOffset takes the offset without sign change.
    auto DefCfa = MCCFIInstruction::createDefCfa(FrameLabel, DwFPReg, -8);
    BuildMI(MBB, At, DL, CFID)
        .addCFIIndex(MMI.addFrameInst(DefCfa));
    // R31 (return addr) = CFA - 4
    auto OffR31 = MCCFIInstruction::createOffset(FrameLabel, DwRAReg, -4);
    BuildMI(MBB, At, DL, CFID)
        .addCFIIndex(MMI.addFrameInst(OffR31));
    // R30 (frame ptr) = CFA - 8
    auto OffR30 = MCCFIInstruction::createOffset(FrameLabel, DwFPReg, -8);
    BuildMI(MBB, At, DL, CFID)
        .addCFIIndex(MMI.addFrameInst(OffR30));
  }

  static unsigned int RegsToMove[] = {
    Hexagon::R1,  Hexagon::R0,  Hexagon::R3,  Hexagon::R2,
    Hexagon::R17, Hexagon::R16, Hexagon::R19, Hexagon::R18,
    Hexagon::R21, Hexagon::R20, Hexagon::R23, Hexagon::R22,
    Hexagon::R25, Hexagon::R24, Hexagon::R27, Hexagon::R26,
    Hexagon::D0,  Hexagon::D1,  Hexagon::D8,  Hexagon::D9,
    Hexagon::D10, Hexagon::D11, Hexagon::D12, Hexagon::D13,
    Hexagon::NoRegister
  };

  const std::vector<CalleeSavedInfo> &CSI = MFI->getCalleeSavedInfo();

  for (unsigned i = 0; RegsToMove[i] != Hexagon::NoRegister; ++i) {
    unsigned Reg = RegsToMove[i];
    auto IfR = [Reg] (const CalleeSavedInfo &C) -> bool {
      return C.getReg() == Reg;
    };
    auto F = std::find_if(CSI.begin(), CSI.end(), IfR);
    if (F == CSI.end())
      continue;

    // Subtract 8 to make room for R30 and R31, which are added above.
    unsigned FrameReg;
    int64_t Offset = getFrameIndexReference(MF, F->getFrameIdx(), FrameReg) - 8;

    if (Reg < Hexagon::D0 || Reg > Hexagon::D15) {
      unsigned DwarfReg = HRI.getDwarfRegNum(Reg, true);
      auto OffReg = MCCFIInstruction::createOffset(FrameLabel, DwarfReg,
                                                   Offset);
      BuildMI(MBB, At, DL, CFID)
          .addCFIIndex(MMI.addFrameInst(OffReg));
    } else {
      // Split the double regs into subregs, and generate appropriate
      // cfi_offsets.
      // The only reason, we are split double regs is, llvm-mc does not
      // understand paired registers for cfi_offset.
      // Eg .cfi_offset r1:0, -64

      unsigned HiReg = HRI.getSubReg(Reg, Hexagon::subreg_hireg);
      unsigned LoReg = HRI.getSubReg(Reg, Hexagon::subreg_loreg);
      unsigned HiDwarfReg = HRI.getDwarfRegNum(HiReg, true);
      unsigned LoDwarfReg = HRI.getDwarfRegNum(LoReg, true);
      auto OffHi = MCCFIInstruction::createOffset(FrameLabel, HiDwarfReg,
                                                  Offset+4);
      BuildMI(MBB, At, DL, CFID)
          .addCFIIndex(MMI.addFrameInst(OffHi));
      auto OffLo = MCCFIInstruction::createOffset(FrameLabel, LoDwarfReg,
                                                  Offset);
      BuildMI(MBB, At, DL, CFID)
          .addCFIIndex(MMI.addFrameInst(OffLo));
    }
  }
}


bool HexagonFrameLowering::hasFP(const MachineFunction &MF) const {
  auto &MFI = *MF.getFrameInfo();
  auto &HRI = *MF.getSubtarget<HexagonSubtarget>().getRegisterInfo();

  bool HasFixed = MFI.getNumFixedObjects();
  bool HasPrealloc = const_cast<MachineFrameInfo&>(MFI)
                        .getLocalFrameObjectCount();
  bool HasExtraAlign = HRI.needsStackRealignment(MF);
  bool HasAlloca = MFI.hasVarSizedObjects();

  // Insert ALLOCFRAME if we need to or at -O0 for the debugger.  Think
  // that this shouldn't be required, but doing so now because gcc does and
  // gdb can't break at the start of the function without it.  Will remove if
  // this turns out to be a gdb bug.
  //
  if (MF.getTarget().getOptLevel() == CodeGenOpt::None)
    return true;

  // By default we want to use SP (since it's always there). FP requires
  // some setup (i.e. ALLOCFRAME).
  // Fixed and preallocated objects need FP if the distance from them to
  // the SP is unknown (as is with alloca or aligna).
  if ((HasFixed || HasPrealloc) && (HasAlloca || HasExtraAlign))
    return true;

  if (MFI.getStackSize() > 0) {
    if (UseAllocframe)
      return true;
  }

  if (MFI.hasCalls() ||
      MF.getInfo<HexagonMachineFunctionInfo>()->hasClobberLR())
    return true;

  return false;
}


enum SpillKind {
  SK_ToMem,
  SK_FromMem,
  SK_FromMemTailcall
};

static const char *
getSpillFunctionFor(unsigned MaxReg, SpillKind SpillType) {
  const char * V4SpillToMemoryFunctions[] = {
    "__save_r16_through_r17",
    "__save_r16_through_r19",
    "__save_r16_through_r21",
    "__save_r16_through_r23",
    "__save_r16_through_r25",
    "__save_r16_through_r27" };

  const char * V4SpillFromMemoryFunctions[] = {
    "__restore_r16_through_r17_and_deallocframe",
    "__restore_r16_through_r19_and_deallocframe",
    "__restore_r16_through_r21_and_deallocframe",
    "__restore_r16_through_r23_and_deallocframe",
    "__restore_r16_through_r25_and_deallocframe",
    "__restore_r16_through_r27_and_deallocframe" };

  const char * V4SpillFromMemoryTailcallFunctions[] = {
    "__restore_r16_through_r17_and_deallocframe_before_tailcall",
    "__restore_r16_through_r19_and_deallocframe_before_tailcall",
    "__restore_r16_through_r21_and_deallocframe_before_tailcall",
    "__restore_r16_through_r23_and_deallocframe_before_tailcall",
    "__restore_r16_through_r25_and_deallocframe_before_tailcall",
    "__restore_r16_through_r27_and_deallocframe_before_tailcall"
  };

  const char **SpillFunc = nullptr;

  switch(SpillType) {
  case SK_ToMem:
    SpillFunc = V4SpillToMemoryFunctions;
    break;
  case SK_FromMem:
    SpillFunc = V4SpillFromMemoryFunctions;
    break;
  case SK_FromMemTailcall:
    SpillFunc = V4SpillFromMemoryTailcallFunctions;
    break;
  }
  assert(SpillFunc && "Unknown spill kind");

  // Spill all callee-saved registers up to the highest register used.
  switch (MaxReg) {
  case Hexagon::R17:
    return SpillFunc[0];
  case Hexagon::R19:
    return SpillFunc[1];
  case Hexagon::R21:
    return SpillFunc[2];
  case Hexagon::R23:
    return SpillFunc[3];
  case Hexagon::R25:
    return SpillFunc[4];
  case Hexagon::R27:
    return SpillFunc[5];
  default:
    llvm_unreachable("Unhandled maximum callee save register");
  }
  return 0;
}

/// Adds all callee-saved registers up to MaxReg to the instruction.
static void addCalleeSaveRegistersAsImpOperand(MachineInstr *Inst,
                                           unsigned MaxReg, bool IsDef) {
  // Add the callee-saved registers as implicit uses.
  for (unsigned R = Hexagon::R16; R <= MaxReg; ++R) {
    MachineOperand ImpUse = MachineOperand::CreateReg(R, IsDef, true);
    Inst->addOperand(ImpUse);
  }
}


int HexagonFrameLowering::getFrameIndexReference(const MachineFunction &MF,
      int FI, unsigned &FrameReg) const {
  auto &MFI = *MF.getFrameInfo();
  auto &HRI = *MF.getSubtarget<HexagonSubtarget>().getRegisterInfo();

  // Large parts of this code are shared with HRI::eliminateFrameIndex.
  int Offset = MFI.getObjectOffset(FI);
  bool HasAlloca = MFI.hasVarSizedObjects();
  bool HasExtraAlign = HRI.needsStackRealignment(MF);
  bool NoOpt = MF.getTarget().getOptLevel() == CodeGenOpt::None;

  unsigned SP = HRI.getStackRegister(), FP = HRI.getFrameRegister();
  unsigned AP = 0;
  if (const MachineInstr *AI = getAlignaInstr(MF))
    AP = AI->getOperand(0).getReg();
  unsigned FrameSize = MFI.getStackSize();

  bool UseFP = false, UseAP = false;  // Default: use SP (except at -O0).
  // Use FP at -O0, except when there are objects with extra alignment.
  // That additional alignment requirement may cause a pad to be inserted,
  // which will make it impossible to use FP to access objects located
  // past the pad.
  if (NoOpt && !HasExtraAlign)
    UseFP = true;
  if (MFI.isFixedObjectIndex(FI) || MFI.isObjectPreAllocated(FI)) {
    // Fixed and preallocated objects will be located before any padding
    // so FP must be used to access them.
    UseFP |= (HasAlloca || HasExtraAlign);
  } else {
    if (HasAlloca) {
      if (HasExtraAlign)
        UseAP = true;
      else
        UseFP = true;
    }
  }

  // If FP was picked, then there had better be FP.
  bool HasFP = hasFP(MF);
  assert((HasFP || !UseFP) && "This function must have frame pointer");

  // Having FP implies allocframe. Allocframe will store extra 8 bytes:
  // FP/LR. If the base register is used to access an object across these
  // 8 bytes, then the offset will need to be adjusted by 8.
  //
  // After allocframe:
  //                    HexagonISelLowering adds 8 to ---+
  //                    the offsets of all stack-based   |
  //                    arguments (*)                    |
  //                                                     |
  //   getObjectOffset < 0   0     8  getObjectOffset >= 8
  // ------------------------+-----+------------------------> increasing
  //     <local objects>     |FP/LR|    <input arguments>     addresses
  // -----------------+------+-----+------------------------>
  //                  |      |
  //    SP/AP point --+      +-- FP points here (**)
  //    somewhere on
  //    this side of FP/LR
  //
  // (*) See LowerFormalArguments. The FP/LR is assumed to be present.
  // (**) *FP == old-FP. FP+0..7 are the bytes of FP/LR.

  // The lowering assumes that FP/LR is present, and so the offsets of
  // the formal arguments start at 8. If FP/LR is not there we need to
  // reduce the offset by 8.
  if (Offset > 0 && !HasFP)
    Offset -= 8;

  if (UseFP)
    FrameReg = FP;
  else if (UseAP)
    FrameReg = AP;
  else
    FrameReg = SP;

  // Calculate the actual offset in the instruction. If there is no FP
  // (in other words, no allocframe), then SP will not be adjusted (i.e.
  // there will be no SP -= FrameSize), so the frame size should not be
  // added to the calculated offset.
  int RealOffset = Offset;
  if (!UseFP && !UseAP && HasFP)
    RealOffset = FrameSize+Offset;
  return RealOffset;
}


bool HexagonFrameLowering::insertCSRSpillsInBlock(MachineBasicBlock &MBB,
      const CSIVect &CSI, const HexagonRegisterInfo &HRI) const {
  if (CSI.empty())
    return true;

  MachineBasicBlock::iterator MI = MBB.begin();
  MachineFunction &MF = *MBB.getParent();
  auto &HII = *MF.getSubtarget<HexagonSubtarget>().getInstrInfo();

  if (useSpillFunction(MF, CSI)) {
    unsigned MaxReg = getMaxCalleeSavedReg(CSI, HRI);
    const char *SpillFun = getSpillFunctionFor(MaxReg, SK_ToMem);
    // Call spill function.
    DebugLoc DL = MI != MBB.end() ? MI->getDebugLoc() : DebugLoc();
    MachineInstr *SaveRegsCall =
        BuildMI(MBB, MI, DL, HII.get(Hexagon::SAVE_REGISTERS_CALL_V4))
          .addExternalSymbol(SpillFun);
    // Add callee-saved registers as use.
    addCalleeSaveRegistersAsImpOperand(SaveRegsCall, MaxReg, false);
    // Add live in registers.
    for (unsigned I = 0; I < CSI.size(); ++I)
      MBB.addLiveIn(CSI[I].getReg());
    return true;
  }

  for (unsigned i = 0, n = CSI.size(); i < n; ++i) {
    unsigned Reg = CSI[i].getReg();
    // Add live in registers. We treat eh_return callee saved register r0 - r3
    // specially. They are not really callee saved registers as they are not
    // supposed to be killed.
    bool IsKill = !HRI.isEHReturnCalleeSaveReg(Reg);
    int FI = CSI[i].getFrameIdx();
    const TargetRegisterClass *RC = HRI.getMinimalPhysRegClass(Reg);
    HII.storeRegToStackSlot(MBB, MI, Reg, IsKill, FI, RC, &HRI);
    if (IsKill)
      MBB.addLiveIn(Reg);
  }
  return true;
}


bool HexagonFrameLowering::insertCSRRestoresInBlock(MachineBasicBlock &MBB,
      const CSIVect &CSI, const HexagonRegisterInfo &HRI) const {
  if (CSI.empty())
    return false;

  MachineBasicBlock::iterator MI = MBB.getFirstTerminator();
  MachineFunction &MF = *MBB.getParent();
  auto &HII = *MF.getSubtarget<HexagonSubtarget>().getInstrInfo();

  if (useRestoreFunction(MF, CSI)) {
    bool HasTC = hasTailCall(MBB) || !hasReturn(MBB);
    unsigned MaxR = getMaxCalleeSavedReg(CSI, HRI);
    SpillKind Kind = HasTC ? SK_FromMemTailcall : SK_FromMem;
    const char *RestoreFn = getSpillFunctionFor(MaxR, Kind);

    // Call spill function.
    DebugLoc DL = MI != MBB.end() ? MI->getDebugLoc()
                                  : MBB.getLastNonDebugInstr()->getDebugLoc();
    MachineInstr *DeallocCall = nullptr;

    if (HasTC) {
      unsigned ROpc = Hexagon::RESTORE_DEALLOC_BEFORE_TAILCALL_V4;
      DeallocCall = BuildMI(MBB, MI, DL, HII.get(ROpc))
          .addExternalSymbol(RestoreFn);
    } else {
      // The block has a return.
      MachineBasicBlock::iterator It = MBB.getFirstTerminator();
      assert(It->isReturn() && std::next(It) == MBB.end());
      unsigned ROpc = Hexagon::RESTORE_DEALLOC_RET_JMP_V4;
      DeallocCall = BuildMI(MBB, It, DL, HII.get(ROpc))
          .addExternalSymbol(RestoreFn);
      // Transfer the function live-out registers.
      DeallocCall->copyImplicitOps(MF, It);
    }
    addCalleeSaveRegistersAsImpOperand(DeallocCall, MaxR, true);
    return true;
  }

  for (unsigned i = 0; i < CSI.size(); ++i) {
    unsigned Reg = CSI[i].getReg();
    const TargetRegisterClass *RC = HRI.getMinimalPhysRegClass(Reg);
    int FI = CSI[i].getFrameIdx();
    HII.loadRegFromStackSlot(MBB, MI, Reg, FI, RC, &HRI);
  }
  return true;
}


void HexagonFrameLowering::eliminateCallFramePseudoInstr(MachineFunction &MF,
      MachineBasicBlock &MBB, MachineBasicBlock::iterator I) const {
  MachineInstr &MI = *I;
  unsigned Opc = MI.getOpcode();
  (void)Opc; // Silence compiler warning.
  assert((Opc == Hexagon::ADJCALLSTACKDOWN || Opc == Hexagon::ADJCALLSTACKUP) &&
         "Cannot handle this call frame pseudo instruction");
  MBB.erase(I);
}


void HexagonFrameLowering::processFunctionBeforeFrameFinalized(
    MachineFunction &MF, RegScavenger *RS) const {
  // If this function has uses aligned stack and also has variable sized stack
  // objects, then we need to map all spill slots to fixed positions, so that
  // they can be accessed through FP. Otherwise they would have to be accessed
  // via AP, which may not be available at the particular place in the program.
  MachineFrameInfo *MFI = MF.getFrameInfo();
  bool HasAlloca = MFI->hasVarSizedObjects();
  bool NeedsAlign = (MFI->getMaxAlignment() > getStackAlignment());

  if (!HasAlloca || !NeedsAlign)
    return;

  unsigned LFS = MFI->getLocalFrameSize();
  int Offset = -LFS;
  for (int i = 0, e = MFI->getObjectIndexEnd(); i != e; ++i) {
    if (!MFI->isSpillSlotObjectIndex(i) || MFI->isDeadObjectIndex(i))
      continue;
    int S = MFI->getObjectSize(i);
    LFS += S;
    Offset -= S;
    MFI->mapLocalFrameObject(i, Offset);
  }

  MFI->setLocalFrameSize(LFS);
  unsigned A = MFI->getLocalFrameMaxAlign();
  assert(A <= 8 && "Unexpected local frame alignment");
  if (A == 0)
    MFI->setLocalFrameMaxAlign(8);
  MFI->setUseLocalStackAllocationBlock(true);
}

/// Returns true if there is no caller saved registers available.
static bool needToReserveScavengingSpillSlots(MachineFunction &MF,
                                              const HexagonRegisterInfo &HRI) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const MCPhysReg *CallerSavedRegs = HRI.getCallerSavedRegs(&MF);
  // Check for an unused caller-saved register.
  for ( ; *CallerSavedRegs; ++CallerSavedRegs) {
    MCPhysReg FreeReg = *CallerSavedRegs;
    if (!MRI.reg_nodbg_empty(FreeReg))
      continue;

    // Check aliased register usage.
    bool IsCurrentRegUsed = false;
    for (MCRegAliasIterator AI(FreeReg, &HRI, false); AI.isValid(); ++AI)
      if (!MRI.reg_nodbg_empty(*AI)) {
        IsCurrentRegUsed = true;
        break;
      }
    if (IsCurrentRegUsed)
      continue;

    // Neither directly used nor used through an aliased register.
    return false;
  }
  // All caller-saved registers are used.
  return true;
}


/// Replaces the predicate spill code pseudo instructions by valid instructions.
bool HexagonFrameLowering::replacePredRegPseudoSpillCode(MachineFunction &MF)
      const {
  auto &HST = static_cast<const HexagonSubtarget&>(MF.getSubtarget());
  auto &HII = *HST.getInstrInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  bool HasReplacedPseudoInst = false;
  // Replace predicate spill pseudo instructions by real code.
  // Loop over all of the basic blocks.
  for (MachineFunction::iterator MBBb = MF.begin(), MBBe = MF.end();
       MBBb != MBBe; ++MBBb) {
    MachineBasicBlock *MBB = &*MBBb;
    // Traverse the basic block.
    MachineBasicBlock::iterator NextII;
    for (MachineBasicBlock::iterator MII = MBB->begin(); MII != MBB->end();
         MII = NextII) {
      MachineInstr *MI = MII;
      NextII = std::next(MII);
      int Opc = MI->getOpcode();
      if (Opc == Hexagon::STriw_pred) {
        HasReplacedPseudoInst = true;
        // STriw_pred FI, 0, SrcReg;
        unsigned VirtReg = MRI.createVirtualRegister(&Hexagon::IntRegsRegClass);
        unsigned SrcReg = MI->getOperand(2).getReg();
        bool IsOrigSrcRegKilled = MI->getOperand(2).isKill();

        assert(MI->getOperand(0).isFI() && "Expect a frame index");
        assert(Hexagon::PredRegsRegClass.contains(SrcReg) &&
               "Not a predicate register");

        // Insert transfer to general purpose register.
        //   VirtReg = C2_tfrpr SrcPredReg
        BuildMI(*MBB, MII, MI->getDebugLoc(), HII.get(Hexagon::C2_tfrpr),
                VirtReg).addReg(SrcReg, getKillRegState(IsOrigSrcRegKilled));

        // Change instruction to S2_storeri_io.
        //   S2_storeri_io FI, 0, VirtReg
        MI->setDesc(HII.get(Hexagon::S2_storeri_io));
        MI->getOperand(2).setReg(VirtReg);
        MI->getOperand(2).setIsKill();

      } else if (Opc == Hexagon::LDriw_pred) {
        // DstReg = LDriw_pred FI, 0
        MachineOperand &M0 = MI->getOperand(0);
        if (M0.isDead()) {
          MBB->erase(MII);
          continue;
        }

        unsigned VirtReg = MRI.createVirtualRegister(&Hexagon::IntRegsRegClass);
        unsigned DestReg = MI->getOperand(0).getReg();

        assert(MI->getOperand(1).isFI() && "Expect a frame index");
        assert(Hexagon::PredRegsRegClass.contains(DestReg) &&
               "Not a predicate register");

        // Change instruction to L2_loadri_io.
        //   VirtReg = L2_loadri_io FI, 0
        MI->setDesc(HII.get(Hexagon::L2_loadri_io));
        MI->getOperand(0).setReg(VirtReg);

        // Insert transfer to general purpose register.
        //   DestReg = C2_tfrrp VirtReg
        const MCInstrDesc &D = HII.get(Hexagon::C2_tfrrp);
        BuildMI(*MBB, std::next(MII), MI->getDebugLoc(), D, DestReg)
          .addReg(VirtReg, getKillRegState(true));
        HasReplacedPseudoInst = true;
      }
    }
  }
  return HasReplacedPseudoInst;
}


void HexagonFrameLowering::determineCalleeSaves(MachineFunction &MF,
                                                BitVector &SavedRegs,
                                                RegScavenger *RS) const {
  TargetFrameLowering::determineCalleeSaves(MF, SavedRegs, RS);

  auto &HST = static_cast<const HexagonSubtarget&>(MF.getSubtarget());
  auto &HRI = *HST.getRegisterInfo();

  bool HasEHReturn = MF.getInfo<HexagonMachineFunctionInfo>()->hasEHReturn();

  // If we have a function containing __builtin_eh_return we want to spill and
  // restore all callee saved registers. Pretend that they are used.
  if (HasEHReturn) {
    for (const MCPhysReg *CSRegs = HRI.getCalleeSavedRegs(&MF); *CSRegs;
         ++CSRegs)
      SavedRegs.set(*CSRegs);
  }

  const TargetRegisterClass &RC = Hexagon::IntRegsRegClass;

  // Replace predicate register pseudo spill code.
  bool HasReplacedPseudoInst = replacePredRegPseudoSpillCode(MF);

  // We need to reserve a a spill slot if scavenging could potentially require
  // spilling a scavenged register.
  if (HasReplacedPseudoInst && needToReserveScavengingSpillSlots(MF, HRI)) {
    MachineFrameInfo *MFI = MF.getFrameInfo();
    for (int i=0; i < NumberScavengerSlots; i++)
      RS->addScavengingFrameIndex(
        MFI->CreateSpillStackObject(RC.getSize(), RC.getAlignment()));
  }
}


#ifndef NDEBUG
static void dump_registers(BitVector &Regs, const TargetRegisterInfo &TRI) {
  dbgs() << '{';
  for (int x = Regs.find_first(); x >= 0; x = Regs.find_next(x)) {
    unsigned R = x;
    dbgs() << ' ' << PrintReg(R, &TRI);
  }
  dbgs() << " }";
}
#endif


bool HexagonFrameLowering::assignCalleeSavedSpillSlots(MachineFunction &MF,
      const TargetRegisterInfo *TRI, std::vector<CalleeSavedInfo> &CSI) const {
  DEBUG(dbgs() << LLVM_FUNCTION_NAME << " on "
               << MF.getFunction()->getName() << '\n');
  MachineFrameInfo *MFI = MF.getFrameInfo();
  BitVector SRegs(Hexagon::NUM_TARGET_REGS);

  // Generate a set of unique, callee-saved registers (SRegs), where each
  // register in the set is maximal in terms of sub-/super-register relation,
  // i.e. for each R in SRegs, no proper super-register of R is also in SRegs.

  // (1) For each callee-saved register, add that register and all of its
  // sub-registers to SRegs.
  DEBUG(dbgs() << "Initial CS registers: {");
  for (unsigned i = 0, n = CSI.size(); i < n; ++i) {
    unsigned R = CSI[i].getReg();
    DEBUG(dbgs() << ' ' << PrintReg(R, TRI));
    for (MCSubRegIterator SR(R, TRI, true); SR.isValid(); ++SR)
      SRegs[*SR] = true;
  }
  DEBUG(dbgs() << " }\n");
  DEBUG(dbgs() << "SRegs.1: "; dump_registers(SRegs, *TRI); dbgs() << "\n");

  // (2) For each reserved register, remove that register and all of its
  // sub- and super-registers from SRegs.
  BitVector Reserved = TRI->getReservedRegs(MF);
  for (int x = Reserved.find_first(); x >= 0; x = Reserved.find_next(x)) {
    unsigned R = x;
    for (MCSuperRegIterator SR(R, TRI, true); SR.isValid(); ++SR)
      SRegs[*SR] = false;
  }
  DEBUG(dbgs() << "Res:     "; dump_registers(Reserved, *TRI); dbgs() << "\n");
  DEBUG(dbgs() << "SRegs.2: "; dump_registers(SRegs, *TRI); dbgs() << "\n");

  // (3) Collect all registers that have at least one sub-register in SRegs,
  // and also have no sub-registers that are reserved. These will be the can-
  // didates for saving as a whole instead of their individual sub-registers.
  // (Saving R17:16 instead of R16 is fine, but only if R17 was not reserved.)
  BitVector TmpSup(Hexagon::NUM_TARGET_REGS);
  for (int x = SRegs.find_first(); x >= 0; x = SRegs.find_next(x)) {
    unsigned R = x;
    for (MCSuperRegIterator SR(R, TRI); SR.isValid(); ++SR)
      TmpSup[*SR] = true;
  }
  for (int x = TmpSup.find_first(); x >= 0; x = TmpSup.find_next(x)) {
    unsigned R = x;
    for (MCSubRegIterator SR(R, TRI, true); SR.isValid(); ++SR) {
      if (!Reserved[*SR])
        continue;
      TmpSup[R] = false;
      break;
    }
  }
  DEBUG(dbgs() << "TmpSup:  "; dump_registers(TmpSup, *TRI); dbgs() << "\n");

  // (4) Include all super-registers found in (3) into SRegs.
  SRegs |= TmpSup;
  DEBUG(dbgs() << "SRegs.4: "; dump_registers(SRegs, *TRI); dbgs() << "\n");

  // (5) For each register R in SRegs, if any super-register of R is in SRegs,
  // remove R from SRegs.
  for (int x = SRegs.find_first(); x >= 0; x = SRegs.find_next(x)) {
    unsigned R = x;
    for (MCSuperRegIterator SR(R, TRI); SR.isValid(); ++SR) {
      if (!SRegs[*SR])
        continue;
      SRegs[R] = false;
      break;
    }
  }
  DEBUG(dbgs() << "SRegs.5: "; dump_registers(SRegs, *TRI); dbgs() << "\n");

  // Now, for each register that has a fixed stack slot, create the stack
  // object for it.
  CSI.clear();

  typedef TargetFrameLowering::SpillSlot SpillSlot;
  unsigned NumFixed;
  int MinOffset = 0;  // CS offsets are negative.
  const SpillSlot *FixedSlots = getCalleeSavedSpillSlots(NumFixed);
  for (const SpillSlot *S = FixedSlots; S != FixedSlots+NumFixed; ++S) {
    if (!SRegs[S->Reg])
      continue;
    const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(S->Reg);
    int FI = MFI->CreateFixedSpillStackObject(RC->getSize(), S->Offset);
    MinOffset = std::min(MinOffset, S->Offset);
    CSI.push_back(CalleeSavedInfo(S->Reg, FI));
    SRegs[S->Reg] = false;
  }

  // There can be some registers that don't have fixed slots. For example,
  // we need to store R0-R3 in functions with exception handling. For each
  // such register, create a non-fixed stack object.
  for (int x = SRegs.find_first(); x >= 0; x = SRegs.find_next(x)) {
    unsigned R = x;
    const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(R);
    int Off = MinOffset - RC->getSize();
    unsigned Align = std::min(RC->getAlignment(), getStackAlignment());
    assert(isPowerOf2_32(Align));
    Off &= -Align;
    int FI = MFI->CreateFixedSpillStackObject(RC->getSize(), Off);
    MinOffset = std::min(MinOffset, Off);
    CSI.push_back(CalleeSavedInfo(R, FI));
    SRegs[R] = false;
  }

  DEBUG({
    dbgs() << "CS information: {";
    for (unsigned i = 0, n = CSI.size(); i < n; ++i) {
      int FI = CSI[i].getFrameIdx();
      int Off = MFI->getObjectOffset(FI);
      dbgs() << ' ' << PrintReg(CSI[i].getReg(), TRI) << ":fi#" << FI << ":sp";
      if (Off >= 0)
        dbgs() << '+';
      dbgs() << Off;
    }
    dbgs() << " }\n";
  });

#ifndef NDEBUG
  // Verify that all registers were handled.
  bool MissedReg = false;
  for (int x = SRegs.find_first(); x >= 0; x = SRegs.find_next(x)) {
    unsigned R = x;
    dbgs() << PrintReg(R, TRI) << ' ';
    MissedReg = true;
  }
  if (MissedReg)
    llvm_unreachable("...there are unhandled callee-saved registers!");
#endif

  return true;
}


void HexagonFrameLowering::expandAlloca(MachineInstr *AI,
      const HexagonInstrInfo &HII, unsigned SP, unsigned CF) const {
  MachineBasicBlock &MB = *AI->getParent();
  DebugLoc DL = AI->getDebugLoc();
  unsigned A = AI->getOperand(2).getImm();

  // Have
  //    Rd  = alloca Rs, #A
  //
  // If Rs and Rd are different registers, use this sequence:
  //    Rd  = sub(r29, Rs)
  //    r29 = sub(r29, Rs)
  //    Rd  = and(Rd, #-A)    ; if necessary
  //    r29 = and(r29, #-A)   ; if necessary
  //    Rd  = add(Rd, #CF)    ; CF size aligned to at most A
  // otherwise, do
  //    Rd  = sub(r29, Rs)
  //    Rd  = and(Rd, #-A)    ; if necessary
  //    r29 = Rd
  //    Rd  = add(Rd, #CF)    ; CF size aligned to at most A

  MachineOperand &RdOp = AI->getOperand(0);
  MachineOperand &RsOp = AI->getOperand(1);
  unsigned Rd = RdOp.getReg(), Rs = RsOp.getReg();

  // Rd = sub(r29, Rs)
  BuildMI(MB, AI, DL, HII.get(Hexagon::A2_sub), Rd)
      .addReg(SP)
      .addReg(Rs);
  if (Rs != Rd) {
    // r29 = sub(r29, Rs)
    BuildMI(MB, AI, DL, HII.get(Hexagon::A2_sub), SP)
        .addReg(SP)
        .addReg(Rs);
  }
  if (A > 8) {
    // Rd  = and(Rd, #-A)
    BuildMI(MB, AI, DL, HII.get(Hexagon::A2_andir), Rd)
        .addReg(Rd)
        .addImm(-int64_t(A));
    if (Rs != Rd)
      BuildMI(MB, AI, DL, HII.get(Hexagon::A2_andir), SP)
          .addReg(SP)
          .addImm(-int64_t(A));
  }
  if (Rs == Rd) {
    // r29 = Rd
    BuildMI(MB, AI, DL, HII.get(TargetOpcode::COPY), SP)
        .addReg(Rd);
  }
  if (CF > 0) {
    // Rd = add(Rd, #CF)
    BuildMI(MB, AI, DL, HII.get(Hexagon::A2_addi), Rd)
        .addReg(Rd)
        .addImm(CF);
  }
}


bool HexagonFrameLowering::needsAligna(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  if (!MFI->hasVarSizedObjects())
    return false;
  unsigned MaxA = MFI->getMaxAlignment();
  if (MaxA <= getStackAlignment())
    return false;
  return true;
}


const MachineInstr *HexagonFrameLowering::getAlignaInstr(
      const MachineFunction &MF) const {
  for (auto &B : MF)
    for (auto &I : B)
      if (I.getOpcode() == Hexagon::ALIGNA)
        return &I;
  return nullptr;
}


// FIXME: Use Function::optForSize().
inline static bool isOptSize(const MachineFunction &MF) {
  AttributeSet AF = MF.getFunction()->getAttributes();
  return AF.hasAttribute(AttributeSet::FunctionIndex,
                         Attribute::OptimizeForSize);
}

inline static bool isMinSize(const MachineFunction &MF) {
  return MF.getFunction()->optForMinSize();
}


/// Determine whether the callee-saved register saves and restores should
/// be generated via inline code. If this function returns "true", inline
/// code will be generated. If this function returns "false", additional
/// checks are performed, which may still lead to the inline code.
bool HexagonFrameLowering::shouldInlineCSR(MachineFunction &MF,
      const CSIVect &CSI) const {
  if (MF.getInfo<HexagonMachineFunctionInfo>()->hasEHReturn())
    return true;
  if (!isOptSize(MF) && !isMinSize(MF))
    if (MF.getTarget().getOptLevel() > CodeGenOpt::Default)
      return true;

  // Check if CSI only has double registers, and if the registers form
  // a contiguous block starting from D8.
  BitVector Regs(Hexagon::NUM_TARGET_REGS);
  for (unsigned i = 0, n = CSI.size(); i < n; ++i) {
    unsigned R = CSI[i].getReg();
    if (!Hexagon::DoubleRegsRegClass.contains(R))
      return true;
    Regs[R] = true;
  }
  int F = Regs.find_first();
  if (F != Hexagon::D8)
    return true;
  while (F >= 0) {
    int N = Regs.find_next(F);
    if (N >= 0 && N != F+1)
      return true;
    F = N;
  }

  return false;
}


bool HexagonFrameLowering::useSpillFunction(MachineFunction &MF,
      const CSIVect &CSI) const {
  if (shouldInlineCSR(MF, CSI))
    return false;
  unsigned NumCSI = CSI.size();
  if (NumCSI <= 1)
    return false;

  unsigned Threshold = isOptSize(MF) ? SpillFuncThresholdOs
                                     : SpillFuncThreshold;
  return Threshold < NumCSI;
}


bool HexagonFrameLowering::useRestoreFunction(MachineFunction &MF,
      const CSIVect &CSI) const {
  if (shouldInlineCSR(MF, CSI))
    return false;
  unsigned NumCSI = CSI.size();
  unsigned Threshold = isOptSize(MF) ? SpillFuncThresholdOs-1
                                     : SpillFuncThreshold;
  return Threshold < NumCSI;
}
