//===-- X86FixupBWInsts.cpp - Fixup Byte or Word instructions -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file defines the pass that looks through the machine instructions
/// late in the compilation, and finds byte or word instructions that
/// can be profitably replaced with 32 bit instructions that give equivalent
/// results for the bits of the results that are used. There are two possible
/// reasons to do this.
///
/// One reason is to avoid false-dependences on the upper portions
/// of the registers.  Only instructions that have a destination register
/// which is not in any of the source registers can be affected by this.
/// Any instruction where one of the source registers is also the destination
/// register is unaffected, because it has a true dependence on the source
/// register already.  So, this consideration primarily affects load
/// instructions and register-to-register moves.  It would
/// seem like cmov(s) would also be affected, but because of the way cmov is
/// really implemented by most machines as reading both the destination and
/// and source regsters, and then "merging" the two based on a condition,
/// it really already should be considered as having a true dependence on the
/// destination register as well.
///
/// The other reason to do this is for potential code size savings.  Word
/// operations need an extra override byte compared to their 32 bit
/// versions. So this can convert many word operations to their larger
/// size, saving a byte in encoding. This could introduce partial register
/// dependences where none existed however.  As an example take:
///   orw  ax, $0x1000
///   addw ax, $3
/// now if this were to get transformed into
///   orw  ax, $1000
///   addl eax, $3
/// because the addl encodes shorter than the addw, this would introduce
/// a use of a register that was only partially written earlier.  On older
/// Intel processors this can be quite a performance penalty, so this should
/// probably only be done when it can be proven that a new partial dependence
/// wouldn't be created, or when your know a newer processor is being
/// targeted, or when optimizing for minimum code size.
///
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
using namespace llvm;

#define FIXUPBW_DESC "X86 Byte/Word Instruction Fixup"
#define FIXUPBW_NAME "x86-fixup-bw-insts"

#define DEBUG_TYPE FIXUPBW_NAME

// Option to allow this optimization pass to have fine-grained control.
// This is turned off by default so as not to affect a large number of
// existing lit tests.
static cl::opt<bool>
    FixupBWInsts("fixup-byte-word-insts",
                 cl::desc("Change byte and word instructions to larger sizes"),
                 cl::init(true), cl::Hidden);

namespace {
class FixupBWInstPass : public MachineFunctionPass {
  /// Loop over all of the instructions in the basic block replacing applicable
  /// byte or word instructions with better alternatives.
  void processBasicBlock(MachineFunction &MF, MachineBasicBlock &MBB);

  /// This sets the \p SuperDestReg to the 32 bit super reg of the original
  /// destination register of the MachineInstr passed in. It returns true if
  /// that super register is dead just prior to \p OrigMI, and false if not.
  bool getSuperRegDestIfDead(MachineInstr *OrigMI,
                             unsigned &SuperDestReg) const;

  /// Change the MachineInstr \p MI into the equivalent extending load to 32 bit
  /// register if it is safe to do so.  Return the replacement instruction if
  /// OK, otherwise return nullptr.
  MachineInstr *tryReplaceLoad(unsigned New32BitOpcode, MachineInstr *MI) const;

  /// Change the MachineInstr \p MI into the equivalent 32-bit copy if it is
  /// safe to do so.  Return the replacement instruction if OK, otherwise return
  /// nullptr.
  MachineInstr *tryReplaceCopy(MachineInstr *MI) const;

  // Change the MachineInstr \p MI into an eqivalent 32 bit instruction if
  // possible.  Return the replacement instruction if OK, return nullptr
  // otherwise. Set WasCandidate to true or false depending on whether the
  // MI was a candidate for this sort of transformation.
  MachineInstr *tryReplaceInstr(MachineInstr *MI, MachineBasicBlock &MBB,
                                bool &WasCandidate) const;
public:
  static char ID;

  const char *getPassName() const override {
    return FIXUPBW_DESC;
  }

  FixupBWInstPass() : MachineFunctionPass(ID) {
    initializeFixupBWInstPassPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineLoopInfo>(); // Machine loop info is used to
                                       // guide some heuristics.
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  /// Loop over all of the basic blocks, replacing byte and word instructions by
  /// equivalent 32 bit instructions where performance or code size can be
  /// improved.
  bool runOnMachineFunction(MachineFunction &MF) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }

private:
  MachineFunction *MF;

  /// Machine instruction info used throughout the class.
  const X86InstrInfo *TII;

  /// Local member for function's OptForSize attribute.
  bool OptForSize;

  /// Machine loop info used for guiding some heruistics.
  MachineLoopInfo *MLI;

  /// Register Liveness information after the current instruction.
  LivePhysRegs LiveRegs;
};
char FixupBWInstPass::ID = 0;
}

INITIALIZE_PASS(FixupBWInstPass, FIXUPBW_NAME, FIXUPBW_DESC, false, false)

FunctionPass *llvm::createX86FixupBWInsts() { return new FixupBWInstPass(); }

bool FixupBWInstPass::runOnMachineFunction(MachineFunction &MF) {
  if (!FixupBWInsts || skipFunction(*MF.getFunction()))
    return false;

  this->MF = &MF;
  TII = MF.getSubtarget<X86Subtarget>().getInstrInfo();
  OptForSize = MF.getFunction()->optForSize();
  MLI = &getAnalysis<MachineLoopInfo>();
  LiveRegs.init(&TII->getRegisterInfo());

  DEBUG(dbgs() << "Start X86FixupBWInsts\n";);

  // Process all basic blocks.
  for (auto &MBB : MF)
    processBasicBlock(MF, MBB);

  DEBUG(dbgs() << "End X86FixupBWInsts\n";);

  return true;
}

// TODO: This method of analysis can miss some legal cases, because the
// super-register could be live into the address expression for a memory
// reference for the instruction, and still be killed/last used by the
// instruction. However, the existing query interfaces don't seem to
// easily allow that to be checked.
//
// What we'd really like to know is whether after OrigMI, the
// only portion of SuperDestReg that is alive is the portion that
// was the destination register of OrigMI.
bool FixupBWInstPass::getSuperRegDestIfDead(MachineInstr *OrigMI,
                                            unsigned &SuperDestReg) const {
  auto *TRI = &TII->getRegisterInfo();

  unsigned OrigDestReg = OrigMI->getOperand(0).getReg();
  SuperDestReg = getX86SubSuperRegister(OrigDestReg, 32);

  const auto SubRegIdx = TRI->getSubRegIndex(SuperDestReg, OrigDestReg);

  // Make sure that the sub-register that this instruction has as its
  // destination is the lowest order sub-register of the super-register.
  // If it isn't, then the register isn't really dead even if the
  // super-register is considered dead.
  if (SubRegIdx == X86::sub_8bit_hi)
    return false;

  if (LiveRegs.contains(SuperDestReg))
    return false;

  if (SubRegIdx == X86::sub_8bit) {
    // In the case of byte registers, we also have to check that the upper
    // byte register is also dead. That is considered to be independent of
    // whether the super-register is dead.
    unsigned UpperByteReg =
        getX86SubSuperRegister(SuperDestReg, 8, /*High=*/true);

    if (LiveRegs.contains(UpperByteReg))
      return false;
  }

  return true;
}

MachineInstr *FixupBWInstPass::tryReplaceLoad(unsigned New32BitOpcode,
                                              MachineInstr *MI) const {
  unsigned NewDestReg;

  // We are going to try to rewrite this load to a larger zero-extending
  // load.  This is safe if all portions of the 32 bit super-register
  // of the original destination register, except for the original destination
  // register are dead. getSuperRegDestIfDead checks that.
  if (!getSuperRegDestIfDead(MI, NewDestReg))
    return nullptr;

  // Safe to change the instruction.
  MachineInstrBuilder MIB =
      BuildMI(*MF, MI->getDebugLoc(), TII->get(New32BitOpcode), NewDestReg);

  unsigned NumArgs = MI->getNumOperands();
  for (unsigned i = 1; i < NumArgs; ++i)
    MIB.addOperand(MI->getOperand(i));

  MIB->setMemRefs(MI->memoperands_begin(), MI->memoperands_end());

  return MIB;
}

MachineInstr *FixupBWInstPass::tryReplaceCopy(MachineInstr *MI) const {
  assert(MI->getNumExplicitOperands() == 2);
  auto &OldDest = MI->getOperand(0);
  auto &OldSrc = MI->getOperand(1);

  unsigned NewDestReg;
  if (!getSuperRegDestIfDead(MI, NewDestReg))
    return nullptr;

  unsigned NewSrcReg = getX86SubSuperRegister(OldSrc.getReg(), 32);

  // This is only correct if we access the same subregister index: otherwise,
  // we could try to replace "movb %ah, %al" with "movl %eax, %eax".
  auto *TRI = &TII->getRegisterInfo();
  if (TRI->getSubRegIndex(NewSrcReg, OldSrc.getReg()) !=
      TRI->getSubRegIndex(NewDestReg, OldDest.getReg()))
    return nullptr;

  // Safe to change the instruction.
  // Don't set src flags, as we don't know if we're also killing the superreg.
  // However, the superregister might not be defined; make it explicit that
  // we don't care about the higher bits by reading it as Undef, and adding
  // an imp-use on the original subregister.
  MachineInstrBuilder MIB =
      BuildMI(*MF, MI->getDebugLoc(), TII->get(X86::MOV32rr), NewDestReg)
          .addReg(NewSrcReg, RegState::Undef)
          .addReg(OldSrc.getReg(), RegState::Implicit);

  // Drop imp-defs/uses that would be redundant with the new def/use.
  for (auto &Op : MI->implicit_operands())
    if (Op.getReg() != (Op.isDef() ? NewDestReg : NewSrcReg))
      MIB.addOperand(Op);

  return MIB;
}

MachineInstr *FixupBWInstPass::tryReplaceInstr(
                  MachineInstr *MI, MachineBasicBlock &MBB,
                  bool &WasCandidate) const {
  MachineInstr *NewMI = nullptr;
  WasCandidate = false;

  // See if this is an instruction of the type we are currently looking for.
  switch (MI->getOpcode()) {

  case X86::MOV8rm:
    // Only replace 8 bit loads with the zero extending versions if
    // in an inner most loop and not optimizing for size. This takes
    // an extra byte to encode, and provides limited performance upside.
    if (MachineLoop *ML = MLI->getLoopFor(&MBB)) {
      if (ML->begin() == ML->end() && !OptForSize) {
        NewMI = tryReplaceLoad(X86::MOVZX32rm8, MI);
        WasCandidate = true;
      }
    }
    break;

  case X86::MOV16rm:
    // Always try to replace 16 bit load with 32 bit zero extending.
    // Code size is the same, and there is sometimes a perf advantage
    // from eliminating a false dependence on the upper portion of
    // the register.
    NewMI = tryReplaceLoad(X86::MOVZX32rm16, MI);
    WasCandidate = true;
    break;

  case X86::MOV8rr:
  case X86::MOV16rr:
    // Always try to replace 8/16 bit copies with a 32 bit copy.
    // Code size is either less (16) or equal (8), and there is sometimes a
    // perf advantage from eliminating a false dependence on the upper portion
    // of the register.
    NewMI = tryReplaceCopy(MI);
    WasCandidate = true;
    break;

  default:
    // nothing to do here.
    break;
  }

  return NewMI;
}

void FixupBWInstPass::processBasicBlock(MachineFunction &MF,
                                        MachineBasicBlock &MBB) {

  // This algorithm doesn't delete the instructions it is replacing
  // right away.  By leaving the existing instructions in place, the
  // register liveness information doesn't change, and this makes the
  // analysis that goes on be better than if the replaced instructions
  // were immediately removed.
  //
  // This algorithm always creates a replacement instruction
  // and notes that and the original in a data structure, until the
  // whole BB has been analyzed.  This keeps the replacement instructions
  // from making it seem as if the larger register might be live.
  SmallVector<std::pair<MachineInstr *, MachineInstr *>, 8> MIReplacements;

  // Start computing liveness for this block. We iterate from the end to be able
  // to update this for each instruction.
  LiveRegs.clear();
  // We run after PEI, so we need to AddPristinesAndCSRs.
  LiveRegs.addLiveOuts(MBB);

  bool WasCandidate = false;

  for (auto I = MBB.rbegin(); I != MBB.rend(); ++I) {
    MachineInstr *MI = &*I;
    
    MachineInstr *NewMI = tryReplaceInstr(MI, MBB, WasCandidate);

    // Add this to replacements if it was a candidate, even if NewMI is
    // nullptr.  We will revisit that in a bit.
    if (WasCandidate) {
      MIReplacements.push_back(std::make_pair(MI, NewMI));
    }

    // We're done with this instruction, update liveness for the next one.
    LiveRegs.stepBackward(*MI);
  }

  while (!MIReplacements.empty()) {
    MachineInstr *MI = MIReplacements.back().first;
    MachineInstr *NewMI = MIReplacements.back().second;
    MIReplacements.pop_back();
    if (NewMI) {
      MBB.insert(MI, NewMI);
      MBB.erase(MI);
    }
  }
}
