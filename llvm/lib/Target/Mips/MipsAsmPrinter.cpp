//===-- MipsAsmPrinter.cpp - Mips LLVM Assembly Printer -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to GAS-format MIPS assembly language.
//
//===----------------------------------------------------------------------===//

#include "InstPrinter/MipsInstPrinter.h"
#include "MCTargetDesc/MipsBaseInfo.h"
#include "MCTargetDesc/MipsMCNaCl.h"
#include "Mips.h"
#include "MipsAsmPrinter.h"
#include "MipsInstrInfo.h"
#include "MipsMCInstLower.h"
#include "MipsTargetStreamer.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Mangler.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetOptions.h"
#include <string>

using namespace llvm;

#define DEBUG_TYPE "mips-asm-printer"

MipsTargetStreamer &MipsAsmPrinter::getTargetStreamer() const {
  return static_cast<MipsTargetStreamer &>(*OutStreamer.getTargetStreamer());
}

bool MipsAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  Subtarget = &TM.getSubtarget<MipsSubtarget>();

  // Initialize TargetLoweringObjectFile.
  const_cast<TargetLoweringObjectFile &>(getObjFileLowering())
      .Initialize(OutContext, TM);

  MipsFI = MF.getInfo<MipsFunctionInfo>();
  if (Subtarget->inMips16Mode())
    for (std::map<
             const char *,
             const llvm::Mips16HardFloatInfo::FuncSignature *>::const_iterator
             it = MipsFI->StubsNeeded.begin();
         it != MipsFI->StubsNeeded.end(); ++it) {
      const char *Symbol = it->first;
      const llvm::Mips16HardFloatInfo::FuncSignature *Signature = it->second;
      if (StubsNeeded.find(Symbol) == StubsNeeded.end())
        StubsNeeded[Symbol] = Signature;
    }
  MCP = MF.getConstantPool();

  // In NaCl, all indirect jump targets must be aligned to bundle size.
  if (Subtarget->isTargetNaCl())
    NaClAlignIndirectJumpTargets(MF);

  AsmPrinter::runOnMachineFunction(MF);
  return true;
}

bool MipsAsmPrinter::lowerOperand(const MachineOperand &MO, MCOperand &MCOp) {
  MCOp = MCInstLowering.LowerOperand(MO);
  return MCOp.isValid();
}

#include "MipsGenMCPseudoLowering.inc"

// Lower PseudoReturn/PseudoIndirectBranch/PseudoIndirectBranch64 to JR, JR_MM,
// JALR, or JALR64 as appropriate for the target
void MipsAsmPrinter::emitPseudoIndirectBranch(MCStreamer &OutStreamer,
                                              const MachineInstr *MI) {
  bool HasLinkReg = false;
  MCInst TmpInst0;

  if (Subtarget->hasMips64r6()) {
    // MIPS64r6 should use (JALR64 ZERO_64, $rs)
    TmpInst0.setOpcode(Mips::JALR64);
    HasLinkReg = true;
  } else if (Subtarget->hasMips32r6()) {
    // MIPS32r6 should use (JALR ZERO, $rs)
    TmpInst0.setOpcode(Mips::JALR);
    HasLinkReg = true;
  } else if (Subtarget->inMicroMipsMode())
    // microMIPS should use (JR_MM $rs)
    TmpInst0.setOpcode(Mips::JR_MM);
  else {
    // Everything else should use (JR $rs)
    TmpInst0.setOpcode(Mips::JR);
  }

  MCOperand MCOp;

  if (HasLinkReg) {
    unsigned ZeroReg = Subtarget->isGP64bit() ? Mips::ZERO_64 : Mips::ZERO;
    TmpInst0.addOperand(MCOperand::CreateReg(ZeroReg));
  }

  lowerOperand(MI->getOperand(0), MCOp);
  TmpInst0.addOperand(MCOp);

  EmitToStreamer(OutStreamer, TmpInst0);
}

void MipsAsmPrinter::EmitInstruction(const MachineInstr *MI) {
  MipsTargetStreamer &TS = getTargetStreamer();
  TS.setCanHaveModuleDir(false);

  if (MI->isDebugValue()) {
    SmallString<128> Str;
    raw_svector_ostream OS(Str);

    PrintDebugValueComment(MI, OS);
    return;
  }

  // If we just ended a constant pool, mark it as such.
  if (InConstantPool && MI->getOpcode() != Mips::CONSTPOOL_ENTRY) {
    OutStreamer.EmitDataRegion(MCDR_DataRegionEnd);
    InConstantPool = false;
  }
  if (MI->getOpcode() == Mips::CONSTPOOL_ENTRY) {
    // CONSTPOOL_ENTRY - This instruction represents a floating
    //constant pool in the function.  The first operand is the ID#
    // for this instruction, the second is the index into the
    // MachineConstantPool that this is, the third is the size in
    // bytes of this constant pool entry.
    // The required alignment is specified on the basic block holding this MI.
    //
    unsigned LabelId = (unsigned)MI->getOperand(0).getImm();
    unsigned CPIdx   = (unsigned)MI->getOperand(1).getIndex();

    // If this is the first entry of the pool, mark it.
    if (!InConstantPool) {
      OutStreamer.EmitDataRegion(MCDR_DataRegion);
      InConstantPool = true;
    }

    OutStreamer.EmitLabel(GetCPISymbol(LabelId));

    const MachineConstantPoolEntry &MCPE = MCP->getConstants()[CPIdx];
    if (MCPE.isMachineConstantPoolEntry())
      EmitMachineConstantPoolValue(MCPE.Val.MachineCPVal);
    else
      EmitGlobalConstant(MCPE.Val.ConstVal);
    return;
  }


  MachineBasicBlock::const_instr_iterator I = MI;
  MachineBasicBlock::const_instr_iterator E = MI->getParent()->instr_end();

  do {
    // Do any auto-generated pseudo lowerings.
    if (emitPseudoExpansionLowering(OutStreamer, &*I))
      continue;

    if (I->getOpcode() == Mips::PseudoReturn ||
        I->getOpcode() == Mips::PseudoReturn64 ||
        I->getOpcode() == Mips::PseudoIndirectBranch ||
        I->getOpcode() == Mips::PseudoIndirectBranch64) {
      emitPseudoIndirectBranch(OutStreamer, &*I);
      continue;
    }

    // The inMips16Mode() test is not permanent.
    // Some instructions are marked as pseudo right now which
    // would make the test fail for the wrong reason but
    // that will be fixed soon. We need this here because we are
    // removing another test for this situation downstream in the
    // callchain.
    //
    if (I->isPseudo() && !Subtarget->inMips16Mode()
        && !isLongBranchPseudo(I->getOpcode()))
      llvm_unreachable("Pseudo opcode found in EmitInstruction()");

    MCInst TmpInst0;
    MCInstLowering.Lower(I, TmpInst0);
    EmitToStreamer(OutStreamer, TmpInst0);
  } while ((++I != E) && I->isInsideBundle()); // Delay slot check
}

//===----------------------------------------------------------------------===//
//
//  Mips Asm Directives
//
//  -- Frame directive "frame Stackpointer, Stacksize, RARegister"
//  Describe the stack frame.
//
//  -- Mask directives "(f)mask  bitmask, offset"
//  Tells the assembler which registers are saved and where.
//  bitmask - contain a little endian bitset indicating which registers are
//            saved on function prologue (e.g. with a 0x80000000 mask, the
//            assembler knows the register 31 (RA) is saved at prologue.
//  offset  - the position before stack pointer subtraction indicating where
//            the first saved register on prologue is located. (e.g. with a
//
//  Consider the following function prologue:
//
//    .frame  $fp,48,$ra
//    .mask   0xc0000000,-8
//       addiu $sp, $sp, -48
//       sw $ra, 40($sp)
//       sw $fp, 36($sp)
//
//    With a 0xc0000000 mask, the assembler knows the register 31 (RA) and
//    30 (FP) are saved at prologue. As the save order on prologue is from
//    left to right, RA is saved first. A -8 offset means that after the
//    stack pointer subtration, the first register in the mask (RA) will be
//    saved at address 48-8=40.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Mask directives
//===----------------------------------------------------------------------===//

// Create a bitmask with all callee saved registers for CPU or Floating Point
// registers. For CPU registers consider RA, GP and FP for saving if necessary.
void MipsAsmPrinter::printSavedRegsBitmask() {
  // CPU and FPU Saved Registers Bitmasks
  unsigned CPUBitmask = 0, FPUBitmask = 0;
  int CPUTopSavedRegOff, FPUTopSavedRegOff;

  // Set the CPU and FPU Bitmasks
  const MachineFrameInfo *MFI = MF->getFrameInfo();
  const std::vector<CalleeSavedInfo> &CSI = MFI->getCalleeSavedInfo();
  // size of stack area to which FP callee-saved regs are saved.
  unsigned CPURegSize = Mips::GPR32RegClass.getSize();
  unsigned FGR32RegSize = Mips::FGR32RegClass.getSize();
  unsigned AFGR64RegSize = Mips::AFGR64RegClass.getSize();
  bool HasAFGR64Reg = false;
  unsigned CSFPRegsSize = 0;
  unsigned i, e = CSI.size();

  // Set FPU Bitmask.
  for (i = 0; i != e; ++i) {
    unsigned Reg = CSI[i].getReg();
    if (Mips::GPR32RegClass.contains(Reg))
      break;

    unsigned RegNum = TM.getRegisterInfo()->getEncodingValue(Reg);
    if (Mips::AFGR64RegClass.contains(Reg)) {
      FPUBitmask |= (3 << RegNum);
      CSFPRegsSize += AFGR64RegSize;
      HasAFGR64Reg = true;
      continue;
    }

    FPUBitmask |= (1 << RegNum);
    CSFPRegsSize += FGR32RegSize;
  }

  // Set CPU Bitmask.
  for (; i != e; ++i) {
    unsigned Reg = CSI[i].getReg();
    unsigned RegNum = TM.getRegisterInfo()->getEncodingValue(Reg);
    CPUBitmask |= (1 << RegNum);
  }

  // FP Regs are saved right below where the virtual frame pointer points to.
  FPUTopSavedRegOff = FPUBitmask ?
    (HasAFGR64Reg ? -AFGR64RegSize : -FGR32RegSize) : 0;

  // CPU Regs are saved below FP Regs.
  CPUTopSavedRegOff = CPUBitmask ? -CSFPRegsSize - CPURegSize : 0;

  MipsTargetStreamer &TS = getTargetStreamer();
  // Print CPUBitmask
  TS.emitMask(CPUBitmask, CPUTopSavedRegOff);

  // Print FPUBitmask
  TS.emitFMask(FPUBitmask, FPUTopSavedRegOff);
}

//===----------------------------------------------------------------------===//
// Frame and Set directives
//===----------------------------------------------------------------------===//

/// Frame Directive
void MipsAsmPrinter::emitFrameDirective() {
  const TargetRegisterInfo &RI = *TM.getRegisterInfo();

  unsigned stackReg  = RI.getFrameRegister(*MF);
  unsigned returnReg = RI.getRARegister();
  unsigned stackSize = MF->getFrameInfo()->getStackSize();

  getTargetStreamer().emitFrame(stackReg, stackSize, returnReg);
}

/// Emit Set directives.
const char *MipsAsmPrinter::getCurrentABIString() const {
  switch (Subtarget->getABI().GetEnumValue()) {
  case MipsABIInfo::ABI::O32:  return "abi32";
  case MipsABIInfo::ABI::N32:  return "abiN32";
  case MipsABIInfo::ABI::N64:  return "abi64";
  case MipsABIInfo::ABI::EABI: return "eabi32"; // TODO: handle eabi64
  default: llvm_unreachable("Unknown Mips ABI");
  }
}

void MipsAsmPrinter::EmitFunctionEntryLabel() {
  MipsTargetStreamer &TS = getTargetStreamer();

  // NaCl sandboxing requires that indirect call instructions are masked.
  // This means that function entry points should be bundle-aligned.
  if (Subtarget->isTargetNaCl())
    EmitAlignment(std::max(MF->getAlignment(), MIPS_NACL_BUNDLE_ALIGN));

  if (Subtarget->inMicroMipsMode())
    TS.emitDirectiveSetMicroMips();
  else
    TS.emitDirectiveSetNoMicroMips();

  if (Subtarget->inMips16Mode())
    TS.emitDirectiveSetMips16();
  else
    TS.emitDirectiveSetNoMips16();

  TS.emitDirectiveEnt(*CurrentFnSym);
  OutStreamer.EmitLabel(CurrentFnSym);
}

/// EmitFunctionBodyStart - Targets can override this to emit stuff before
/// the first basic block in the function.
void MipsAsmPrinter::EmitFunctionBodyStart() {
  MipsTargetStreamer &TS = getTargetStreamer();

  MCInstLowering.Initialize(&MF->getContext());

  bool IsNakedFunction =
    MF->getFunction()->
      getAttributes().hasAttribute(AttributeSet::FunctionIndex,
                                   Attribute::Naked);
  if (!IsNakedFunction)
    emitFrameDirective();

  if (!IsNakedFunction)
    printSavedRegsBitmask();

  if (!Subtarget->inMips16Mode()) {
    TS.emitDirectiveSetNoReorder();
    TS.emitDirectiveSetNoMacro();
    TS.emitDirectiveSetNoAt();
  }
}

/// EmitFunctionBodyEnd - Targets can override this to emit stuff after
/// the last basic block in the function.
void MipsAsmPrinter::EmitFunctionBodyEnd() {
  MipsTargetStreamer &TS = getTargetStreamer();

  // There are instruction for this macros, but they must
  // always be at the function end, and we can't emit and
  // break with BB logic.
  if (!Subtarget->inMips16Mode()) {
    TS.emitDirectiveSetAt();
    TS.emitDirectiveSetMacro();
    TS.emitDirectiveSetReorder();
  }
  TS.emitDirectiveEnd(CurrentFnSym->getName());
  // Make sure to terminate any constant pools that were at the end
  // of the function.
  if (!InConstantPool)
    return;
  InConstantPool = false;
  OutStreamer.EmitDataRegion(MCDR_DataRegionEnd);
}

/// isBlockOnlyReachableByFallthough - Return true if the basic block has
/// exactly one predecessor and the control transfer mechanism between
/// the predecessor and this block is a fall-through.
bool MipsAsmPrinter::isBlockOnlyReachableByFallthrough(const MachineBasicBlock*
                                                       MBB) const {
  // The predecessor has to be immediately before this block.
  const MachineBasicBlock *Pred = *MBB->pred_begin();

  // If the predecessor is a switch statement, assume a jump table
  // implementation, so it is not a fall through.
  if (const BasicBlock *bb = Pred->getBasicBlock())
    if (isa<SwitchInst>(bb->getTerminator()))
      return false;

  // If this is a landing pad, it isn't a fall through.  If it has no preds,
  // then nothing falls through to it.
  if (MBB->isLandingPad() || MBB->pred_empty())
    return false;

  // If there isn't exactly one predecessor, it can't be a fall through.
  MachineBasicBlock::const_pred_iterator PI = MBB->pred_begin(), PI2 = PI;
  ++PI2;

  if (PI2 != MBB->pred_end())
    return false;

  // The predecessor has to be immediately before this block.
  if (!Pred->isLayoutSuccessor(MBB))
    return false;

  // If the block is completely empty, then it definitely does fall through.
  if (Pred->empty())
    return true;

  // Otherwise, check the last instruction.
  // Check if the last terminator is an unconditional branch.
  MachineBasicBlock::const_iterator I = Pred->end();
  while (I != Pred->begin() && !(--I)->isTerminator()) ;

  return !I->isBarrier();
}

// Print out an operand for an inline asm expression.
bool MipsAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNum,
                                     unsigned AsmVariant,const char *ExtraCode,
                                     raw_ostream &O) {
  // Does this asm operand have a single letter operand modifier?
  if (ExtraCode && ExtraCode[0]) {
    if (ExtraCode[1] != 0) return true; // Unknown modifier.

    const MachineOperand &MO = MI->getOperand(OpNum);
    switch (ExtraCode[0]) {
    default:
      // See if this is a generic print operand
      return AsmPrinter::PrintAsmOperand(MI,OpNum,AsmVariant,ExtraCode,O);
    case 'X': // hex const int
      if ((MO.getType()) != MachineOperand::MO_Immediate)
        return true;
      O << "0x" << StringRef(utohexstr(MO.getImm())).lower();
      return false;
    case 'x': // hex const int (low 16 bits)
      if ((MO.getType()) != MachineOperand::MO_Immediate)
        return true;
      O << "0x" << StringRef(utohexstr(MO.getImm() & 0xffff)).lower();
      return false;
    case 'd': // decimal const int
      if ((MO.getType()) != MachineOperand::MO_Immediate)
        return true;
      O << MO.getImm();
      return false;
    case 'm': // decimal const int minus 1
      if ((MO.getType()) != MachineOperand::MO_Immediate)
        return true;
      O << MO.getImm() - 1;
      return false;
    case 'z': {
      // $0 if zero, regular printing otherwise
      if (MO.getType() == MachineOperand::MO_Immediate && MO.getImm() == 0) {
        O << "$0";
        return false;
      }
      // If not, call printOperand as normal.
      break;
    }
    case 'D': // Second part of a double word register operand
    case 'L': // Low order register of a double word register operand
    case 'M': // High order register of a double word register operand
    {
      if (OpNum == 0)
        return true;
      const MachineOperand &FlagsOP = MI->getOperand(OpNum - 1);
      if (!FlagsOP.isImm())
        return true;
      unsigned Flags = FlagsOP.getImm();
      unsigned NumVals = InlineAsm::getNumOperandRegisters(Flags);
      // Number of registers represented by this operand. We are looking
      // for 2 for 32 bit mode and 1 for 64 bit mode.
      if (NumVals != 2) {
        if (Subtarget->isGP64bit() && NumVals == 1 && MO.isReg()) {
          unsigned Reg = MO.getReg();
          O << '$' << MipsInstPrinter::getRegisterName(Reg);
          return false;
        }
        return true;
      }

      unsigned RegOp = OpNum;
      if (!Subtarget->isGP64bit()){
        // Endianess reverses which register holds the high or low value
        // between M and L.
        switch(ExtraCode[0]) {
        case 'M':
          RegOp = (Subtarget->isLittle()) ? OpNum + 1 : OpNum;
          break;
        case 'L':
          RegOp = (Subtarget->isLittle()) ? OpNum : OpNum + 1;
          break;
        case 'D': // Always the second part
          RegOp = OpNum + 1;
        }
        if (RegOp >= MI->getNumOperands())
          return true;
        const MachineOperand &MO = MI->getOperand(RegOp);
        if (!MO.isReg())
          return true;
        unsigned Reg = MO.getReg();
        O << '$' << MipsInstPrinter::getRegisterName(Reg);
        return false;
      }
    }
    case 'w':
      // Print MSA registers for the 'f' constraint
      // In LLVM, the 'w' modifier doesn't need to do anything.
      // We can just call printOperand as normal.
      break;
    }
  }

  printOperand(MI, OpNum, O);
  return false;
}

bool MipsAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                           unsigned OpNum, unsigned AsmVariant,
                                           const char *ExtraCode,
                                           raw_ostream &O) {
  int Offset = 0;
  // Currently we are expecting either no ExtraCode or 'D'
  if (ExtraCode) {
    if (ExtraCode[0] == 'D')
      Offset = 4;
    else
      return true; // Unknown modifier.
  }

  const MachineOperand &MO = MI->getOperand(OpNum);
  assert(MO.isReg() && "unexpected inline asm memory operand");
  O << Offset << "($" << MipsInstPrinter::getRegisterName(MO.getReg()) << ")";

  return false;
}

void MipsAsmPrinter::printOperand(const MachineInstr *MI, int opNum,
                                  raw_ostream &O) {
  const DataLayout *DL = TM.getDataLayout();
  const MachineOperand &MO = MI->getOperand(opNum);
  bool closeP = false;

  if (MO.getTargetFlags())
    closeP = true;

  switch(MO.getTargetFlags()) {
  case MipsII::MO_GPREL:    O << "%gp_rel("; break;
  case MipsII::MO_GOT_CALL: O << "%call16("; break;
  case MipsII::MO_GOT:      O << "%got(";    break;
  case MipsII::MO_ABS_HI:   O << "%hi(";     break;
  case MipsII::MO_ABS_LO:   O << "%lo(";     break;
  case MipsII::MO_TLSGD:    O << "%tlsgd(";  break;
  case MipsII::MO_GOTTPREL: O << "%gottprel("; break;
  case MipsII::MO_TPREL_HI: O << "%tprel_hi("; break;
  case MipsII::MO_TPREL_LO: O << "%tprel_lo("; break;
  case MipsII::MO_GPOFF_HI: O << "%hi(%neg(%gp_rel("; break;
  case MipsII::MO_GPOFF_LO: O << "%lo(%neg(%gp_rel("; break;
  case MipsII::MO_GOT_DISP: O << "%got_disp("; break;
  case MipsII::MO_GOT_PAGE: O << "%got_page("; break;
  case MipsII::MO_GOT_OFST: O << "%got_ofst("; break;
  }

  switch (MO.getType()) {
    case MachineOperand::MO_Register:
      O << '$'
        << StringRef(MipsInstPrinter::getRegisterName(MO.getReg())).lower();
      break;

    case MachineOperand::MO_Immediate:
      O << MO.getImm();
      break;

    case MachineOperand::MO_MachineBasicBlock:
      O << *MO.getMBB()->getSymbol();
      return;

    case MachineOperand::MO_GlobalAddress:
      O << *getSymbol(MO.getGlobal());
      break;

    case MachineOperand::MO_BlockAddress: {
      MCSymbol *BA = GetBlockAddressSymbol(MO.getBlockAddress());
      O << BA->getName();
      break;
    }

    case MachineOperand::MO_ConstantPoolIndex:
      O << DL->getPrivateGlobalPrefix() << "CPI"
        << getFunctionNumber() << "_" << MO.getIndex();
      if (MO.getOffset())
        O << "+" << MO.getOffset();
      break;

    default:
      llvm_unreachable("<unknown operand type>");
  }

  if (closeP) O << ")";
}

void MipsAsmPrinter::printUnsignedImm(const MachineInstr *MI, int opNum,
                                      raw_ostream &O) {
  const MachineOperand &MO = MI->getOperand(opNum);
  if (MO.isImm())
    O << (unsigned short int)MO.getImm();
  else
    printOperand(MI, opNum, O);
}

void MipsAsmPrinter::printUnsignedImm8(const MachineInstr *MI, int opNum,
                                       raw_ostream &O) {
  const MachineOperand &MO = MI->getOperand(opNum);
  if (MO.isImm())
    O << (unsigned short int)(unsigned char)MO.getImm();
  else
    printOperand(MI, opNum, O);
}

void MipsAsmPrinter::
printMemOperand(const MachineInstr *MI, int opNum, raw_ostream &O) {
  // Load/Store memory operands -- imm($reg)
  // If PIC target the target is loaded as the
  // pattern lw $25,%call16($28)
  printOperand(MI, opNum+1, O);
  O << "(";
  printOperand(MI, opNum, O);
  O << ")";
}

void MipsAsmPrinter::
printMemOperandEA(const MachineInstr *MI, int opNum, raw_ostream &O) {
  // when using stack locations for not load/store instructions
  // print the same way as all normal 3 operand instructions.
  printOperand(MI, opNum, O);
  O << ", ";
  printOperand(MI, opNum+1, O);
  return;
}

void MipsAsmPrinter::
printFCCOperand(const MachineInstr *MI, int opNum, raw_ostream &O,
                const char *Modifier) {
  const MachineOperand &MO = MI->getOperand(opNum);
  O << Mips::MipsFCCToString((Mips::CondCode)MO.getImm());
}

void MipsAsmPrinter::EmitStartOfAsmFile(Module &M) {
  bool IsABICalls = Subtarget->isABICalls();
  if (IsABICalls) {
    getTargetStreamer().emitDirectiveAbiCalls();
    Reloc::Model RM = TM.getRelocationModel();
    // FIXME: This condition should be a lot more complicated that it is here.
    //        Ideally it should test for properties of the ABI and not the ABI
    //        itself.
    //        For the moment, I'm only correcting enough to make MIPS-IV work.
    if (RM == Reloc::Static && !Subtarget->isABI_N64())
      getTargetStreamer().emitDirectiveOptionPic0();
  }

  // Tell the assembler which ABI we are using
  std::string SectionName = std::string(".mdebug.") + getCurrentABIString();
  OutStreamer.SwitchSection(OutContext.getELFSection(
      SectionName, ELF::SHT_PROGBITS, 0, SectionKind::getDataRel()));

  // NaN: At the moment we only support:
  // 1. .nan legacy (default)
  // 2. .nan 2008
  Subtarget->isNaN2008() ? getTargetStreamer().emitDirectiveNaN2008()
    : getTargetStreamer().emitDirectiveNaNLegacy();

  // TODO: handle O64 ABI

  if (Subtarget->isABI_EABI()) {
    if (Subtarget->isGP32bit())
      OutStreamer.SwitchSection(
          OutContext.getELFSection(".gcc_compiled_long32", ELF::SHT_PROGBITS, 0,
                                   SectionKind::getDataRel()));
    else
      OutStreamer.SwitchSection(
          OutContext.getELFSection(".gcc_compiled_long64", ELF::SHT_PROGBITS, 0,
                                   SectionKind::getDataRel()));
  }

  getTargetStreamer().updateABIInfo(*Subtarget);

  // We should always emit a '.module fp=...' but binutils 2.24 does not accept
  // it. We therefore emit it when it contradicts the ABI defaults (-mfpxx or
  // -mfp64) and omit it otherwise.
  if (Subtarget->isABI_O32() && (Subtarget->isABI_FPXX() ||
                                 Subtarget->isFP64bit()))
    getTargetStreamer().emitDirectiveModuleFP();

  // We should always emit a '.module [no]oddspreg' but binutils 2.24 does not
  // accept it. We therefore emit it when it contradicts the default or an
  // option has changed the default (i.e. FPXX) and omit it otherwise.
  if (Subtarget->isABI_O32() && (!Subtarget->useOddSPReg() ||
                                 Subtarget->isABI_FPXX()))
    getTargetStreamer().emitDirectiveModuleOddSPReg(Subtarget->useOddSPReg(),
                                                    Subtarget->isABI_O32());
}

void MipsAsmPrinter::emitInlineAsmStart(
    const MCSubtargetInfo &StartInfo) const {
  MipsTargetStreamer &TS = getTargetStreamer();

  // GCC's choice of assembler options for inline assembly code ('at', 'macro'
  // and 'reorder') is different from LLVM's choice for generated code ('noat',
  // 'nomacro' and 'noreorder').
  // In order to maintain compatibility with inline assembly code which depends
  // on GCC's assembler options being used, we have to switch to those options
  // for the duration of the inline assembly block and then switch back.
  TS.emitDirectiveSetPush();
  TS.emitDirectiveSetAt();
  TS.emitDirectiveSetMacro();
  TS.emitDirectiveSetReorder();
  OutStreamer.AddBlankLine();
}

void MipsAsmPrinter::emitInlineAsmEnd(const MCSubtargetInfo &StartInfo,
                                      const MCSubtargetInfo *EndInfo) const {
  OutStreamer.AddBlankLine();
  getTargetStreamer().emitDirectiveSetPop();
}

void MipsAsmPrinter::EmitJal(MCSymbol *Symbol) {
  MCInst I;
  I.setOpcode(Mips::JAL);
  I.addOperand(
      MCOperand::CreateExpr(MCSymbolRefExpr::Create(Symbol, OutContext)));
  OutStreamer.EmitInstruction(I, getSubtargetInfo());
}

void MipsAsmPrinter::EmitInstrReg(unsigned Opcode, unsigned Reg) {
  MCInst I;
  I.setOpcode(Opcode);
  I.addOperand(MCOperand::CreateReg(Reg));
  OutStreamer.EmitInstruction(I, getSubtargetInfo());
}

void MipsAsmPrinter::EmitInstrRegReg(unsigned Opcode, unsigned Reg1,
                                     unsigned Reg2) {
  MCInst I;
  //
  // Because of the current td files for Mips32, the operands for MTC1
  // appear backwards from their normal assembly order. It's not a trivial
  // change to fix this in the td file so we adjust for it here.
  //
  if (Opcode == Mips::MTC1) {
    unsigned Temp = Reg1;
    Reg1 = Reg2;
    Reg2 = Temp;
  }
  I.setOpcode(Opcode);
  I.addOperand(MCOperand::CreateReg(Reg1));
  I.addOperand(MCOperand::CreateReg(Reg2));
  OutStreamer.EmitInstruction(I, getSubtargetInfo());
}

void MipsAsmPrinter::EmitInstrRegRegReg(unsigned Opcode, unsigned Reg1,
                                        unsigned Reg2, unsigned Reg3) {
  MCInst I;
  I.setOpcode(Opcode);
  I.addOperand(MCOperand::CreateReg(Reg1));
  I.addOperand(MCOperand::CreateReg(Reg2));
  I.addOperand(MCOperand::CreateReg(Reg3));
  OutStreamer.EmitInstruction(I, getSubtargetInfo());
}

void MipsAsmPrinter::EmitMovFPIntPair(unsigned MovOpc, unsigned Reg1,
                                      unsigned Reg2, unsigned FPReg1,
                                      unsigned FPReg2, bool LE) {
  if (!LE) {
    unsigned temp = Reg1;
    Reg1 = Reg2;
    Reg2 = temp;
  }
  EmitInstrRegReg(MovOpc, Reg1, FPReg1);
  EmitInstrRegReg(MovOpc, Reg2, FPReg2);
}

void MipsAsmPrinter::EmitSwapFPIntParams(Mips16HardFloatInfo::FPParamVariant PV,
                                         bool LE, bool ToFP) {
  using namespace Mips16HardFloatInfo;
  unsigned MovOpc = ToFP ? Mips::MTC1 : Mips::MFC1;
  switch (PV) {
  case FSig:
    EmitInstrRegReg(MovOpc, Mips::A0, Mips::F12);
    break;
  case FFSig:
    EmitMovFPIntPair(MovOpc, Mips::A0, Mips::A1, Mips::F12, Mips::F14, LE);
    break;
  case FDSig:
    EmitInstrRegReg(MovOpc, Mips::A0, Mips::F12);
    EmitMovFPIntPair(MovOpc, Mips::A2, Mips::A3, Mips::F14, Mips::F15, LE);
    break;
  case DSig:
    EmitMovFPIntPair(MovOpc, Mips::A0, Mips::A1, Mips::F12, Mips::F13, LE);
    break;
  case DDSig:
    EmitMovFPIntPair(MovOpc, Mips::A0, Mips::A1, Mips::F12, Mips::F13, LE);
    EmitMovFPIntPair(MovOpc, Mips::A2, Mips::A3, Mips::F14, Mips::F15, LE);
    break;
  case DFSig:
    EmitMovFPIntPair(MovOpc, Mips::A0, Mips::A1, Mips::F12, Mips::F13, LE);
    EmitInstrRegReg(MovOpc, Mips::A2, Mips::F14);
    break;
  case NoSig:
    return;
  }
}

void
MipsAsmPrinter::EmitSwapFPIntRetval(Mips16HardFloatInfo::FPReturnVariant RV,
                                    bool LE) {
  using namespace Mips16HardFloatInfo;
  unsigned MovOpc = Mips::MFC1;
  switch (RV) {
  case FRet:
    EmitInstrRegReg(MovOpc, Mips::V0, Mips::F0);
    break;
  case DRet:
    EmitMovFPIntPair(MovOpc, Mips::V0, Mips::V1, Mips::F0, Mips::F1, LE);
    break;
  case CFRet:
    EmitMovFPIntPair(MovOpc, Mips::V0, Mips::V1, Mips::F0, Mips::F1, LE);
    break;
  case CDRet:
    EmitMovFPIntPair(MovOpc, Mips::V0, Mips::V1, Mips::F0, Mips::F1, LE);
    EmitMovFPIntPair(MovOpc, Mips::A0, Mips::A1, Mips::F2, Mips::F3, LE);
    break;
  case NoFPRet:
    break;
  }
}

void MipsAsmPrinter::EmitFPCallStub(
    const char *Symbol, const Mips16HardFloatInfo::FuncSignature *Signature) {
  MCSymbol *MSymbol = OutContext.GetOrCreateSymbol(StringRef(Symbol));
  using namespace Mips16HardFloatInfo;
  bool LE = Subtarget->isLittle();
  //
  // .global xxxx
  //
  OutStreamer.EmitSymbolAttribute(MSymbol, MCSA_Global);
  const char *RetType;
  //
  // make the comment field identifying the return and parameter
  // types of the floating point stub
  // # Stub function to call rettype xxxx (params)
  //
  switch (Signature->RetSig) {
  case FRet:
    RetType = "float";
    break;
  case DRet:
    RetType = "double";
    break;
  case CFRet:
    RetType = "complex";
    break;
  case CDRet:
    RetType = "double complex";
    break;
  case NoFPRet:
    RetType = "";
    break;
  }
  const char *Parms;
  switch (Signature->ParamSig) {
  case FSig:
    Parms = "float";
    break;
  case FFSig:
    Parms = "float, float";
    break;
  case FDSig:
    Parms = "float, double";
    break;
  case DSig:
    Parms = "double";
    break;
  case DDSig:
    Parms = "double, double";
    break;
  case DFSig:
    Parms = "double, float";
    break;
  case NoSig:
    Parms = "";
    break;
  }
  OutStreamer.AddComment("\t# Stub function to call " + Twine(RetType) + " " +
                         Twine(Symbol) + " (" + Twine(Parms) + ")");
  //
  // probably not necessary but we save and restore the current section state
  //
  OutStreamer.PushSection();
  //
  // .section mips16.call.fpxxxx,"ax",@progbits
  //
  const MCSectionELF *M = OutContext.getELFSection(
      ".mips16.call.fp." + std::string(Symbol), ELF::SHT_PROGBITS,
      ELF::SHF_ALLOC | ELF::SHF_EXECINSTR, SectionKind::getText());
  OutStreamer.SwitchSection(M, nullptr);
  //
  // .align 2
  //
  OutStreamer.EmitValueToAlignment(4);
  MipsTargetStreamer &TS = getTargetStreamer();
  //
  // .set nomips16
  // .set nomicromips
  //
  TS.emitDirectiveSetNoMips16();
  TS.emitDirectiveSetNoMicroMips();
  //
  // .ent __call_stub_fp_xxxx
  // .type  __call_stub_fp_xxxx,@function
  //  __call_stub_fp_xxxx:
  //
  std::string x = "__call_stub_fp_" + std::string(Symbol);
  MCSymbol *Stub = OutContext.GetOrCreateSymbol(StringRef(x));
  TS.emitDirectiveEnt(*Stub);
  MCSymbol *MType =
      OutContext.GetOrCreateSymbol("__call_stub_fp_" + Twine(Symbol));
  OutStreamer.EmitSymbolAttribute(MType, MCSA_ELF_TypeFunction);
  OutStreamer.EmitLabel(Stub);
  //
  // we just handle non pic for now. these function will not be
  // called otherwise. when the full stub generation is moved here
  // we need to deal with pic.
  //
  if (Subtarget->getRelocationModel() == Reloc::PIC_)
    llvm_unreachable("should not be here if we are compiling pic");
  TS.emitDirectiveSetReorder();
  //
  // We need to add a MipsMCExpr class to MCTargetDesc to fully implement
  // stubs without raw text but this current patch is for compiler generated
  // functions and they all return some value.
  // The calling sequence for non pic is different in that case and we need
  // to implement %lo and %hi in order to handle the case of no return value
  // See the corresponding method in Mips16HardFloat for details.
  //
  // mov the return address to S2.
  // we have no stack space to store it and we are about to make another call.
  // We need to make sure that the enclosing function knows to save S2
  // This should have already been handled.
  //
  // Mov $18, $31

  EmitInstrRegRegReg(Mips::ADDu, Mips::S2, Mips::RA, Mips::ZERO);

  EmitSwapFPIntParams(Signature->ParamSig, LE, true);

  // Jal xxxx
  //
  EmitJal(MSymbol);

  // fix return values
  EmitSwapFPIntRetval(Signature->RetSig, LE);
  //
  // do the return
  // if (Signature->RetSig == NoFPRet)
  //  llvm_unreachable("should not be any stubs here with no return value");
  // else
  EmitInstrReg(Mips::JR, Mips::S2);

  MCSymbol *Tmp = OutContext.CreateTempSymbol();
  OutStreamer.EmitLabel(Tmp);
  const MCSymbolRefExpr *E = MCSymbolRefExpr::Create(Stub, OutContext);
  const MCSymbolRefExpr *T = MCSymbolRefExpr::Create(Tmp, OutContext);
  const MCExpr *T_min_E = MCBinaryExpr::CreateSub(T, E, OutContext);
  OutStreamer.EmitELFSize(Stub, T_min_E);
  TS.emitDirectiveEnd(x);
  OutStreamer.PopSection();
}

void MipsAsmPrinter::EmitEndOfAsmFile(Module &M) {
  // Emit needed stubs
  //
  for (std::map<
           const char *,
           const llvm::Mips16HardFloatInfo::FuncSignature *>::const_iterator
           it = StubsNeeded.begin();
       it != StubsNeeded.end(); ++it) {
    const char *Symbol = it->first;
    const llvm::Mips16HardFloatInfo::FuncSignature *Signature = it->second;
    EmitFPCallStub(Symbol, Signature);
  }
  // return to the text section
  OutStreamer.SwitchSection(OutContext.getObjectFileInfo()->getTextSection());
}

void MipsAsmPrinter::PrintDebugValueComment(const MachineInstr *MI,
                                           raw_ostream &OS) {
  // TODO: implement
}

// Align all targets of indirect branches on bundle size.  Used only if target
// is NaCl.
void MipsAsmPrinter::NaClAlignIndirectJumpTargets(MachineFunction &MF) {
  // Align all blocks that are jumped to through jump table.
  if (MachineJumpTableInfo *JtInfo = MF.getJumpTableInfo()) {
    const std::vector<MachineJumpTableEntry> &JT = JtInfo->getJumpTables();
    for (unsigned I = 0; I < JT.size(); ++I) {
      const std::vector<MachineBasicBlock*> &MBBs = JT[I].MBBs;

      for (unsigned J = 0; J < MBBs.size(); ++J)
        MBBs[J]->setAlignment(MIPS_NACL_BUNDLE_ALIGN);
    }
  }

  // If basic block address is taken, block can be target of indirect branch.
  for (MachineFunction::iterator MBB = MF.begin(), E = MF.end();
                                 MBB != E; ++MBB) {
    if (MBB->hasAddressTaken())
      MBB->setAlignment(MIPS_NACL_BUNDLE_ALIGN);
  }
}

bool MipsAsmPrinter::isLongBranchPseudo(int Opcode) const {
  return (Opcode == Mips::LONG_BRANCH_LUi
          || Opcode == Mips::LONG_BRANCH_ADDiu
          || Opcode == Mips::LONG_BRANCH_DADDiu);
}

// Force static initialization.
extern "C" void LLVMInitializeMipsAsmPrinter() {
  RegisterAsmPrinter<MipsAsmPrinter> X(TheMipsTarget);
  RegisterAsmPrinter<MipsAsmPrinter> Y(TheMipselTarget);
  RegisterAsmPrinter<MipsAsmPrinter> A(TheMips64Target);
  RegisterAsmPrinter<MipsAsmPrinter> B(TheMips64elTarget);
}
