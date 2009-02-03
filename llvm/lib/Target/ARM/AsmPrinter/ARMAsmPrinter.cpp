//===-- ARMAsmPrinter.cpp - ARM LLVM assembly writer ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to GAS-format ARM assembly language.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asm-printer"
#include "ARM.h"
#include "ARMTargetMachine.h"
#include "ARMAddressingModes.h"
#include "ARMConstantPoolValue.h"
#include "ARMMachineFunctionInfo.h"
#include "llvm/Constants.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/DwarfWriter.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cctype>
using namespace llvm;

STATISTIC(EmittedInsts, "Number of machine instrs printed");

namespace {
  struct VISIBILITY_HIDDEN ARMAsmPrinter : public AsmPrinter {
    ARMAsmPrinter(raw_ostream &O, TargetMachine &TM, const TargetAsmInfo *T)
      : AsmPrinter(O, TM, T), DW(0), MMI(NULL), AFI(NULL), MCP(NULL),
        InCPMode(false) {
      Subtarget = &TM.getSubtarget<ARMSubtarget>();
    }

    DwarfWriter *DW;
    MachineModuleInfo *MMI;

    /// Subtarget - Keep a pointer to the ARMSubtarget around so that we can
    /// make the right decision when printing asm code for different targets.
    const ARMSubtarget *Subtarget;

    /// AFI - Keep a pointer to ARMFunctionInfo for the current
    /// MachineFunction.
    ARMFunctionInfo *AFI;

    /// MCP - Keep a pointer to constantpool entries of the current
    /// MachineFunction.
    const MachineConstantPool *MCP;

    /// We name each basic block in a Function with a unique number, so
    /// that we can consistently refer to them later. This is cleared
    /// at the beginning of each call to runOnMachineFunction().
    ///
    typedef std::map<const Value *, unsigned> ValueMapTy;
    ValueMapTy NumberForBB;

    /// GVNonLazyPtrs - Keeps the set of GlobalValues that require
    /// non-lazy-pointers for indirect access.
    StringSet<> GVNonLazyPtrs;

    /// HiddenGVNonLazyPtrs - Keeps the set of GlobalValues with hidden
    /// visibility that require non-lazy-pointers for indirect access.
    StringSet<> HiddenGVNonLazyPtrs;

    /// FnStubs - Keeps the set of external function GlobalAddresses that the
    /// asm printer should generate stubs for.
    StringSet<> FnStubs;

    /// True if asm printer is printing a series of CONSTPOOL_ENTRY.
    bool InCPMode;
    
    virtual const char *getPassName() const {
      return "ARM Assembly Printer";
    }

    void printOperand(const MachineInstr *MI, int opNum,
                      const char *Modifier = 0);
    void printSOImmOperand(const MachineInstr *MI, int opNum);
    void printSOImm2PartOperand(const MachineInstr *MI, int opNum);
    void printSORegOperand(const MachineInstr *MI, int opNum);
    void printAddrMode2Operand(const MachineInstr *MI, int OpNo);
    void printAddrMode2OffsetOperand(const MachineInstr *MI, int OpNo);
    void printAddrMode3Operand(const MachineInstr *MI, int OpNo);
    void printAddrMode3OffsetOperand(const MachineInstr *MI, int OpNo);
    void printAddrMode4Operand(const MachineInstr *MI, int OpNo,
                               const char *Modifier = 0);
    void printAddrMode5Operand(const MachineInstr *MI, int OpNo,
                               const char *Modifier = 0);
    void printAddrModePCOperand(const MachineInstr *MI, int OpNo,
                                const char *Modifier = 0);
    void printThumbAddrModeRROperand(const MachineInstr *MI, int OpNo);
    void printThumbAddrModeRI5Operand(const MachineInstr *MI, int OpNo,
                                      unsigned Scale);
    void printThumbAddrModeS1Operand(const MachineInstr *MI, int OpNo);
    void printThumbAddrModeS2Operand(const MachineInstr *MI, int OpNo);
    void printThumbAddrModeS4Operand(const MachineInstr *MI, int OpNo);
    void printThumbAddrModeSPOperand(const MachineInstr *MI, int OpNo);
    void printPredicateOperand(const MachineInstr *MI, int opNum);
    void printSBitModifierOperand(const MachineInstr *MI, int opNum);
    void printPCLabel(const MachineInstr *MI, int opNum);
    void printRegisterList(const MachineInstr *MI, int opNum);
    void printCPInstOperand(const MachineInstr *MI, int opNum,
                            const char *Modifier);
    void printJTBlockOperand(const MachineInstr *MI, int opNum);

    virtual bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                 unsigned AsmVariant, const char *ExtraCode);

    void printModuleLevelGV(const GlobalVariable* GVar);
    bool printInstruction(const MachineInstr *MI);  // autogenerated.
    void printMachineInstruction(const MachineInstr *MI);
    bool runOnMachineFunction(MachineFunction &F);
    bool doInitialization(Module &M);
    bool doFinalization(Module &M);

    /// EmitMachineConstantPoolValue - Print a machine constantpool value to
    /// the .s file.
    virtual void EmitMachineConstantPoolValue(MachineConstantPoolValue *MCPV) {
      printDataDirective(MCPV->getType());

      ARMConstantPoolValue *ACPV = static_cast<ARMConstantPoolValue*>(MCPV);
      GlobalValue *GV = ACPV->getGV();
      std::string Name = GV ? Mang->getValueName(GV) : TAI->getGlobalPrefix();
      if (!GV)
        Name += ACPV->getSymbol();
      if (ACPV->isNonLazyPointer()) {
        if (GV->hasHiddenVisibility())
          HiddenGVNonLazyPtrs.insert(Name);
        else
          GVNonLazyPtrs.insert(Name);
        printSuffixedName(Name, "$non_lazy_ptr");
      } else if (ACPV->isStub()) {
        FnStubs.insert(Name);
        printSuffixedName(Name, "$stub");
      } else
        O << Name;
      if (ACPV->hasModifier()) O << "(" << ACPV->getModifier() << ")";
      if (ACPV->getPCAdjustment() != 0) {
        O << "-(" << TAI->getPrivateGlobalPrefix() << "PC"
          << utostr(ACPV->getLabelId())
          << "+" << (unsigned)ACPV->getPCAdjustment();
         if (ACPV->mustAddCurrentAddress())
           O << "-.";
         O << ")";
      }
      O << "\n";

      // If the constant pool value is a extern weak symbol, remember to emit
      // the weak reference.
      if (GV && GV->hasExternalWeakLinkage())
        ExtWeakSymbols.insert(GV);
    }
    
    void getAnalysisUsage(AnalysisUsage &AU) const {
      AsmPrinter::getAnalysisUsage(AU);
      AU.setPreservesAll();
      AU.addRequired<MachineModuleInfo>();
      AU.addRequired<DwarfWriter>();
    }
  };
} // end of anonymous namespace

#include "ARMGenAsmWriter.inc"

/// runOnMachineFunction - This uses the printInstruction()
/// method to print assembly for each instruction.
///
bool ARMAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  AFI = MF.getInfo<ARMFunctionInfo>();
  MCP = MF.getConstantPool();

  SetupMachineFunction(MF);
  O << "\n";

  // NOTE: we don't print out constant pools here, they are handled as
  // instructions.

  O << "\n";
  // Print out labels for the function.
  const Function *F = MF.getFunction();
  switch (F->getLinkage()) {
  default: assert(0 && "Unknown linkage type!");
  case Function::PrivateLinkage:
  case Function::InternalLinkage:
    SwitchToTextSection("\t.text", F);
    break;
  case Function::ExternalLinkage:
    SwitchToTextSection("\t.text", F);
    O << "\t.globl\t" << CurrentFnName << "\n";
    break;
  case Function::WeakLinkage:
  case Function::LinkOnceLinkage:
    if (Subtarget->isTargetDarwin()) {
      SwitchToTextSection(
                ".section __TEXT,__textcoal_nt,coalesced,pure_instructions", F);
      O << "\t.globl\t" << CurrentFnName << "\n";
      O << "\t.weak_definition\t" << CurrentFnName << "\n";
    } else {
      O << TAI->getWeakRefDirective() << CurrentFnName << "\n";
    }
    break;
  }

  printVisibility(CurrentFnName, F->getVisibility());

  if (AFI->isThumbFunction()) {
    EmitAlignment(1, F, AFI->getAlign());
    O << "\t.code\t16\n";
    O << "\t.thumb_func";
    if (Subtarget->isTargetDarwin())
      O << "\t" << CurrentFnName;
    O << "\n";
    InCPMode = false;
  } else
    EmitAlignment(2, F);

  O << CurrentFnName << ":\n";
  // Emit pre-function debug information.
  DW->BeginFunction(&MF);

  if (Subtarget->isTargetDarwin()) {
    // If the function is empty, then we need to emit *something*. Otherwise,
    // the function's label might be associated with something that it wasn't
    // meant to be associated with. We emit a noop in this situation.
    MachineFunction::iterator I = MF.begin();

    if (++I == MF.end() && MF.front().empty())
      O << "\tnop\n";
  }

  // Print out code for the function.
  for (MachineFunction::const_iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {
    // Print a label for the basic block.
    if (I != MF.begin()) {
      printBasicBlockLabel(I, true, true);
      O << '\n';
    }
    for (MachineBasicBlock::const_iterator II = I->begin(), E = I->end();
         II != E; ++II) {
      // Print the assembly for the instruction.
      printMachineInstruction(II);
    }
  }

  if (TAI->hasDotTypeDotSizeDirective())
    O << "\t.size " << CurrentFnName << ", .-" << CurrentFnName << "\n";

  // Emit post-function debug information.
  DW->EndFunction(&MF);

  O.flush();

  return false;
}

void ARMAsmPrinter::printOperand(const MachineInstr *MI, int opNum,
                                 const char *Modifier) {
  const MachineOperand &MO = MI->getOperand(opNum);
  switch (MO.getType()) {
  case MachineOperand::MO_Register:
    if (TargetRegisterInfo::isPhysicalRegister(MO.getReg()))
      O << TM.getRegisterInfo()->get(MO.getReg()).AsmName;
    else
      assert(0 && "not implemented");
    break;
  case MachineOperand::MO_Immediate: {
    if (!Modifier || strcmp(Modifier, "no_hash") != 0)
      O << "#";

    O << (int)MO.getImm();
    break;
  }
  case MachineOperand::MO_MachineBasicBlock:
    printBasicBlockLabel(MO.getMBB());
    return;
  case MachineOperand::MO_GlobalAddress: {
    bool isCallOp = Modifier && !strcmp(Modifier, "call");
    GlobalValue *GV = MO.getGlobal();
    std::string Name = Mang->getValueName(GV);
    bool isExt = (GV->isDeclaration() || GV->hasWeakLinkage() ||
                  GV->hasLinkOnceLinkage());
    if (isExt && isCallOp && Subtarget->isTargetDarwin() &&
        TM.getRelocationModel() != Reloc::Static) {
      printSuffixedName(Name, "$stub");
      FnStubs.insert(Name);
    } else
      O << Name;

    printOffset(MO.getOffset());

    if (isCallOp && Subtarget->isTargetELF() &&
        TM.getRelocationModel() == Reloc::PIC_)
      O << "(PLT)";
    if (GV->hasExternalWeakLinkage())
      ExtWeakSymbols.insert(GV);
    break;
  }
  case MachineOperand::MO_ExternalSymbol: {
    bool isCallOp = Modifier && !strcmp(Modifier, "call");
    std::string Name(TAI->getGlobalPrefix());
    Name += MO.getSymbolName();
    if (isCallOp && Subtarget->isTargetDarwin() &&
        TM.getRelocationModel() != Reloc::Static) {
      printSuffixedName(Name, "$stub");
      FnStubs.insert(Name);
    } else
      O << Name;
    if (isCallOp && Subtarget->isTargetELF() &&
        TM.getRelocationModel() == Reloc::PIC_)
      O << "(PLT)";
    break;
  }
  case MachineOperand::MO_ConstantPoolIndex:
    O << TAI->getPrivateGlobalPrefix() << "CPI" << getFunctionNumber()
      << '_' << MO.getIndex();
    break;
  case MachineOperand::MO_JumpTableIndex:
    O << TAI->getPrivateGlobalPrefix() << "JTI" << getFunctionNumber()
      << '_' << MO.getIndex();
    break;
  default:
    O << "<unknown operand type>"; abort (); break;
  }
}

static void printSOImm(raw_ostream &O, int64_t V, const TargetAsmInfo *TAI) {
  assert(V < (1 << 12) && "Not a valid so_imm value!");
  unsigned Imm = ARM_AM::getSOImmValImm(V);
  unsigned Rot = ARM_AM::getSOImmValRot(V);

  // Print low-level immediate formation info, per
  // A5.1.3: "Data-processing operands - Immediate".
  if (Rot) {
    O << "#" << Imm << ", " << Rot;
    // Pretty printed version.
    O << ' ' << TAI->getCommentString() << ' ' << (int)ARM_AM::rotr32(Imm, Rot);
  } else {
    O << "#" << Imm;
  }
}

/// printSOImmOperand - SOImm is 4-bit rotate amount in bits 8-11 with 8-bit
/// immediate in bits 0-7.
void ARMAsmPrinter::printSOImmOperand(const MachineInstr *MI, int OpNum) {
  const MachineOperand &MO = MI->getOperand(OpNum);
  assert(MO.isImm() && "Not a valid so_imm value!");
  printSOImm(O, MO.getImm(), TAI);
}

/// printSOImm2PartOperand - SOImm is broken into two pieces using a 'mov'
/// followed by an 'orr' to materialize.
void ARMAsmPrinter::printSOImm2PartOperand(const MachineInstr *MI, int OpNum) {
  const MachineOperand &MO = MI->getOperand(OpNum);
  assert(MO.isImm() && "Not a valid so_imm value!");
  unsigned V1 = ARM_AM::getSOImmTwoPartFirst(MO.getImm());
  unsigned V2 = ARM_AM::getSOImmTwoPartSecond(MO.getImm());
  printSOImm(O, ARM_AM::getSOImmVal(V1), TAI);
  O << "\n\torr";
  printPredicateOperand(MI, 2);
  O << " ";
  printOperand(MI, 0); 
  O << ", ";
  printOperand(MI, 0); 
  O << ", ";
  printSOImm(O, ARM_AM::getSOImmVal(V2), TAI);
}

// so_reg is a 4-operand unit corresponding to register forms of the A5.1
// "Addressing Mode 1 - Data-processing operands" forms.  This includes:
//    REG 0   0    - e.g. R5
//    REG REG 0,SH_OPC     - e.g. R5, ROR R3
//    REG 0   IMM,SH_OPC  - e.g. R5, LSL #3
void ARMAsmPrinter::printSORegOperand(const MachineInstr *MI, int Op) {
  const MachineOperand &MO1 = MI->getOperand(Op);
  const MachineOperand &MO2 = MI->getOperand(Op+1);
  const MachineOperand &MO3 = MI->getOperand(Op+2);

  assert(TargetRegisterInfo::isPhysicalRegister(MO1.getReg()));
  O << TM.getRegisterInfo()->get(MO1.getReg()).AsmName;

  // Print the shift opc.
  O << ", "
    << ARM_AM::getShiftOpcStr(ARM_AM::getSORegShOp(MO3.getImm()))
    << " ";

  if (MO2.getReg()) {
    assert(TargetRegisterInfo::isPhysicalRegister(MO2.getReg()));
    O << TM.getRegisterInfo()->get(MO2.getReg()).AsmName;
    assert(ARM_AM::getSORegOffset(MO3.getImm()) == 0);
  } else {
    O << "#" << ARM_AM::getSORegOffset(MO3.getImm());
  }
}

void ARMAsmPrinter::printAddrMode2Operand(const MachineInstr *MI, int Op) {
  const MachineOperand &MO1 = MI->getOperand(Op);
  const MachineOperand &MO2 = MI->getOperand(Op+1);
  const MachineOperand &MO3 = MI->getOperand(Op+2);

  if (!MO1.isReg()) {   // FIXME: This is for CP entries, but isn't right.
    printOperand(MI, Op);
    return;
  }

  O << "[" << TM.getRegisterInfo()->get(MO1.getReg()).AsmName;

  if (!MO2.getReg()) {
    if (ARM_AM::getAM2Offset(MO3.getImm()))  // Don't print +0.
      O << ", #"
        << (char)ARM_AM::getAM2Op(MO3.getImm())
        << ARM_AM::getAM2Offset(MO3.getImm());
    O << "]";
    return;
  }

  O << ", "
    << (char)ARM_AM::getAM2Op(MO3.getImm())
    << TM.getRegisterInfo()->get(MO2.getReg()).AsmName;
  
  if (unsigned ShImm = ARM_AM::getAM2Offset(MO3.getImm()))
    O << ", "
      << ARM_AM::getShiftOpcStr(ARM_AM::getAM2ShiftOpc(MO3.getImm()))
      << " #" << ShImm;
  O << "]";
}

void ARMAsmPrinter::printAddrMode2OffsetOperand(const MachineInstr *MI, int Op){
  const MachineOperand &MO1 = MI->getOperand(Op);
  const MachineOperand &MO2 = MI->getOperand(Op+1);

  if (!MO1.getReg()) {
    unsigned ImmOffs = ARM_AM::getAM2Offset(MO2.getImm());
    assert(ImmOffs && "Malformed indexed load / store!");
    O << "#"
      << (char)ARM_AM::getAM2Op(MO2.getImm())
      << ImmOffs;
    return;
  }

  O << (char)ARM_AM::getAM2Op(MO2.getImm())
    << TM.getRegisterInfo()->get(MO1.getReg()).AsmName;
  
  if (unsigned ShImm = ARM_AM::getAM2Offset(MO2.getImm()))
    O << ", "
      << ARM_AM::getShiftOpcStr(ARM_AM::getAM2ShiftOpc(MO2.getImm()))
      << " #" << ShImm;
}

void ARMAsmPrinter::printAddrMode3Operand(const MachineInstr *MI, int Op) {
  const MachineOperand &MO1 = MI->getOperand(Op);
  const MachineOperand &MO2 = MI->getOperand(Op+1);
  const MachineOperand &MO3 = MI->getOperand(Op+2);
  
  assert(TargetRegisterInfo::isPhysicalRegister(MO1.getReg()));
  O << "[" << TM.getRegisterInfo()->get(MO1.getReg()).AsmName;

  if (MO2.getReg()) {
    O << ", "
      << (char)ARM_AM::getAM3Op(MO3.getImm())
      << TM.getRegisterInfo()->get(MO2.getReg()).AsmName
      << "]";
    return;
  }
  
  if (unsigned ImmOffs = ARM_AM::getAM3Offset(MO3.getImm()))
    O << ", #"
      << (char)ARM_AM::getAM3Op(MO3.getImm())
      << ImmOffs;
  O << "]";
}

void ARMAsmPrinter::printAddrMode3OffsetOperand(const MachineInstr *MI, int Op){
  const MachineOperand &MO1 = MI->getOperand(Op);
  const MachineOperand &MO2 = MI->getOperand(Op+1);

  if (MO1.getReg()) {
    O << (char)ARM_AM::getAM3Op(MO2.getImm())
      << TM.getRegisterInfo()->get(MO1.getReg()).AsmName;
    return;
  }

  unsigned ImmOffs = ARM_AM::getAM3Offset(MO2.getImm());
  assert(ImmOffs && "Malformed indexed load / store!");
  O << "#"
    << (char)ARM_AM::getAM3Op(MO2.getImm())
    << ImmOffs;
}
  
void ARMAsmPrinter::printAddrMode4Operand(const MachineInstr *MI, int Op,
                                          const char *Modifier) {
  const MachineOperand &MO1 = MI->getOperand(Op);
  const MachineOperand &MO2 = MI->getOperand(Op+1);
  ARM_AM::AMSubMode Mode = ARM_AM::getAM4SubMode(MO2.getImm());
  if (Modifier && strcmp(Modifier, "submode") == 0) {
    if (MO1.getReg() == ARM::SP) {
      bool isLDM = (MI->getOpcode() == ARM::LDM ||
                    MI->getOpcode() == ARM::LDM_RET);
      O << ARM_AM::getAMSubModeAltStr(Mode, isLDM);
    } else
      O << ARM_AM::getAMSubModeStr(Mode);
  } else {
    printOperand(MI, Op);
    if (ARM_AM::getAM4WBFlag(MO2.getImm()))
      O << "!";
  }
}

void ARMAsmPrinter::printAddrMode5Operand(const MachineInstr *MI, int Op,
                                          const char *Modifier) {
  const MachineOperand &MO1 = MI->getOperand(Op);
  const MachineOperand &MO2 = MI->getOperand(Op+1);

  if (!MO1.isReg()) {   // FIXME: This is for CP entries, but isn't right.
    printOperand(MI, Op);
    return;
  }
  
  assert(TargetRegisterInfo::isPhysicalRegister(MO1.getReg()));

  if (Modifier && strcmp(Modifier, "submode") == 0) {
    ARM_AM::AMSubMode Mode = ARM_AM::getAM5SubMode(MO2.getImm());
    if (MO1.getReg() == ARM::SP) {
      bool isFLDM = (MI->getOpcode() == ARM::FLDMD ||
                     MI->getOpcode() == ARM::FLDMS);
      O << ARM_AM::getAMSubModeAltStr(Mode, isFLDM);
    } else
      O << ARM_AM::getAMSubModeStr(Mode);
    return;
  } else if (Modifier && strcmp(Modifier, "base") == 0) {
    // Used for FSTM{D|S} and LSTM{D|S} operations.
    O << TM.getRegisterInfo()->get(MO1.getReg()).AsmName;
    if (ARM_AM::getAM5WBFlag(MO2.getImm()))
      O << "!";
    return;
  }
  
  O << "[" << TM.getRegisterInfo()->get(MO1.getReg()).AsmName;
  
  if (unsigned ImmOffs = ARM_AM::getAM5Offset(MO2.getImm())) {
    O << ", #"
      << (char)ARM_AM::getAM5Op(MO2.getImm())
      << ImmOffs*4;
  }
  O << "]";
}

void ARMAsmPrinter::printAddrModePCOperand(const MachineInstr *MI, int Op,
                                           const char *Modifier) {
  if (Modifier && strcmp(Modifier, "label") == 0) {
    printPCLabel(MI, Op+1);
    return;
  }

  const MachineOperand &MO1 = MI->getOperand(Op);
  assert(TargetRegisterInfo::isPhysicalRegister(MO1.getReg()));
  O << "[pc, +" << TM.getRegisterInfo()->get(MO1.getReg()).AsmName << "]";
}

void
ARMAsmPrinter::printThumbAddrModeRROperand(const MachineInstr *MI, int Op) {
  const MachineOperand &MO1 = MI->getOperand(Op);
  const MachineOperand &MO2 = MI->getOperand(Op+1);
  O << "[" << TM.getRegisterInfo()->get(MO1.getReg()).AsmName;
  O << ", " << TM.getRegisterInfo()->get(MO2.getReg()).AsmName << "]";
}

void
ARMAsmPrinter::printThumbAddrModeRI5Operand(const MachineInstr *MI, int Op,
                                            unsigned Scale) {
  const MachineOperand &MO1 = MI->getOperand(Op);
  const MachineOperand &MO2 = MI->getOperand(Op+1);
  const MachineOperand &MO3 = MI->getOperand(Op+2);

  if (!MO1.isReg()) {   // FIXME: This is for CP entries, but isn't right.
    printOperand(MI, Op);
    return;
  }

  O << "[" << TM.getRegisterInfo()->get(MO1.getReg()).AsmName;
  if (MO3.getReg())
    O << ", " << TM.getRegisterInfo()->get(MO3.getReg()).AsmName;
  else if (unsigned ImmOffs = MO2.getImm()) {
    O << ", #" << ImmOffs;
    if (Scale > 1)
      O << " * " << Scale;
  }
  O << "]";
}

void
ARMAsmPrinter::printThumbAddrModeS1Operand(const MachineInstr *MI, int Op) {
  printThumbAddrModeRI5Operand(MI, Op, 1);
}
void
ARMAsmPrinter::printThumbAddrModeS2Operand(const MachineInstr *MI, int Op) {
  printThumbAddrModeRI5Operand(MI, Op, 2);
}
void
ARMAsmPrinter::printThumbAddrModeS4Operand(const MachineInstr *MI, int Op) {
  printThumbAddrModeRI5Operand(MI, Op, 4);
}

void ARMAsmPrinter::printThumbAddrModeSPOperand(const MachineInstr *MI,int Op) {
  const MachineOperand &MO1 = MI->getOperand(Op);
  const MachineOperand &MO2 = MI->getOperand(Op+1);
  O << "[" << TM.getRegisterInfo()->get(MO1.getReg()).AsmName;
  if (unsigned ImmOffs = MO2.getImm())
    O << ", #" << ImmOffs << " * 4";
  O << "]";
}

void ARMAsmPrinter::printPredicateOperand(const MachineInstr *MI, int opNum) {
  ARMCC::CondCodes CC = (ARMCC::CondCodes)MI->getOperand(opNum).getImm();
  if (CC != ARMCC::AL)
    O << ARMCondCodeToString(CC);
}

void ARMAsmPrinter::printSBitModifierOperand(const MachineInstr *MI, int opNum){
  unsigned Reg = MI->getOperand(opNum).getReg();
  if (Reg) {
    assert(Reg == ARM::CPSR && "Expect ARM CPSR register!");
    O << 's';
  }
}

void ARMAsmPrinter::printPCLabel(const MachineInstr *MI, int opNum) {
  int Id = (int)MI->getOperand(opNum).getImm();
  O << TAI->getPrivateGlobalPrefix() << "PC" << Id;
}

void ARMAsmPrinter::printRegisterList(const MachineInstr *MI, int opNum) {
  O << "{";
  for (unsigned i = opNum, e = MI->getNumOperands(); i != e; ++i) {
    printOperand(MI, i);
    if (i != e-1) O << ", ";
  }
  O << "}";
}

void ARMAsmPrinter::printCPInstOperand(const MachineInstr *MI, int OpNo,
                                       const char *Modifier) {
  assert(Modifier && "This operand only works with a modifier!");
  // There are two aspects to a CONSTANTPOOL_ENTRY operand, the label and the
  // data itself.
  if (!strcmp(Modifier, "label")) {
    unsigned ID = MI->getOperand(OpNo).getImm();
    O << TAI->getPrivateGlobalPrefix() << "CPI" << getFunctionNumber()
      << '_' << ID << ":\n";
  } else {
    assert(!strcmp(Modifier, "cpentry") && "Unknown modifier for CPE");
    unsigned CPI = MI->getOperand(OpNo).getIndex();

    const MachineConstantPoolEntry &MCPE = MCP->getConstants()[CPI];
    
    if (MCPE.isMachineConstantPoolEntry()) {
      EmitMachineConstantPoolValue(MCPE.Val.MachineCPVal);
    } else {
      EmitGlobalConstant(MCPE.Val.ConstVal);
      // remember to emit the weak reference
      if (const GlobalValue *GV = dyn_cast<GlobalValue>(MCPE.Val.ConstVal))
        if (GV->hasExternalWeakLinkage())
          ExtWeakSymbols.insert(GV);
    }
  }
}

void ARMAsmPrinter::printJTBlockOperand(const MachineInstr *MI, int OpNo) {
  const MachineOperand &MO1 = MI->getOperand(OpNo);
  const MachineOperand &MO2 = MI->getOperand(OpNo+1); // Unique Id
  unsigned JTI = MO1.getIndex();
  O << TAI->getPrivateGlobalPrefix() << "JTI" << getFunctionNumber()
    << '_' << JTI << '_' << MO2.getImm() << ":\n";

  const char *JTEntryDirective = TAI->getJumpTableDirective();
  if (!JTEntryDirective)
    JTEntryDirective = TAI->getData32bitsDirective();

  const MachineFunction *MF = MI->getParent()->getParent();
  const MachineJumpTableInfo *MJTI = MF->getJumpTableInfo();
  const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
  const std::vector<MachineBasicBlock*> &JTBBs = JT[JTI].MBBs;
  bool UseSet= TAI->getSetDirective() && TM.getRelocationModel() == Reloc::PIC_;
  std::set<MachineBasicBlock*> JTSets;
  for (unsigned i = 0, e = JTBBs.size(); i != e; ++i) {
    MachineBasicBlock *MBB = JTBBs[i];
    if (UseSet && JTSets.insert(MBB).second)
      printPICJumpTableSetLabel(JTI, MO2.getImm(), MBB);

    O << JTEntryDirective << ' ';
    if (UseSet)
      O << TAI->getPrivateGlobalPrefix() << getFunctionNumber()
        << '_' << JTI << '_' << MO2.getImm()
        << "_set_" << MBB->getNumber();
    else if (TM.getRelocationModel() == Reloc::PIC_) {
      printBasicBlockLabel(MBB, false, false, false);
      // If the arch uses custom Jump Table directives, don't calc relative to JT
      if (!TAI->getJumpTableDirective()) 
        O << '-' << TAI->getPrivateGlobalPrefix() << "JTI"
          << getFunctionNumber() << '_' << JTI << '_' << MO2.getImm();
    } else
      printBasicBlockLabel(MBB, false, false, false);
    if (i != e-1)
      O << '\n';
  }
}


bool ARMAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                    unsigned AsmVariant, const char *ExtraCode){
  // Does this asm operand have a single letter operand modifier?
  if (ExtraCode && ExtraCode[0]) {
    if (ExtraCode[1] != 0) return true; // Unknown modifier.
    
    switch (ExtraCode[0]) {
    default: return true;  // Unknown modifier.
    case 'c': // Don't print "$" before a global var name or constant.
    case 'P': // Print a VFP double precision register.
      printOperand(MI, OpNo);
      return false;
    case 'Q':
      if (TM.getTargetData()->isLittleEndian())
        break;
      // Fallthrough
    case 'R':
      if (TM.getTargetData()->isBigEndian())
        break;
      // Fallthrough
    case 'H': // Write second word of DI / DF reference.  
      // Verify that this operand has two consecutive registers.
      if (!MI->getOperand(OpNo).isReg() ||
          OpNo+1 == MI->getNumOperands() ||
          !MI->getOperand(OpNo+1).isReg())
        return true;
      ++OpNo;   // Return the high-part.
    }
  }
  
  printOperand(MI, OpNo);
  return false;
}

void ARMAsmPrinter::printMachineInstruction(const MachineInstr *MI) {
  ++EmittedInsts;

  int Opc = MI->getOpcode();
  switch (Opc) {
  case ARM::CONSTPOOL_ENTRY:
    if (!InCPMode && AFI->isThumbFunction()) {
      EmitAlignment(2);
      InCPMode = true;
    }
    break;
  default: {
    if (InCPMode && AFI->isThumbFunction())
      InCPMode = false;
  }}

  // Call the autogenerated instruction printer routines.
  printInstruction(MI);
}

bool ARMAsmPrinter::doInitialization(Module &M) {

  bool Result = AsmPrinter::doInitialization(M);

  // Emit initial debug information.
  MMI = getAnalysisIfAvailable<MachineModuleInfo>();
  assert(MMI);
  DW = getAnalysisIfAvailable<DwarfWriter>();
  assert(DW && "Dwarf Writer is not available");
  DW->BeginModule(&M, MMI, O, this, TAI);

  // Darwin wants symbols to be quoted if they have complex names.
  if (Subtarget->isTargetDarwin())
    Mang->setUseQuotes(true);

  return Result;
}

/// PrintUnmangledNameSafely - Print out the printable characters in the name.
/// Don't print things like \n or \0.
static void PrintUnmangledNameSafely(const Value *V, raw_ostream &OS) {
  for (const char *Name = V->getNameStart(), *E = Name+V->getNameLen();
       Name != E; ++Name)
    if (isprint(*Name))
      OS << *Name;
}

void ARMAsmPrinter::printModuleLevelGV(const GlobalVariable* GVar) {
  const TargetData *TD = TM.getTargetData();

  if (!GVar->hasInitializer())   // External global require no code
    return;

  // Check to see if this is a special global used by LLVM, if so, emit it.

  if (EmitSpecialLLVMGlobal(GVar)) {
    if (Subtarget->isTargetDarwin() &&
        TM.getRelocationModel() == Reloc::Static) {
      if (GVar->getName() == "llvm.global_ctors")
        O << ".reference .constructors_used\n";
      else if (GVar->getName() == "llvm.global_dtors")
        O << ".reference .destructors_used\n";
    }
    return;
  }

  std::string name = Mang->getValueName(GVar);
  Constant *C = GVar->getInitializer();
  const Type *Type = C->getType();
  unsigned Size = TD->getTypePaddedSize(Type);
  unsigned Align = TD->getPreferredAlignmentLog(GVar);
  bool isDarwin = Subtarget->isTargetDarwin();

  printVisibility(name, GVar->getVisibility());

  if (Subtarget->isTargetELF())
    O << "\t.type " << name << ",%object\n";

  if (C->isNullValue() && !GVar->hasSection() && !GVar->isThreadLocal()) {
    // FIXME: This seems to be pretty darwin-specific

    if (GVar->hasExternalLinkage()) {
      SwitchToSection(TAI->SectionForGlobal(GVar));
      if (const char *Directive = TAI->getZeroFillDirective()) {
        O << "\t.globl\t" << name << "\n";
        O << Directive << "__DATA, __common, " << name << ", "
          << Size << ", " << Align << "\n";
        return;
      }
    }

    if (GVar->hasLocalLinkage() || GVar->mayBeOverridden()) {
      if (Size == 0) Size = 1;   // .comm Foo, 0 is undefined, avoid it.

      if (isDarwin) {
        if (GVar->hasLocalLinkage()) {
          O << TAI->getLCOMMDirective()  << name << "," << Size
            << ',' << Align;
        } else if (GVar->hasCommonLinkage()) {
          O << TAI->getCOMMDirective()  << name << "," << Size
            << ',' << Align;
        } else {
          SwitchToSection(TAI->SectionForGlobal(GVar));
          O << "\t.globl " << name << '\n'
            << TAI->getWeakDefDirective() << name << '\n';
          EmitAlignment(Align, GVar);
          O << name << ":\t\t\t\t" << TAI->getCommentString() << ' ';
          PrintUnmangledNameSafely(GVar, O);
          O << '\n';
          EmitGlobalConstant(C);
          return;
        }
      } else if (TAI->getLCOMMDirective() != NULL) {
        if (GVar->hasLocalLinkage()) {
          O << TAI->getLCOMMDirective() << name << "," << Size;
        } else {
          O << TAI->getCOMMDirective()  << name << "," << Size;
          if (TAI->getCOMMDirectiveTakesAlignment())
            O << ',' << (TAI->getAlignmentIsInBytes() ? (1 << Align) : Align);
        }
      } else {
        SwitchToSection(TAI->SectionForGlobal(GVar));
        if (GVar->hasLocalLinkage())
          O << "\t.local\t" << name << "\n";
        O << TAI->getCOMMDirective()  << name << "," << Size;
        if (TAI->getCOMMDirectiveTakesAlignment())
          O << "," << (TAI->getAlignmentIsInBytes() ? (1 << Align) : Align);
      }
      O << "\t\t" << TAI->getCommentString() << " ";
      PrintUnmangledNameSafely(GVar, O);
      O << "\n";
      return;
    }
  }

  SwitchToSection(TAI->SectionForGlobal(GVar));
  switch (GVar->getLinkage()) {
   case GlobalValue::CommonLinkage:
   case GlobalValue::LinkOnceLinkage:
   case GlobalValue::WeakLinkage:
    if (isDarwin) {
      O << "\t.globl " << name << "\n"
        << "\t.weak_definition " << name << "\n";
    } else {
      O << "\t.weak " << name << "\n";
    }
    break;
   case GlobalValue::AppendingLinkage:
    // FIXME: appending linkage variables should go into a section of
    // their name or something.  For now, just emit them as external.
   case GlobalValue::ExternalLinkage:
    O << "\t.globl " << name << "\n";
    // FALL THROUGH
   case GlobalValue::PrivateLinkage:
   case GlobalValue::InternalLinkage:
    break;
   default:
    assert(0 && "Unknown linkage type!");
    break;
  }

  EmitAlignment(Align, GVar);
  O << name << ":\t\t\t\t" << TAI->getCommentString() << " ";
  PrintUnmangledNameSafely(GVar, O);
  O << "\n";
  if (TAI->hasDotTypeDotSizeDirective())
    O << "\t.size " << name << ", " << Size << "\n";

  // If the initializer is a extern weak symbol, remember to emit the weak
  // reference!
  if (const GlobalValue *GV = dyn_cast<GlobalValue>(C))
    if (GV->hasExternalWeakLinkage())
      ExtWeakSymbols.insert(GV);

  EmitGlobalConstant(C);
  O << '\n';
}


bool ARMAsmPrinter::doFinalization(Module &M) {
  for (Module::const_global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I)
    printModuleLevelGV(I);

  if (Subtarget->isTargetDarwin()) {
    SwitchToDataSection("");

    // Output stubs for dynamically-linked functions
    for (StringSet<>::iterator i = FnStubs.begin(), e = FnStubs.end();
         i != e; ++i) {
      if (TM.getRelocationModel() == Reloc::PIC_)
        SwitchToTextSection(".section __TEXT,__picsymbolstub4,symbol_stubs,"
                            "none,16", 0);
      else
        SwitchToTextSection(".section __TEXT,__symbol_stub4,symbol_stubs,"
                            "none,12", 0);

      EmitAlignment(2);
      O << "\t.code\t32\n";

      const char *p = i->getKeyData();
      printSuffixedName(p, "$stub");
      O << ":\n";
      O << "\t.indirect_symbol " << p << "\n";
      O << "\tldr ip, ";
      printSuffixedName(p, "$slp");
      O << "\n";
      if (TM.getRelocationModel() == Reloc::PIC_) {
        printSuffixedName(p, "$scv");
        O << ":\n";
        O << "\tadd ip, pc, ip\n";
      }
      O << "\tldr pc, [ip, #0]\n";
      printSuffixedName(p, "$slp");
      O << ":\n";
      O << "\t.long\t";
      printSuffixedName(p, "$lazy_ptr");
      if (TM.getRelocationModel() == Reloc::PIC_) {
        O << "-(";
        printSuffixedName(p, "$scv");
        O << "+8)\n";
      } else
        O << "\n";
      SwitchToDataSection(".lazy_symbol_pointer", 0);
      printSuffixedName(p, "$lazy_ptr");
      O << ":\n";
      O << "\t.indirect_symbol " << p << "\n";
      O << "\t.long\tdyld_stub_binding_helper\n";
    }
    O << "\n";

    // Output non-lazy-pointers for external and common global variables.
    if (!GVNonLazyPtrs.empty()) {
      SwitchToDataSection("\t.non_lazy_symbol_pointer", 0);
      for (StringSet<>::iterator i =  GVNonLazyPtrs.begin(),
             e = GVNonLazyPtrs.end(); i != e; ++i) {
        const char *p = i->getKeyData();
        printSuffixedName(p, "$non_lazy_ptr");
        O << ":\n";
        O << "\t.indirect_symbol " << p << "\n";
        O << "\t.long\t0\n";
      }
    }

    if (!HiddenGVNonLazyPtrs.empty()) {
      SwitchToSection(TAI->getDataSection());
      for (StringSet<>::iterator i = HiddenGVNonLazyPtrs.begin(),
             e = HiddenGVNonLazyPtrs.end(); i != e; ++i) {
        const char *p = i->getKeyData();
        EmitAlignment(2);
        printSuffixedName(p, "$non_lazy_ptr");
        O << ":\n";
        O << "\t.long " << p << "\n";
      }
    }


    // Emit initial debug information.
    DW->EndModule();

    // Funny Darwin hack: This flag tells the linker that no global symbols
    // contain code that falls through to other global symbols (e.g. the obvious
    // implementation of multiple entry points).  If this doesn't occur, the
    // linker can safely perform dead code stripping.  Since LLVM never
    // generates code that does this, it is always safe to set.
    O << "\t.subsections_via_symbols\n";
  } else {
    // Emit final debug information for ELF.
    DW->EndModule();
  }

  return AsmPrinter::doFinalization(M);
}

/// createARMCodePrinterPass - Returns a pass that prints the ARM
/// assembly code for a MachineFunction to the given output stream,
/// using the given target machine description.  This should work
/// regardless of whether the function is in SSA form.
///
FunctionPass *llvm::createARMCodePrinterPass(raw_ostream &o,
                                             ARMTargetMachine &tm) {
  return new ARMAsmPrinter(o, tm, tm.getTargetAsmInfo());
}

namespace {
  static struct Register {
    Register() {
      ARMTargetMachine::registerAsmPrinter(createARMCodePrinterPass);
    }
  } Registrator;
}
