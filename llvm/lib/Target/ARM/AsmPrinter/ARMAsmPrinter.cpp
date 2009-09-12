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
#include "ARMBuildAttrs.h"
#include "ARMTargetMachine.h"
#include "ARMAddressingModes.h"
#include "ARMConstantPoolValue.h"
#include "ARMMachineFunctionInfo.h"
#include "llvm/Constants.h"
#include "llvm/Module.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/DwarfWriter.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/FormattedStream.h"
#include <cctype>
using namespace llvm;

STATISTIC(EmittedInsts, "Number of machine instrs printed");

namespace {
  class VISIBILITY_HIDDEN ARMAsmPrinter : public AsmPrinter {
    DwarfWriter *DW;

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
    StringMap<std::string> GVNonLazyPtrs;

    /// HiddenGVNonLazyPtrs - Keeps the set of GlobalValues with hidden
    /// visibility that require non-lazy-pointers for indirect access.
    StringMap<std::string> HiddenGVNonLazyPtrs;

    /// True if asm printer is printing a series of CONSTPOOL_ENTRY.
    bool InCPMode;
  public:
    explicit ARMAsmPrinter(formatted_raw_ostream &O, TargetMachine &TM,
                           const TargetAsmInfo *T, bool V)
      : AsmPrinter(O, TM, T, V), DW(0), AFI(NULL), MCP(NULL),
        InCPMode(false) {
      Subtarget = &TM.getSubtarget<ARMSubtarget>();
    }

    virtual const char *getPassName() const {
      return "ARM Assembly Printer";
    }

    void printOperand(const MachineInstr *MI, int OpNum,
                      const char *Modifier = 0);
    void printSOImmOperand(const MachineInstr *MI, int OpNum);
    void printSOImm2PartOperand(const MachineInstr *MI, int OpNum);
    void printSORegOperand(const MachineInstr *MI, int OpNum);
    void printAddrMode2Operand(const MachineInstr *MI, int OpNum);
    void printAddrMode2OffsetOperand(const MachineInstr *MI, int OpNum);
    void printAddrMode3Operand(const MachineInstr *MI, int OpNum);
    void printAddrMode3OffsetOperand(const MachineInstr *MI, int OpNum);
    void printAddrMode4Operand(const MachineInstr *MI, int OpNum,
                               const char *Modifier = 0);
    void printAddrMode5Operand(const MachineInstr *MI, int OpNum,
                               const char *Modifier = 0);
    void printAddrMode6Operand(const MachineInstr *MI, int OpNum);
    void printAddrModePCOperand(const MachineInstr *MI, int OpNum,
                                const char *Modifier = 0);
    void printBitfieldInvMaskImmOperand (const MachineInstr *MI, int OpNum);

    void printThumbITMask(const MachineInstr *MI, int OpNum);
    void printThumbAddrModeRROperand(const MachineInstr *MI, int OpNum);
    void printThumbAddrModeRI5Operand(const MachineInstr *MI, int OpNum,
                                      unsigned Scale);
    void printThumbAddrModeS1Operand(const MachineInstr *MI, int OpNum);
    void printThumbAddrModeS2Operand(const MachineInstr *MI, int OpNum);
    void printThumbAddrModeS4Operand(const MachineInstr *MI, int OpNum);
    void printThumbAddrModeSPOperand(const MachineInstr *MI, int OpNum);

    void printT2SOOperand(const MachineInstr *MI, int OpNum);
    void printT2AddrModeImm12Operand(const MachineInstr *MI, int OpNum);
    void printT2AddrModeImm8Operand(const MachineInstr *MI, int OpNum);
    void printT2AddrModeImm8s4Operand(const MachineInstr *MI, int OpNum);
    void printT2AddrModeImm8OffsetOperand(const MachineInstr *MI, int OpNum);
    void printT2AddrModeSoRegOperand(const MachineInstr *MI, int OpNum);

    void printPredicateOperand(const MachineInstr *MI, int OpNum);
    void printSBitModifierOperand(const MachineInstr *MI, int OpNum);
    void printPCLabel(const MachineInstr *MI, int OpNum);
    void printRegisterList(const MachineInstr *MI, int OpNum);
    void printCPInstOperand(const MachineInstr *MI, int OpNum,
                            const char *Modifier);
    void printJTBlockOperand(const MachineInstr *MI, int OpNum);
    void printJT2BlockOperand(const MachineInstr *MI, int OpNum);
    void printTBAddrMode(const MachineInstr *MI, int OpNum);
    void printNoHashImmediate(const MachineInstr *MI, int OpNum);

    virtual bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNum,
                                 unsigned AsmVariant, const char *ExtraCode);
    virtual bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNum,
                                       unsigned AsmVariant,
                                       const char *ExtraCode);

    void PrintGlobalVariable(const GlobalVariable* GVar);
    void printInstruction(const MachineInstr *MI);  // autogenerated.
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
      std::string Name;
      
      if (GV) {
        bool isIndirect = Subtarget->isTargetDarwin() &&
          Subtarget->GVIsIndirectSymbol(GV,
                                        TM.getRelocationModel() == Reloc::Static);
        if (!isIndirect)
          Name = Mang->getMangledName(GV);
        else {
          // FIXME: Remove this when Darwin transition to @GOT like syntax.
          std::string SymName = Mang->getMangledName(GV);
          Name = Mang->getMangledName(GV, "$non_lazy_ptr", true);
          if (GV->hasHiddenVisibility())
            HiddenGVNonLazyPtrs[SymName] = Name;
          else
            GVNonLazyPtrs[SymName] = Name;
        }
      } else if (!strncmp(ACPV->getSymbol(), "L_lsda_", 7))
        Name = ACPV->getSymbol();
      else
        Name = Mang->makeNameProper(ACPV->getSymbol());
      O << Name;      
      
      if (ACPV->hasModifier()) O << "(" << ACPV->getModifier() << ")";
      if (ACPV->getPCAdjustment() != 0) {
        O << "-(" << TAI->getPrivateGlobalPrefix() << "PC"
          << ACPV->getLabelId()
          << "+" << (unsigned)ACPV->getPCAdjustment();
         if (ACPV->mustAddCurrentAddress())
           O << "-.";
         O << ")";
      }
      O << "\n";
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
  this->MF = &MF;

  AFI = MF.getInfo<ARMFunctionInfo>();
  MCP = MF.getConstantPool();

  SetupMachineFunction(MF);
  O << "\n";

  // NOTE: we don't print out constant pools here, they are handled as
  // instructions.

  O << '\n';
  
  // Print out labels for the function.
  const Function *F = MF.getFunction();
  OutStreamer.SwitchSection(getObjFileLowering().SectionForGlobal(F, Mang, TM));

  switch (F->getLinkage()) {
  default: llvm_unreachable("Unknown linkage type!");
  case Function::PrivateLinkage:
  case Function::LinkerPrivateLinkage:
  case Function::InternalLinkage:
    break;
  case Function::ExternalLinkage:
    O << "\t.globl\t" << CurrentFnName << "\n";
    break;
  case Function::WeakAnyLinkage:
  case Function::WeakODRLinkage:
  case Function::LinkOnceAnyLinkage:
  case Function::LinkOnceODRLinkage:
    if (Subtarget->isTargetDarwin()) {
      O << "\t.globl\t" << CurrentFnName << "\n";
      O << "\t.weak_definition\t" << CurrentFnName << "\n";
    } else {
      O << TAI->getWeakRefDirective() << CurrentFnName << "\n";
    }
    break;
  }

  printVisibility(CurrentFnName, F->getVisibility());

  if (AFI->isThumbFunction()) {
    EmitAlignment(MF.getAlignment(), F, AFI->getAlign());
    O << "\t.code\t16\n";
    O << "\t.thumb_func";
    if (Subtarget->isTargetDarwin())
      O << "\t" << CurrentFnName;
    O << "\n";
    InCPMode = false;
  } else {
    EmitAlignment(MF.getAlignment(), F);
  }

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
      printBasicBlockLabel(I, true, true, VerboseAsm);
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

  return false;
}

void ARMAsmPrinter::printOperand(const MachineInstr *MI, int OpNum,
                                 const char *Modifier) {
  const MachineOperand &MO = MI->getOperand(OpNum);
  switch (MO.getType()) {
  case MachineOperand::MO_Register: {
    unsigned Reg = MO.getReg();
    if (TargetRegisterInfo::isPhysicalRegister(Reg)) {
      if (Modifier && strcmp(Modifier, "dregpair") == 0) {
        unsigned DRegLo = TRI->getSubReg(Reg, 5); // arm_dsubreg_0
        unsigned DRegHi = TRI->getSubReg(Reg, 6); // arm_dsubreg_1
        O << '{'
          << TRI->getAsmName(DRegLo) << ',' << TRI->getAsmName(DRegHi)
          << '}';
      } else if (Modifier && strcmp(Modifier, "lane") == 0) {
        unsigned RegNum = ARMRegisterInfo::getRegisterNumbering(Reg);
        unsigned DReg = TRI->getMatchingSuperReg(Reg, RegNum & 1 ? 0 : 1,
                                                 &ARM::DPRRegClass);
        O << TRI->getAsmName(DReg) << '[' << (RegNum & 1) << ']';
      } else {
        O << TRI->getAsmName(Reg);
      }
    } else
      llvm_unreachable("not implemented");
    break;
  }
  case MachineOperand::MO_Immediate: {
    O << '#' << MO.getImm();
    break;
  }
  case MachineOperand::MO_MachineBasicBlock:
    printBasicBlockLabel(MO.getMBB());
    return;
  case MachineOperand::MO_GlobalAddress: {
    bool isCallOp = Modifier && !strcmp(Modifier, "call");
    GlobalValue *GV = MO.getGlobal();
    O << Mang->getMangledName(GV);

    printOffset(MO.getOffset());

    if (isCallOp && Subtarget->isTargetELF() &&
        TM.getRelocationModel() == Reloc::PIC_)
      O << "(PLT)";
    break;
  }
  case MachineOperand::MO_ExternalSymbol: {
    bool isCallOp = Modifier && !strcmp(Modifier, "call");
    std::string Name = Mang->makeNameProper(MO.getSymbolName());
    
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

static void printSOImm(formatted_raw_ostream &O, int64_t V, bool VerboseAsm,
                       const TargetAsmInfo *TAI) {
  // Break it up into two parts that make up a shifter immediate.
  V = ARM_AM::getSOImmVal(V);
  assert(V != -1 && "Not a valid so_imm value!");

  unsigned Imm = ARM_AM::getSOImmValImm(V);
  unsigned Rot = ARM_AM::getSOImmValRot(V);

  // Print low-level immediate formation info, per
  // A5.1.3: "Data-processing operands - Immediate".
  if (Rot) {
    O << "#" << Imm << ", " << Rot;
    // Pretty printed version.
    if (VerboseAsm)
      O << ' ' << TAI->getCommentString()
        << ' ' << (int)ARM_AM::rotr32(Imm, Rot);
  } else {
    O << "#" << Imm;
  }
}

/// printSOImmOperand - SOImm is 4-bit rotate amount in bits 8-11 with 8-bit
/// immediate in bits 0-7.
void ARMAsmPrinter::printSOImmOperand(const MachineInstr *MI, int OpNum) {
  const MachineOperand &MO = MI->getOperand(OpNum);
  assert(MO.isImm() && "Not a valid so_imm value!");
  printSOImm(O, MO.getImm(), VerboseAsm, TAI);
}

/// printSOImm2PartOperand - SOImm is broken into two pieces using a 'mov'
/// followed by an 'orr' to materialize.
void ARMAsmPrinter::printSOImm2PartOperand(const MachineInstr *MI, int OpNum) {
  const MachineOperand &MO = MI->getOperand(OpNum);
  assert(MO.isImm() && "Not a valid so_imm value!");
  unsigned V1 = ARM_AM::getSOImmTwoPartFirst(MO.getImm());
  unsigned V2 = ARM_AM::getSOImmTwoPartSecond(MO.getImm());
  printSOImm(O, V1, VerboseAsm, TAI);
  O << "\n\torr";
  printPredicateOperand(MI, 2);
  O << " ";
  printOperand(MI, 0); 
  O << ", ";
  printOperand(MI, 0); 
  O << ", ";
  printSOImm(O, V2, VerboseAsm, TAI);
}

// so_reg is a 4-operand unit corresponding to register forms of the A5.1
// "Addressing Mode 1 - Data-processing operands" forms.  This includes:
//    REG 0   0           - e.g. R5
//    REG REG 0,SH_OPC    - e.g. R5, ROR R3
//    REG 0   IMM,SH_OPC  - e.g. R5, LSL #3
void ARMAsmPrinter::printSORegOperand(const MachineInstr *MI, int Op) {
  const MachineOperand &MO1 = MI->getOperand(Op);
  const MachineOperand &MO2 = MI->getOperand(Op+1);
  const MachineOperand &MO3 = MI->getOperand(Op+2);

  assert(TargetRegisterInfo::isPhysicalRegister(MO1.getReg()));
  O << TRI->getAsmName(MO1.getReg());

  // Print the shift opc.
  O << ", "
    << ARM_AM::getShiftOpcStr(ARM_AM::getSORegShOp(MO3.getImm()))
    << " ";

  if (MO2.getReg()) {
    assert(TargetRegisterInfo::isPhysicalRegister(MO2.getReg()));
    O << TRI->getAsmName(MO2.getReg());
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

  O << "[" << TRI->getAsmName(MO1.getReg());

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
    << TRI->getAsmName(MO2.getReg());
  
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
    << TRI->getAsmName(MO1.getReg());
  
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
  O << "[" << TRI->getAsmName(MO1.getReg());

  if (MO2.getReg()) {
    O << ", "
      << (char)ARM_AM::getAM3Op(MO3.getImm())
      << TRI->getAsmName(MO2.getReg())
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
      << TRI->getAsmName(MO1.getReg());
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
      // FIXME
      bool isLDM = (MI->getOpcode() == ARM::LDM ||
                    MI->getOpcode() == ARM::LDM_RET ||
                    MI->getOpcode() == ARM::t2LDM ||
                    MI->getOpcode() == ARM::t2LDM_RET);
      O << ARM_AM::getAMSubModeAltStr(Mode, isLDM);
    } else
      O << ARM_AM::getAMSubModeStr(Mode);
  } else if (Modifier && strcmp(Modifier, "wide") == 0) {
    ARM_AM::AMSubMode Mode = ARM_AM::getAM4SubMode(MO2.getImm());
    if (Mode == ARM_AM::ia)
      O << ".w";
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
    O << TRI->getAsmName(MO1.getReg());
    if (ARM_AM::getAM5WBFlag(MO2.getImm()))
      O << "!";
    return;
  }
  
  O << "[" << TRI->getAsmName(MO1.getReg());
  
  if (unsigned ImmOffs = ARM_AM::getAM5Offset(MO2.getImm())) {
    O << ", #"
      << (char)ARM_AM::getAM5Op(MO2.getImm())
      << ImmOffs*4;
  }
  O << "]";
}

void ARMAsmPrinter::printAddrMode6Operand(const MachineInstr *MI, int Op) {
  const MachineOperand &MO1 = MI->getOperand(Op);
  const MachineOperand &MO2 = MI->getOperand(Op+1);
  const MachineOperand &MO3 = MI->getOperand(Op+2);

  // FIXME: No support yet for specifying alignment.
  O << "[" << TRI->getAsmName(MO1.getReg()) << "]";

  if (ARM_AM::getAM6WBFlag(MO3.getImm())) {
    if (MO2.getReg() == 0)
      O << "!";
    else
      O << ", " << TRI->getAsmName(MO2.getReg());
  }
}

void ARMAsmPrinter::printAddrModePCOperand(const MachineInstr *MI, int Op,
                                           const char *Modifier) {
  if (Modifier && strcmp(Modifier, "label") == 0) {
    printPCLabel(MI, Op+1);
    return;
  }

  const MachineOperand &MO1 = MI->getOperand(Op);
  assert(TargetRegisterInfo::isPhysicalRegister(MO1.getReg()));
  O << "[pc, +" << TRI->getAsmName(MO1.getReg()) << "]";
}

void
ARMAsmPrinter::printBitfieldInvMaskImmOperand(const MachineInstr *MI, int Op) {
  const MachineOperand &MO = MI->getOperand(Op);
  uint32_t v = ~MO.getImm();
  int32_t lsb = CountTrailingZeros_32(v);
  int32_t width = (32 - CountLeadingZeros_32 (v)) - lsb;
  assert(MO.isImm() && "Not a valid bf_inv_mask_imm value!");
  O << "#" << lsb << ", #" << width;
}

//===--------------------------------------------------------------------===//

void
ARMAsmPrinter::printThumbITMask(const MachineInstr *MI, int Op) {
  // (3 - the number of trailing zeros) is the number of then / else.
  unsigned Mask = MI->getOperand(Op).getImm();
  unsigned NumTZ = CountTrailingZeros_32(Mask);
  assert(NumTZ <= 3 && "Invalid IT mask!");
  for (unsigned Pos = 3, e = NumTZ; Pos > e; --Pos) {
    bool T = (Mask & (1 << Pos)) == 0;
    if (T)
      O << 't';
    else
      O << 'e';
  }
}

void
ARMAsmPrinter::printThumbAddrModeRROperand(const MachineInstr *MI, int Op) {
  const MachineOperand &MO1 = MI->getOperand(Op);
  const MachineOperand &MO2 = MI->getOperand(Op+1);
  O << "[" << TRI->getAsmName(MO1.getReg());
  O << ", " << TRI->getAsmName(MO2.getReg()) << "]";
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

  O << "[" << TRI->getAsmName(MO1.getReg());
  if (MO3.getReg())
    O << ", " << TRI->getAsmName(MO3.getReg());
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
  O << "[" << TRI->getAsmName(MO1.getReg());
  if (unsigned ImmOffs = MO2.getImm())
    O << ", #" << ImmOffs << " * 4";
  O << "]";
}

//===--------------------------------------------------------------------===//

// Constant shifts t2_so_reg is a 2-operand unit corresponding to the Thumb2
// register with shift forms.
// REG 0   0           - e.g. R5
// REG IMM, SH_OPC     - e.g. R5, LSL #3
void ARMAsmPrinter::printT2SOOperand(const MachineInstr *MI, int OpNum) {
  const MachineOperand &MO1 = MI->getOperand(OpNum);
  const MachineOperand &MO2 = MI->getOperand(OpNum+1);

  unsigned Reg = MO1.getReg();
  assert(TargetRegisterInfo::isPhysicalRegister(Reg));
  O << TRI->getAsmName(Reg);

  // Print the shift opc.
  O << ", "
    << ARM_AM::getShiftOpcStr(ARM_AM::getSORegShOp(MO2.getImm()))
    << " ";

  assert(MO2.isImm() && "Not a valid t2_so_reg value!");
  O << "#" << ARM_AM::getSORegOffset(MO2.getImm());
}

void ARMAsmPrinter::printT2AddrModeImm12Operand(const MachineInstr *MI,
                                                int OpNum) {
  const MachineOperand &MO1 = MI->getOperand(OpNum);
  const MachineOperand &MO2 = MI->getOperand(OpNum+1);

  O << "[" << TRI->getAsmName(MO1.getReg());

  unsigned OffImm = MO2.getImm();
  if (OffImm)  // Don't print +0.
    O << ", #+" << OffImm;
  O << "]";
}

void ARMAsmPrinter::printT2AddrModeImm8Operand(const MachineInstr *MI,
                                               int OpNum) {
  const MachineOperand &MO1 = MI->getOperand(OpNum);
  const MachineOperand &MO2 = MI->getOperand(OpNum+1);

  O << "[" << TRI->getAsmName(MO1.getReg());

  int32_t OffImm = (int32_t)MO2.getImm();
  // Don't print +0.
  if (OffImm < 0)
    O << ", #-" << -OffImm;
  else if (OffImm > 0)
    O << ", #+" << OffImm;
  O << "]";
}

void ARMAsmPrinter::printT2AddrModeImm8s4Operand(const MachineInstr *MI,
                                                 int OpNum) {
  const MachineOperand &MO1 = MI->getOperand(OpNum);
  const MachineOperand &MO2 = MI->getOperand(OpNum+1);

  O << "[" << TRI->getAsmName(MO1.getReg());

  int32_t OffImm = (int32_t)MO2.getImm() / 4;
  // Don't print +0.
  if (OffImm < 0)
    O << ", #-" << -OffImm << " * 4";
  else if (OffImm > 0)
    O << ", #+" << OffImm << " * 4";
  O << "]";
}

void ARMAsmPrinter::printT2AddrModeImm8OffsetOperand(const MachineInstr *MI,
                                                     int OpNum) {
  const MachineOperand &MO1 = MI->getOperand(OpNum);
  int32_t OffImm = (int32_t)MO1.getImm();
  // Don't print +0.
  if (OffImm < 0)
    O << "#-" << -OffImm;
  else if (OffImm > 0)
    O << "#+" << OffImm;
}

void ARMAsmPrinter::printT2AddrModeSoRegOperand(const MachineInstr *MI,
                                                int OpNum) {
  const MachineOperand &MO1 = MI->getOperand(OpNum);
  const MachineOperand &MO2 = MI->getOperand(OpNum+1);
  const MachineOperand &MO3 = MI->getOperand(OpNum+2);

  O << "[" << TRI->getAsmName(MO1.getReg());

  assert(MO2.getReg() && "Invalid so_reg load / store address!");
  O << ", " << TRI->getAsmName(MO2.getReg());

  unsigned ShAmt = MO3.getImm();
  if (ShAmt) {
    assert(ShAmt <= 3 && "Not a valid Thumb2 addressing mode!");
    O << ", lsl #" << ShAmt;
  }
  O << "]";
}


//===--------------------------------------------------------------------===//

void ARMAsmPrinter::printPredicateOperand(const MachineInstr *MI, int OpNum) {
  ARMCC::CondCodes CC = (ARMCC::CondCodes)MI->getOperand(OpNum).getImm();
  if (CC != ARMCC::AL)
    O << ARMCondCodeToString(CC);
}

void ARMAsmPrinter::printSBitModifierOperand(const MachineInstr *MI, int OpNum){
  unsigned Reg = MI->getOperand(OpNum).getReg();
  if (Reg) {
    assert(Reg == ARM::CPSR && "Expect ARM CPSR register!");
    O << 's';
  }
}

void ARMAsmPrinter::printPCLabel(const MachineInstr *MI, int OpNum) {
  int Id = (int)MI->getOperand(OpNum).getImm();
  O << TAI->getPrivateGlobalPrefix() << "PC" << Id;
}

void ARMAsmPrinter::printRegisterList(const MachineInstr *MI, int OpNum) {
  O << "{";
  for (unsigned i = OpNum, e = MI->getNumOperands(); i != e; ++i) {
    if (MI->getOperand(i).isImplicit())
      continue;
    if ((int)i != OpNum) O << ", ";
    printOperand(MI, i);
  }
  O << "}";
}

void ARMAsmPrinter::printCPInstOperand(const MachineInstr *MI, int OpNum,
                                       const char *Modifier) {
  assert(Modifier && "This operand only works with a modifier!");
  // There are two aspects to a CONSTANTPOOL_ENTRY operand, the label and the
  // data itself.
  if (!strcmp(Modifier, "label")) {
    unsigned ID = MI->getOperand(OpNum).getImm();
    O << TAI->getPrivateGlobalPrefix() << "CPI" << getFunctionNumber()
      << '_' << ID << ":\n";
  } else {
    assert(!strcmp(Modifier, "cpentry") && "Unknown modifier for CPE");
    unsigned CPI = MI->getOperand(OpNum).getIndex();

    const MachineConstantPoolEntry &MCPE = MCP->getConstants()[CPI];
    
    if (MCPE.isMachineConstantPoolEntry()) {
      EmitMachineConstantPoolValue(MCPE.Val.MachineCPVal);
    } else {
      EmitGlobalConstant(MCPE.Val.ConstVal);
    }
  }
}

void ARMAsmPrinter::printJTBlockOperand(const MachineInstr *MI, int OpNum) {
  assert(!Subtarget->isThumb2() && "Thumb2 should use double-jump jumptables!");

  const MachineOperand &MO1 = MI->getOperand(OpNum);
  const MachineOperand &MO2 = MI->getOperand(OpNum+1); // Unique Id
  unsigned JTI = MO1.getIndex();
  O << TAI->getPrivateGlobalPrefix() << "JTI" << getFunctionNumber()
    << '_' << JTI << '_' << MO2.getImm() << ":\n";

  const char *JTEntryDirective = TAI->getData32bitsDirective();

  const MachineFunction *MF = MI->getParent()->getParent();
  const MachineJumpTableInfo *MJTI = MF->getJumpTableInfo();
  const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
  const std::vector<MachineBasicBlock*> &JTBBs = JT[JTI].MBBs;
  bool UseSet= TAI->getSetDirective() && TM.getRelocationModel() == Reloc::PIC_;
  SmallPtrSet<MachineBasicBlock*, 8> JTSets;
  for (unsigned i = 0, e = JTBBs.size(); i != e; ++i) {
    MachineBasicBlock *MBB = JTBBs[i];
    bool isNew = JTSets.insert(MBB);

    if (UseSet && isNew)
      printPICJumpTableSetLabel(JTI, MO2.getImm(), MBB);

    O << JTEntryDirective << ' ';
    if (UseSet)
      O << TAI->getPrivateGlobalPrefix() << getFunctionNumber()
        << '_' << JTI << '_' << MO2.getImm()
        << "_set_" << MBB->getNumber();
    else if (TM.getRelocationModel() == Reloc::PIC_) {
      printBasicBlockLabel(MBB, false, false, false);
      O << '-' << TAI->getPrivateGlobalPrefix() << "JTI"
        << getFunctionNumber() << '_' << JTI << '_' << MO2.getImm();
    } else {
      printBasicBlockLabel(MBB, false, false, false);
    }
    if (i != e-1)
      O << '\n';
  }
}

void ARMAsmPrinter::printJT2BlockOperand(const MachineInstr *MI, int OpNum) {
  const MachineOperand &MO1 = MI->getOperand(OpNum);
  const MachineOperand &MO2 = MI->getOperand(OpNum+1); // Unique Id
  unsigned JTI = MO1.getIndex();
  O << TAI->getPrivateGlobalPrefix() << "JTI" << getFunctionNumber()
    << '_' << JTI << '_' << MO2.getImm() << ":\n";

  const MachineFunction *MF = MI->getParent()->getParent();
  const MachineJumpTableInfo *MJTI = MF->getJumpTableInfo();
  const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
  const std::vector<MachineBasicBlock*> &JTBBs = JT[JTI].MBBs;
  bool ByteOffset = false, HalfWordOffset = false;
  if (MI->getOpcode() == ARM::t2TBB)
    ByteOffset = true;
  else if (MI->getOpcode() == ARM::t2TBH)
    HalfWordOffset = true;

  for (unsigned i = 0, e = JTBBs.size(); i != e; ++i) {
    MachineBasicBlock *MBB = JTBBs[i];
    if (ByteOffset)
      O << TAI->getData8bitsDirective();
    else if (HalfWordOffset)
      O << TAI->getData16bitsDirective();
    if (ByteOffset || HalfWordOffset) {
      O << '(';
      printBasicBlockLabel(MBB, false, false, false);
      O << "-" << TAI->getPrivateGlobalPrefix() << "JTI" << getFunctionNumber()
        << '_' << JTI << '_' << MO2.getImm() << ")/2";
    } else {
      O << "\tb.w ";
      printBasicBlockLabel(MBB, false, false, false);
    }
    if (i != e-1)
      O << '\n';
  }

  // Make sure the instruction that follows TBB is 2-byte aligned.
  // FIXME: Constant island pass should insert an "ALIGN" instruction instead.
  if (ByteOffset && (JTBBs.size() & 1)) {
    O << '\n';
    EmitAlignment(1);
  }
}

void ARMAsmPrinter::printTBAddrMode(const MachineInstr *MI, int OpNum) {
  O << "[pc, " << TRI->getAsmName(MI->getOperand(OpNum).getReg());
  if (MI->getOpcode() == ARM::t2TBH)
    O << ", lsl #1";
  O << ']';
}

void ARMAsmPrinter::printNoHashImmediate(const MachineInstr *MI, int OpNum) {
  O << MI->getOperand(OpNum).getImm();
}

bool ARMAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNum,
                                    unsigned AsmVariant, const char *ExtraCode){
  // Does this asm operand have a single letter operand modifier?
  if (ExtraCode && ExtraCode[0]) {
    if (ExtraCode[1] != 0) return true; // Unknown modifier.

    switch (ExtraCode[0]) {
    default: return true;  // Unknown modifier.
    case 'a': // Print as a memory address.
      if (MI->getOperand(OpNum).isReg()) {
        O << "[" << TRI->getAsmName(MI->getOperand(OpNum).getReg()) << "]";
        return false;
      }
      // Fallthrough
    case 'c': // Don't print "#" before an immediate operand.
      if (!MI->getOperand(OpNum).isImm())
        return true;
      printNoHashImmediate(MI, OpNum);
      return false;
    case 'P': // Print a VFP double precision register.
      printOperand(MI, OpNum);
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
      if (!MI->getOperand(OpNum).isReg() ||
          OpNum+1 == MI->getNumOperands() ||
          !MI->getOperand(OpNum+1).isReg())
        return true;
      ++OpNum;   // Return the high-part.
    }
  }
  
  printOperand(MI, OpNum);
  return false;
}

bool ARMAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                          unsigned OpNum, unsigned AsmVariant,
                                          const char *ExtraCode) {
  if (ExtraCode && ExtraCode[0])
    return true; // Unknown modifier.
  printAddrMode2Operand(MI, OpNum);
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
  DW = getAnalysisIfAvailable<DwarfWriter>();

  // Use unified assembler syntax mode for Thumb.
  if (Subtarget->isThumb())
    O << "\t.syntax unified\n";

  // Emit ARM Build Attributes
  if (Subtarget->isTargetELF()) {
    // CPU Type
    std::string CPUString = Subtarget->getCPUString();
    if (CPUString != "generic")
      O << "\t.cpu " << CPUString << '\n';

    // FIXME: Emit FPU type
    if (Subtarget->hasVFP2())
      O << "\t.eabi_attribute " << ARMBuildAttrs::VFP_arch << ", 2\n";

    // Signal various FP modes.
    if (!UnsafeFPMath)
      O << "\t.eabi_attribute " << ARMBuildAttrs::ABI_FP_denormal << ", 1\n"
        << "\t.eabi_attribute " << ARMBuildAttrs::ABI_FP_exceptions << ", 1\n";

    if (FiniteOnlyFPMath())
      O << "\t.eabi_attribute " << ARMBuildAttrs::ABI_FP_number_model << ", 1\n";
    else
      O << "\t.eabi_attribute " << ARMBuildAttrs::ABI_FP_number_model << ", 3\n";

    // 8-bytes alignment stuff.
    O << "\t.eabi_attribute " << ARMBuildAttrs::ABI_align8_needed << ", 1\n"
      << "\t.eabi_attribute " << ARMBuildAttrs::ABI_align8_preserved << ", 1\n";

    // Hard float.  Use both S and D registers and conform to AAPCS-VFP.
    if (Subtarget->isAAPCS_ABI() && FloatABIType == FloatABI::Hard)
      O << "\t.eabi_attribute " << ARMBuildAttrs::ABI_HardFP_use << ", 3\n"
        << "\t.eabi_attribute " << ARMBuildAttrs::ABI_VFP_args << ", 1\n";

    // FIXME: Should we signal R9 usage?
  }

  return Result;
}

void ARMAsmPrinter::PrintGlobalVariable(const GlobalVariable* GVar) {
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

  std::string name = Mang->getMangledName(GVar);
  Constant *C = GVar->getInitializer();
  const Type *Type = C->getType();
  unsigned Size = TD->getTypeAllocSize(Type);
  unsigned Align = TD->getPreferredAlignmentLog(GVar);
  bool isDarwin = Subtarget->isTargetDarwin();

  printVisibility(name, GVar->getVisibility());

  if (Subtarget->isTargetELF())
    O << "\t.type " << name << ",%object\n";
  
  const MCSection *TheSection =
    getObjFileLowering().SectionForGlobal(GVar, Mang, TM);
  OutStreamer.SwitchSection(TheSection);

  // FIXME: get this stuff from section kind flags.
  if (C->isNullValue() && !GVar->hasSection() && !GVar->isThreadLocal() &&
      // Don't put things that should go in the cstring section into "comm".
      !TheSection->getKind().isMergeableCString()) {
    if (GVar->hasExternalLinkage()) {
      if (const char *Directive = TAI->getZeroFillDirective()) {
        O << "\t.globl\t" << name << "\n";
        O << Directive << "__DATA, __common, " << name << ", "
          << Size << ", " << Align << "\n";
        return;
      }
    }

    if (GVar->hasLocalLinkage() || GVar->isWeakForLinker()) {
      if (Size == 0) Size = 1;   // .comm Foo, 0 is undefined, avoid it.

      if (isDarwin) {
        if (GVar->hasLocalLinkage()) {
          O << TAI->getLCOMMDirective()  << name << "," << Size
            << ',' << Align;
        } else if (GVar->hasCommonLinkage()) {
          O << TAI->getCOMMDirective()  << name << "," << Size
            << ',' << Align;
        } else {
          OutStreamer.SwitchSection(TheSection);
          O << "\t.globl " << name << '\n'
            << TAI->getWeakDefDirective() << name << '\n';
          EmitAlignment(Align, GVar);
          O << name << ":";
          if (VerboseAsm) {
            O << "\t\t\t\t" << TAI->getCommentString() << ' ';
            WriteAsOperand(O, GVar, /*PrintType=*/false, GVar->getParent());
          }
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
        if (GVar->hasLocalLinkage())
          O << "\t.local\t" << name << "\n";
        O << TAI->getCOMMDirective()  << name << "," << Size;
        if (TAI->getCOMMDirectiveTakesAlignment())
          O << "," << (TAI->getAlignmentIsInBytes() ? (1 << Align) : Align);
      }
      if (VerboseAsm) {
        O << "\t\t" << TAI->getCommentString() << " ";
        WriteAsOperand(O, GVar, /*PrintType=*/false, GVar->getParent());
      }
      O << "\n";
      return;
    }
  }
  
  switch (GVar->getLinkage()) {
  case GlobalValue::CommonLinkage:
  case GlobalValue::LinkOnceAnyLinkage:
  case GlobalValue::LinkOnceODRLinkage:
  case GlobalValue::WeakAnyLinkage:
  case GlobalValue::WeakODRLinkage:
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
    break;
  case GlobalValue::PrivateLinkage:
  case GlobalValue::LinkerPrivateLinkage:
  case GlobalValue::InternalLinkage:
    break;
  default:
    llvm_unreachable("Unknown linkage type!");
  }

  EmitAlignment(Align, GVar);
  O << name << ":";
  if (VerboseAsm) {
    O << "\t\t\t\t" << TAI->getCommentString() << " ";
    WriteAsOperand(O, GVar, /*PrintType=*/false, GVar->getParent());
  }
  O << "\n";
  if (TAI->hasDotTypeDotSizeDirective())
    O << "\t.size " << name << ", " << Size << "\n";

  EmitGlobalConstant(C);
  O << '\n';
}


bool ARMAsmPrinter::doFinalization(Module &M) {
  if (Subtarget->isTargetDarwin()) {
    // All darwin targets use mach-o.
    TargetLoweringObjectFileMachO &TLOFMacho = 
      static_cast<TargetLoweringObjectFileMachO &>(getObjFileLowering());
    
    O << '\n';

    // Output non-lazy-pointers for external and common global variables.
    if (!GVNonLazyPtrs.empty()) {
      // Switch with ".non_lazy_symbol_pointer" directive.
      OutStreamer.SwitchSection(TLOFMacho.getNonLazySymbolPointerSection());
      EmitAlignment(2);
      for (StringMap<std::string>::iterator I = GVNonLazyPtrs.begin(),
           E = GVNonLazyPtrs.end(); I != E; ++I) {
        O << I->second << ":\n";
        O << "\t.indirect_symbol " << I->getKeyData() << "\n";
        O << "\t.long\t0\n";
      }
    }

    if (!HiddenGVNonLazyPtrs.empty()) {
      OutStreamer.SwitchSection(getObjFileLowering().getDataSection());
      EmitAlignment(2);
      for (StringMap<std::string>::iterator I = HiddenGVNonLazyPtrs.begin(),
             E = HiddenGVNonLazyPtrs.end(); I != E; ++I) {
        O << I->second << ":\n";
        O << "\t.long " << I->getKeyData() << "\n";
      }
    }

    // Funny Darwin hack: This flag tells the linker that no global symbols
    // contain code that falls through to other global symbols (e.g. the obvious
    // implementation of multiple entry points).  If this doesn't occur, the
    // linker can safely perform dead code stripping.  Since LLVM never
    // generates code that does this, it is always safe to set.
    O << "\t.subsections_via_symbols\n";
  }

  return AsmPrinter::doFinalization(M);
}

// Force static initialization.
extern "C" void LLVMInitializeARMAsmPrinter() { 
  RegisterAsmPrinter<ARMAsmPrinter> X(TheARMTarget);
  RegisterAsmPrinter<ARMAsmPrinter> Y(TheThumbTarget);
}
