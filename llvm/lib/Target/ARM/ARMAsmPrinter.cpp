//===-- ARMAsmPrinter.cpp - Print machine code to an ARM .s file ----------===//
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

#include "ARMAsmPrinter.h"
#include "ARM.h"
#include "ARMConstantPoolValue.h"
#include "ARMMachineFunctionInfo.h"
#include "ARMTargetMachine.h"
#include "ARMTargetObjectFile.h"
#include "InstPrinter/ARMInstPrinter.h"
#include "MCTargetDesc/ARMAddressingModes.h"
#include "MCTargetDesc/ARMMCExpr.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/ARMBuildAttributes.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetParser.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include <cctype>
using namespace llvm;

#define DEBUG_TYPE "asm-printer"

ARMAsmPrinter::ARMAsmPrinter(TargetMachine &TM,
                             std::unique_ptr<MCStreamer> Streamer)
    : AsmPrinter(TM, std::move(Streamer)), AFI(nullptr), MCP(nullptr),
      InConstantPool(false), OptimizationGoals(-1) {}

void ARMAsmPrinter::EmitFunctionBodyEnd() {
  // Make sure to terminate any constant pools that were at the end
  // of the function.
  if (!InConstantPool)
    return;
  InConstantPool = false;
  OutStreamer->EmitDataRegion(MCDR_DataRegionEnd);
}

void ARMAsmPrinter::EmitFunctionEntryLabel() {
  if (AFI->isThumbFunction()) {
    OutStreamer->EmitAssemblerFlag(MCAF_Code16);
    OutStreamer->EmitThumbFunc(CurrentFnSym);
  }

  OutStreamer->EmitLabel(CurrentFnSym);
}

void ARMAsmPrinter::EmitXXStructor(const DataLayout &DL, const Constant *CV) {
  uint64_t Size = getDataLayout().getTypeAllocSize(CV->getType());
  assert(Size && "C++ constructor pointer had zero size!");

  const GlobalValue *GV = dyn_cast<GlobalValue>(CV->stripPointerCasts());
  assert(GV && "C++ constructor pointer was not a GlobalValue!");

  const MCExpr *E = MCSymbolRefExpr::create(GetARMGVSymbol(GV,
                                                           ARMII::MO_NO_FLAG),
                                            (Subtarget->isTargetELF()
                                             ? MCSymbolRefExpr::VK_ARM_TARGET1
                                             : MCSymbolRefExpr::VK_None),
                                            OutContext);

  OutStreamer->EmitValue(E, Size);
}

/// runOnMachineFunction - This uses the EmitInstruction()
/// method to print assembly for each instruction.
///
bool ARMAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  AFI = MF.getInfo<ARMFunctionInfo>();
  MCP = MF.getConstantPool();
  Subtarget = &MF.getSubtarget<ARMSubtarget>();

  SetupMachineFunction(MF);
  const Function* F = MF.getFunction();
  const TargetMachine& TM = MF.getTarget();

  // Calculate this function's optimization goal.
  unsigned OptimizationGoal;
  if (F->hasFnAttribute(Attribute::OptimizeNone))
    // For best debugging illusion, speed and small size sacrificed
    OptimizationGoal = 6;
  else if (F->optForMinSize())
    // Aggressively for small size, speed and debug illusion sacrificed
    OptimizationGoal = 4;
  else if (F->optForSize())
    // For small size, but speed and debugging illusion preserved
    OptimizationGoal = 3;
  else if (TM.getOptLevel() == CodeGenOpt::Aggressive)
    // Aggressively for speed, small size and debug illusion sacrificed
    OptimizationGoal = 2;
  else if (TM.getOptLevel() > CodeGenOpt::None)
    // For speed, but small size and good debug illusion preserved
    OptimizationGoal = 1;
  else // TM.getOptLevel() == CodeGenOpt::None
    // For good debugging, but speed and small size preserved
    OptimizationGoal = 5;

  // Combine a new optimization goal with existing ones.
  if (OptimizationGoals == -1) // uninitialized goals
    OptimizationGoals = OptimizationGoal;
  else if (OptimizationGoals != (int)OptimizationGoal) // conflicting goals
    OptimizationGoals = 0;

  if (Subtarget->isTargetCOFF()) {
    bool Internal = F->hasInternalLinkage();
    COFF::SymbolStorageClass Scl = Internal ? COFF::IMAGE_SYM_CLASS_STATIC
                                            : COFF::IMAGE_SYM_CLASS_EXTERNAL;
    int Type = COFF::IMAGE_SYM_DTYPE_FUNCTION << COFF::SCT_COMPLEX_TYPE_SHIFT;

    OutStreamer->BeginCOFFSymbolDef(CurrentFnSym);
    OutStreamer->EmitCOFFSymbolStorageClass(Scl);
    OutStreamer->EmitCOFFSymbolType(Type);
    OutStreamer->EndCOFFSymbolDef();
  }

  // Emit the rest of the function body.
  EmitFunctionBody();

  // If we need V4T thumb mode Register Indirect Jump pads, emit them.
  // These are created per function, rather than per TU, since it's
  // relatively easy to exceed the thumb branch range within a TU.
  if (! ThumbIndirectPads.empty()) {
    OutStreamer->EmitAssemblerFlag(MCAF_Code16);
    EmitAlignment(1);
    for (unsigned i = 0, e = ThumbIndirectPads.size(); i < e; i++) {
      OutStreamer->EmitLabel(ThumbIndirectPads[i].second);
      EmitToStreamer(*OutStreamer, MCInstBuilder(ARM::tBX)
        .addReg(ThumbIndirectPads[i].first)
        // Add predicate operands.
        .addImm(ARMCC::AL)
        .addReg(0));
    }
    ThumbIndirectPads.clear();
  }

  // We didn't modify anything.
  return false;
}

void ARMAsmPrinter::printOperand(const MachineInstr *MI, int OpNum,
                                 raw_ostream &O) {
  const MachineOperand &MO = MI->getOperand(OpNum);
  unsigned TF = MO.getTargetFlags();

  switch (MO.getType()) {
  default: llvm_unreachable("<unknown operand type>");
  case MachineOperand::MO_Register: {
    unsigned Reg = MO.getReg();
    assert(TargetRegisterInfo::isPhysicalRegister(Reg));
    assert(!MO.getSubReg() && "Subregs should be eliminated!");
    if(ARM::GPRPairRegClass.contains(Reg)) {
      const MachineFunction &MF = *MI->getParent()->getParent();
      const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
      Reg = TRI->getSubReg(Reg, ARM::gsub_0);
    }
    O << ARMInstPrinter::getRegisterName(Reg);
    break;
  }
  case MachineOperand::MO_Immediate: {
    int64_t Imm = MO.getImm();
    O << '#';
    if (TF == ARMII::MO_LO16)
      O << ":lower16:";
    else if (TF == ARMII::MO_HI16)
      O << ":upper16:";
    O << Imm;
    break;
  }
  case MachineOperand::MO_MachineBasicBlock:
    MO.getMBB()->getSymbol()->print(O, MAI);
    return;
  case MachineOperand::MO_GlobalAddress: {
    const GlobalValue *GV = MO.getGlobal();
    if (TF & ARMII::MO_LO16)
      O << ":lower16:";
    else if (TF & ARMII::MO_HI16)
      O << ":upper16:";
    GetARMGVSymbol(GV, TF)->print(O, MAI);

    printOffset(MO.getOffset(), O);
    break;
  }
  case MachineOperand::MO_ConstantPoolIndex:
    GetCPISymbol(MO.getIndex())->print(O, MAI);
    break;
  }
}

//===--------------------------------------------------------------------===//

MCSymbol *ARMAsmPrinter::
GetARMJTIPICJumpTableLabel(unsigned uid) const {
  const DataLayout &DL = getDataLayout();
  SmallString<60> Name;
  raw_svector_ostream(Name) << DL.getPrivateGlobalPrefix() << "JTI"
                            << getFunctionNumber() << '_' << uid;
  return OutContext.getOrCreateSymbol(Name);
}

bool ARMAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNum,
                                    unsigned AsmVariant, const char *ExtraCode,
                                    raw_ostream &O) {
  // Does this asm operand have a single letter operand modifier?
  if (ExtraCode && ExtraCode[0]) {
    if (ExtraCode[1] != 0) return true; // Unknown modifier.

    switch (ExtraCode[0]) {
    default:
      // See if this is a generic print operand
      return AsmPrinter::PrintAsmOperand(MI, OpNum, AsmVariant, ExtraCode, O);
    case 'a': // Print as a memory address.
      if (MI->getOperand(OpNum).isReg()) {
        O << "["
          << ARMInstPrinter::getRegisterName(MI->getOperand(OpNum).getReg())
          << "]";
        return false;
      }
      LLVM_FALLTHROUGH;
    case 'c': // Don't print "#" before an immediate operand.
      if (!MI->getOperand(OpNum).isImm())
        return true;
      O << MI->getOperand(OpNum).getImm();
      return false;
    case 'P': // Print a VFP double precision register.
    case 'q': // Print a NEON quad precision register.
      printOperand(MI, OpNum, O);
      return false;
    case 'y': // Print a VFP single precision register as indexed double.
      if (MI->getOperand(OpNum).isReg()) {
        unsigned Reg = MI->getOperand(OpNum).getReg();
        const TargetRegisterInfo *TRI = MF->getSubtarget().getRegisterInfo();
        // Find the 'd' register that has this 's' register as a sub-register,
        // and determine the lane number.
        for (MCSuperRegIterator SR(Reg, TRI); SR.isValid(); ++SR) {
          if (!ARM::DPRRegClass.contains(*SR))
            continue;
          bool Lane0 = TRI->getSubReg(*SR, ARM::ssub_0) == Reg;
          O << ARMInstPrinter::getRegisterName(*SR) << (Lane0 ? "[0]" : "[1]");
          return false;
        }
      }
      return true;
    case 'B': // Bitwise inverse of integer or symbol without a preceding #.
      if (!MI->getOperand(OpNum).isImm())
        return true;
      O << ~(MI->getOperand(OpNum).getImm());
      return false;
    case 'L': // The low 16 bits of an immediate constant.
      if (!MI->getOperand(OpNum).isImm())
        return true;
      O << (MI->getOperand(OpNum).getImm() & 0xffff);
      return false;
    case 'M': { // A register range suitable for LDM/STM.
      if (!MI->getOperand(OpNum).isReg())
        return true;
      const MachineOperand &MO = MI->getOperand(OpNum);
      unsigned RegBegin = MO.getReg();
      // This takes advantage of the 2 operand-ness of ldm/stm and that we've
      // already got the operands in registers that are operands to the
      // inline asm statement.
      O << "{";
      if (ARM::GPRPairRegClass.contains(RegBegin)) {
        const TargetRegisterInfo *TRI = MF->getSubtarget().getRegisterInfo();
        unsigned Reg0 = TRI->getSubReg(RegBegin, ARM::gsub_0);
        O << ARMInstPrinter::getRegisterName(Reg0) << ", ";
        RegBegin = TRI->getSubReg(RegBegin, ARM::gsub_1);
      }
      O << ARMInstPrinter::getRegisterName(RegBegin);

      // FIXME: The register allocator not only may not have given us the
      // registers in sequence, but may not be in ascending registers. This
      // will require changes in the register allocator that'll need to be
      // propagated down here if the operands change.
      unsigned RegOps = OpNum + 1;
      while (MI->getOperand(RegOps).isReg()) {
        O << ", "
          << ARMInstPrinter::getRegisterName(MI->getOperand(RegOps).getReg());
        RegOps++;
      }

      O << "}";

      return false;
    }
    case 'R': // The most significant register of a pair.
    case 'Q': { // The least significant register of a pair.
      if (OpNum == 0)
        return true;
      const MachineOperand &FlagsOP = MI->getOperand(OpNum - 1);
      if (!FlagsOP.isImm())
        return true;
      unsigned Flags = FlagsOP.getImm();

      // This operand may not be the one that actually provides the register. If
      // it's tied to a previous one then we should refer instead to that one
      // for registers and their classes.
      unsigned TiedIdx;
      if (InlineAsm::isUseOperandTiedToDef(Flags, TiedIdx)) {
        for (OpNum = InlineAsm::MIOp_FirstOperand; TiedIdx; --TiedIdx) {
          unsigned OpFlags = MI->getOperand(OpNum).getImm();
          OpNum += InlineAsm::getNumOperandRegisters(OpFlags) + 1;
        }
        Flags = MI->getOperand(OpNum).getImm();

        // Later code expects OpNum to be pointing at the register rather than
        // the flags.
        OpNum += 1;
      }

      unsigned NumVals = InlineAsm::getNumOperandRegisters(Flags);
      unsigned RC;
      InlineAsm::hasRegClassConstraint(Flags, RC);
      if (RC == ARM::GPRPairRegClassID) {
        if (NumVals != 1)
          return true;
        const MachineOperand &MO = MI->getOperand(OpNum);
        if (!MO.isReg())
          return true;
        const TargetRegisterInfo *TRI = MF->getSubtarget().getRegisterInfo();
        unsigned Reg = TRI->getSubReg(MO.getReg(), ExtraCode[0] == 'Q' ?
            ARM::gsub_0 : ARM::gsub_1);
        O << ARMInstPrinter::getRegisterName(Reg);
        return false;
      }
      if (NumVals != 2)
        return true;
      unsigned RegOp = ExtraCode[0] == 'Q' ? OpNum : OpNum + 1;
      if (RegOp >= MI->getNumOperands())
        return true;
      const MachineOperand &MO = MI->getOperand(RegOp);
      if (!MO.isReg())
        return true;
      unsigned Reg = MO.getReg();
      O << ARMInstPrinter::getRegisterName(Reg);
      return false;
    }

    case 'e': // The low doubleword register of a NEON quad register.
    case 'f': { // The high doubleword register of a NEON quad register.
      if (!MI->getOperand(OpNum).isReg())
        return true;
      unsigned Reg = MI->getOperand(OpNum).getReg();
      if (!ARM::QPRRegClass.contains(Reg))
        return true;
      const TargetRegisterInfo *TRI = MF->getSubtarget().getRegisterInfo();
      unsigned SubReg = TRI->getSubReg(Reg, ExtraCode[0] == 'e' ?
                                       ARM::dsub_0 : ARM::dsub_1);
      O << ARMInstPrinter::getRegisterName(SubReg);
      return false;
    }

    // This modifier is not yet supported.
    case 'h': // A range of VFP/NEON registers suitable for VLD1/VST1.
      return true;
    case 'H': { // The highest-numbered register of a pair.
      const MachineOperand &MO = MI->getOperand(OpNum);
      if (!MO.isReg())
        return true;
      const MachineFunction &MF = *MI->getParent()->getParent();
      const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
      unsigned Reg = MO.getReg();
      if(!ARM::GPRPairRegClass.contains(Reg))
        return false;
      Reg = TRI->getSubReg(Reg, ARM::gsub_1);
      O << ARMInstPrinter::getRegisterName(Reg);
      return false;
    }
    }
  }

  printOperand(MI, OpNum, O);
  return false;
}

bool ARMAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                          unsigned OpNum, unsigned AsmVariant,
                                          const char *ExtraCode,
                                          raw_ostream &O) {
  // Does this asm operand have a single letter operand modifier?
  if (ExtraCode && ExtraCode[0]) {
    if (ExtraCode[1] != 0) return true; // Unknown modifier.

    switch (ExtraCode[0]) {
      case 'A': // A memory operand for a VLD1/VST1 instruction.
      default: return true;  // Unknown modifier.
      case 'm': // The base register of a memory operand.
        if (!MI->getOperand(OpNum).isReg())
          return true;
        O << ARMInstPrinter::getRegisterName(MI->getOperand(OpNum).getReg());
        return false;
    }
  }

  const MachineOperand &MO = MI->getOperand(OpNum);
  assert(MO.isReg() && "unexpected inline asm memory operand");
  O << "[" << ARMInstPrinter::getRegisterName(MO.getReg()) << "]";
  return false;
}

static bool isThumb(const MCSubtargetInfo& STI) {
  return STI.getFeatureBits()[ARM::ModeThumb];
}

void ARMAsmPrinter::emitInlineAsmEnd(const MCSubtargetInfo &StartInfo,
                                     const MCSubtargetInfo *EndInfo) const {
  // If either end mode is unknown (EndInfo == NULL) or different than
  // the start mode, then restore the start mode.
  const bool WasThumb = isThumb(StartInfo);
  if (!EndInfo || WasThumb != isThumb(*EndInfo)) {
    OutStreamer->EmitAssemblerFlag(WasThumb ? MCAF_Code16 : MCAF_Code32);
  }
}

void ARMAsmPrinter::EmitStartOfAsmFile(Module &M) {
  const Triple &TT = TM.getTargetTriple();
  // Use unified assembler syntax.
  OutStreamer->EmitAssemblerFlag(MCAF_SyntaxUnified);

  // Emit ARM Build Attributes
  if (TT.isOSBinFormatELF())
    emitAttributes();

  // Use the triple's architecture and subarchitecture to determine
  // if we're thumb for the purposes of the top level code16 assembler
  // flag.
  bool isThumb = TT.getArch() == Triple::thumb ||
                 TT.getArch() == Triple::thumbeb ||
                 TT.getSubArch() == Triple::ARMSubArch_v7m ||
                 TT.getSubArch() == Triple::ARMSubArch_v6m;
  if (!M.getModuleInlineAsm().empty() && isThumb)
    OutStreamer->EmitAssemblerFlag(MCAF_Code16);
}

static void
emitNonLazySymbolPointer(MCStreamer &OutStreamer, MCSymbol *StubLabel,
                         MachineModuleInfoImpl::StubValueTy &MCSym) {
  // L_foo$stub:
  OutStreamer.EmitLabel(StubLabel);
  //   .indirect_symbol _foo
  OutStreamer.EmitSymbolAttribute(MCSym.getPointer(), MCSA_IndirectSymbol);

  if (MCSym.getInt())
    // External to current translation unit.
    OutStreamer.EmitIntValue(0, 4/*size*/);
  else
    // Internal to current translation unit.
    //
    // When we place the LSDA into the TEXT section, the type info
    // pointers need to be indirect and pc-rel. We accomplish this by
    // using NLPs; however, sometimes the types are local to the file.
    // We need to fill in the value for the NLP in those cases.
    OutStreamer.EmitValue(
        MCSymbolRefExpr::create(MCSym.getPointer(), OutStreamer.getContext()),
        4 /*size*/);
}


void ARMAsmPrinter::EmitEndOfAsmFile(Module &M) {
  const Triple &TT = TM.getTargetTriple();
  if (TT.isOSBinFormatMachO()) {
    // All darwin targets use mach-o.
    const TargetLoweringObjectFileMachO &TLOFMacho =
      static_cast<const TargetLoweringObjectFileMachO &>(getObjFileLowering());
    MachineModuleInfoMachO &MMIMacho =
      MMI->getObjFileInfo<MachineModuleInfoMachO>();

    // Output non-lazy-pointers for external and common global variables.
    MachineModuleInfoMachO::SymbolListTy Stubs = MMIMacho.GetGVStubList();

    if (!Stubs.empty()) {
      // Switch with ".non_lazy_symbol_pointer" directive.
      OutStreamer->SwitchSection(TLOFMacho.getNonLazySymbolPointerSection());
      EmitAlignment(2);

      for (auto &Stub : Stubs)
        emitNonLazySymbolPointer(*OutStreamer, Stub.first, Stub.second);

      Stubs.clear();
      OutStreamer->AddBlankLine();
    }

    Stubs = MMIMacho.GetThreadLocalGVStubList();
    if (!Stubs.empty()) {
      // Switch with ".non_lazy_symbol_pointer" directive.
      OutStreamer->SwitchSection(TLOFMacho.getThreadLocalPointerSection());
      EmitAlignment(2);

      for (auto &Stub : Stubs)
        emitNonLazySymbolPointer(*OutStreamer, Stub.first, Stub.second);

      Stubs.clear();
      OutStreamer->AddBlankLine();
    }

    // Funny Darwin hack: This flag tells the linker that no global symbols
    // contain code that falls through to other global symbols (e.g. the obvious
    // implementation of multiple entry points).  If this doesn't occur, the
    // linker can safely perform dead code stripping.  Since LLVM never
    // generates code that does this, it is always safe to set.
    OutStreamer->EmitAssemblerFlag(MCAF_SubsectionsViaSymbols);
  }

  if (TT.isOSBinFormatCOFF()) {
    const auto &TLOF =
        static_cast<const TargetLoweringObjectFileCOFF &>(getObjFileLowering());

    std::string Flags;
    raw_string_ostream OS(Flags);

    for (const auto &Function : M)
      TLOF.emitLinkerFlagsForGlobal(OS, &Function, *Mang);
    for (const auto &Global : M.globals())
      TLOF.emitLinkerFlagsForGlobal(OS, &Global, *Mang);
    for (const auto &Alias : M.aliases())
      TLOF.emitLinkerFlagsForGlobal(OS, &Alias, *Mang);

    OS.flush();

    // Output collected flags
    if (!Flags.empty()) {
      OutStreamer->SwitchSection(TLOF.getDrectveSection());
      OutStreamer->EmitBytes(Flags);
    }
  }

  // The last attribute to be emitted is ABI_optimization_goals
  MCTargetStreamer &TS = *OutStreamer->getTargetStreamer();
  ARMTargetStreamer &ATS = static_cast<ARMTargetStreamer &>(TS);

  if (OptimizationGoals > 0 &&
      (Subtarget->isTargetAEABI() || Subtarget->isTargetGNUAEABI() ||
       Subtarget->isTargetMuslAEABI()))
    ATS.emitAttribute(ARMBuildAttrs::ABI_optimization_goals, OptimizationGoals);
  OptimizationGoals = -1;

  ATS.finishAttributeSection();
}

static bool isV8M(const ARMSubtarget *Subtarget) {
  // Note that v8M Baseline is a subset of v6T2!
  return (Subtarget->hasV8MBaselineOps() && !Subtarget->hasV6T2Ops()) ||
         Subtarget->hasV8MMainlineOps();
}

//===----------------------------------------------------------------------===//
// Helper routines for EmitStartOfAsmFile() and EmitEndOfAsmFile()
// FIXME:
// The following seem like one-off assembler flags, but they actually need
// to appear in the .ARM.attributes section in ELF.
// Instead of subclassing the MCELFStreamer, we do the work here.

static ARMBuildAttrs::CPUArch getArchForCPU(StringRef CPU,
                                            const ARMSubtarget *Subtarget) {
  if (CPU == "xscale")
    return ARMBuildAttrs::v5TEJ;

  if (Subtarget->hasV8Ops())
    return ARMBuildAttrs::v8_A;
  else if (Subtarget->hasV8MMainlineOps())
    return ARMBuildAttrs::v8_M_Main;
  else if (Subtarget->hasV7Ops()) {
    if (Subtarget->isMClass() && Subtarget->hasDSP())
      return ARMBuildAttrs::v7E_M;
    return ARMBuildAttrs::v7;
  } else if (Subtarget->hasV6T2Ops())
    return ARMBuildAttrs::v6T2;
  else if (Subtarget->hasV8MBaselineOps())
    return ARMBuildAttrs::v8_M_Base;
  else if (Subtarget->hasV6MOps())
    return ARMBuildAttrs::v6S_M;
  else if (Subtarget->hasV6Ops())
    return ARMBuildAttrs::v6;
  else if (Subtarget->hasV5TEOps())
    return ARMBuildAttrs::v5TE;
  else if (Subtarget->hasV5TOps())
    return ARMBuildAttrs::v5T;
  else if (Subtarget->hasV4TOps())
    return ARMBuildAttrs::v4T;
  else
    return ARMBuildAttrs::v4;
}

// Returns true if all functions have the same function attribute value
static bool haveAllFunctionsAttribute(const Module &M, StringRef Attr,
                                      StringRef Value) {
  for (auto &F : M)
    if (F.getFnAttribute(Attr).getValueAsString() != Value)
      return false;

  return true;
}


void ARMAsmPrinter::emitAttributes() {
  MCTargetStreamer &TS = *OutStreamer->getTargetStreamer();
  ARMTargetStreamer &ATS = static_cast<ARMTargetStreamer &>(TS);

  ATS.emitTextAttribute(ARMBuildAttrs::conformance, "2.09");

  ATS.switchVendor("aeabi");

  // Compute ARM ELF Attributes based on the default subtarget that
  // we'd have constructed. The existing ARM behavior isn't LTO clean
  // anyhow.
  // FIXME: For ifunc related functions we could iterate over and look
  // for a feature string that doesn't match the default one.
  const Triple &TT = TM.getTargetTriple();
  StringRef CPU = TM.getTargetCPU();
  StringRef FS = TM.getTargetFeatureString();
  std::string ArchFS = ARM_MC::ParseARMTriple(TT, CPU);
  if (!FS.empty()) {
    if (!ArchFS.empty())
      ArchFS = (Twine(ArchFS) + "," + FS).str();
    else
      ArchFS = FS;
  }
  const ARMBaseTargetMachine &ATM =
      static_cast<const ARMBaseTargetMachine &>(TM);
  const ARMSubtarget STI(TT, CPU, ArchFS, ATM, ATM.isLittleEndian());

  const std::string &CPUString = STI.getCPUString();

  if (!StringRef(CPUString).startswith("generic")) {
    // FIXME: remove krait check when GNU tools support krait cpu
    if (STI.isKrait()) {
      ATS.emitTextAttribute(ARMBuildAttrs::CPU_name, "cortex-a9");
      // We consider krait as a "cortex-a9" + hwdiv CPU
      // Enable hwdiv through ".arch_extension idiv"
      if (STI.hasDivide() || STI.hasDivideInARMMode())
        ATS.emitArchExtension(ARM::AEK_HWDIV | ARM::AEK_HWDIVARM);
    } else
      ATS.emitTextAttribute(ARMBuildAttrs::CPU_name, CPUString);
  }

  ATS.emitAttribute(ARMBuildAttrs::CPU_arch, getArchForCPU(CPUString, &STI));

  // Tag_CPU_arch_profile must have the default value of 0 when "Architecture
  // profile is not applicable (e.g. pre v7, or cross-profile code)".
  if (STI.hasV7Ops() || isV8M(&STI)) {
    if (STI.isAClass()) {
      ATS.emitAttribute(ARMBuildAttrs::CPU_arch_profile,
                        ARMBuildAttrs::ApplicationProfile);
    } else if (STI.isRClass()) {
      ATS.emitAttribute(ARMBuildAttrs::CPU_arch_profile,
                        ARMBuildAttrs::RealTimeProfile);
    } else if (STI.isMClass()) {
      ATS.emitAttribute(ARMBuildAttrs::CPU_arch_profile,
                        ARMBuildAttrs::MicroControllerProfile);
    }
  }

  ATS.emitAttribute(ARMBuildAttrs::ARM_ISA_use,
                    STI.hasARMOps() ? ARMBuildAttrs::Allowed
                                    : ARMBuildAttrs::Not_Allowed);
  if (isV8M(&STI)) {
    ATS.emitAttribute(ARMBuildAttrs::THUMB_ISA_use,
                      ARMBuildAttrs::AllowThumbDerived);
  } else if (STI.isThumb1Only()) {
    ATS.emitAttribute(ARMBuildAttrs::THUMB_ISA_use, ARMBuildAttrs::Allowed);
  } else if (STI.hasThumb2()) {
    ATS.emitAttribute(ARMBuildAttrs::THUMB_ISA_use,
                      ARMBuildAttrs::AllowThumb32);
  }

  if (STI.hasNEON()) {
    /* NEON is not exactly a VFP architecture, but GAS emit one of
     * neon/neon-fp-armv8/neon-vfpv4/vfpv3/vfpv2 for .fpu parameters */
    if (STI.hasFPARMv8()) {
      if (STI.hasCrypto())
        ATS.emitFPU(ARM::FK_CRYPTO_NEON_FP_ARMV8);
      else
        ATS.emitFPU(ARM::FK_NEON_FP_ARMV8);
    } else if (STI.hasVFP4())
      ATS.emitFPU(ARM::FK_NEON_VFPV4);
    else
      ATS.emitFPU(STI.hasFP16() ? ARM::FK_NEON_FP16 : ARM::FK_NEON);
    // Emit Tag_Advanced_SIMD_arch for ARMv8 architecture
    if (STI.hasV8Ops())
      ATS.emitAttribute(ARMBuildAttrs::Advanced_SIMD_arch,
                        STI.hasV8_1aOps() ? ARMBuildAttrs::AllowNeonARMv8_1a:
                                            ARMBuildAttrs::AllowNeonARMv8);
  } else {
    if (STI.hasFPARMv8())
      // FPv5 and FP-ARMv8 have the same instructions, so are modeled as one
      // FPU, but there are two different names for it depending on the CPU.
      ATS.emitFPU(STI.hasD16()
                  ? (STI.isFPOnlySP() ? ARM::FK_FPV5_SP_D16 : ARM::FK_FPV5_D16)
                  : ARM::FK_FP_ARMV8);
    else if (STI.hasVFP4())
      ATS.emitFPU(STI.hasD16()
                  ? (STI.isFPOnlySP() ? ARM::FK_FPV4_SP_D16 : ARM::FK_VFPV4_D16)
                  : ARM::FK_VFPV4);
    else if (STI.hasVFP3())
      ATS.emitFPU(STI.hasD16()
                  // +d16
                  ? (STI.isFPOnlySP()
                     ? (STI.hasFP16() ? ARM::FK_VFPV3XD_FP16 : ARM::FK_VFPV3XD)
                     : (STI.hasFP16() ? ARM::FK_VFPV3_D16_FP16 : ARM::FK_VFPV3_D16))
                  // -d16
                  : (STI.hasFP16() ? ARM::FK_VFPV3_FP16 : ARM::FK_VFPV3));
    else if (STI.hasVFP2())
      ATS.emitFPU(ARM::FK_VFPV2);
  }

  // RW data addressing.
  if (isPositionIndependent()) {
    ATS.emitAttribute(ARMBuildAttrs::ABI_PCS_RW_data,
                      ARMBuildAttrs::AddressRWPCRel);
  } else if (STI.isRWPI()) {
    // RWPI specific attributes.
    ATS.emitAttribute(ARMBuildAttrs::ABI_PCS_RW_data,
                      ARMBuildAttrs::AddressRWSBRel);
  }

  // RO data addressing.
  if (isPositionIndependent() || STI.isROPI()) {
    ATS.emitAttribute(ARMBuildAttrs::ABI_PCS_RO_data,
                      ARMBuildAttrs::AddressROPCRel);
  }

  // GOT use.
  if (isPositionIndependent()) {
    ATS.emitAttribute(ARMBuildAttrs::ABI_PCS_GOT_use,
                      ARMBuildAttrs::AddressGOT);
  } else {
    ATS.emitAttribute(ARMBuildAttrs::ABI_PCS_GOT_use,
                      ARMBuildAttrs::AddressDirect);
  }

  // Set FP Denormals.
  if (haveAllFunctionsAttribute(*MMI->getModule(), "denormal-fp-math",
                                "preserve-sign") ||
      TM.Options.FPDenormalType == FPDenormal::PreserveSign)
    ATS.emitAttribute(ARMBuildAttrs::ABI_FP_denormal,
                      ARMBuildAttrs::PreserveFPSign);
  else if (haveAllFunctionsAttribute(*MMI->getModule(), "denormal-fp-math",
                                     "positive-zero") ||
           TM.Options.FPDenormalType == FPDenormal::PositiveZero)
    ATS.emitAttribute(ARMBuildAttrs::ABI_FP_denormal,
                      ARMBuildAttrs::PositiveZero);
  else if (!TM.Options.UnsafeFPMath)
    ATS.emitAttribute(ARMBuildAttrs::ABI_FP_denormal,
                      ARMBuildAttrs::IEEEDenormals);
  else {
    if (!STI.hasVFP2()) {
      // When the target doesn't have an FPU (by design or
      // intention), the assumptions made on the software support
      // mirror that of the equivalent hardware support *if it
      // existed*. For v7 and better we indicate that denormals are
      // flushed preserving sign, and for V6 we indicate that
      // denormals are flushed to positive zero.
      if (STI.hasV7Ops())
        ATS.emitAttribute(ARMBuildAttrs::ABI_FP_denormal,
                          ARMBuildAttrs::PreserveFPSign);
    } else if (STI.hasVFP3()) {
      // In VFPv4, VFPv4U, VFPv3, or VFPv3U, it is preserved. That is,
      // the sign bit of the zero matches the sign bit of the input or
      // result that is being flushed to zero.
      ATS.emitAttribute(ARMBuildAttrs::ABI_FP_denormal,
                        ARMBuildAttrs::PreserveFPSign);
    }
    // For VFPv2 implementations it is implementation defined as
    // to whether denormals are flushed to positive zero or to
    // whatever the sign of zero is (ARM v7AR ARM 2.7.5). Historically
    // LLVM has chosen to flush this to positive zero (most likely for
    // GCC compatibility), so that's the chosen value here (the
    // absence of its emission implies zero).
  }

  // Set FP exceptions and rounding
  if (haveAllFunctionsAttribute(*MMI->getModule(), "no-trapping-math", "true") ||
      TM.Options.NoTrappingFPMath)
    ATS.emitAttribute(ARMBuildAttrs::ABI_FP_exceptions,
                      ARMBuildAttrs::Not_Allowed);
  else if (!TM.Options.UnsafeFPMath) {
    ATS.emitAttribute(ARMBuildAttrs::ABI_FP_exceptions, ARMBuildAttrs::Allowed);

    // If the user has permitted this code to choose the IEEE 754
    // rounding at run-time, emit the rounding attribute.
    if (TM.Options.HonorSignDependentRoundingFPMathOption)
      ATS.emitAttribute(ARMBuildAttrs::ABI_FP_rounding, ARMBuildAttrs::Allowed);
  }

  // TM.Options.NoInfsFPMath && TM.Options.NoNaNsFPMath is the
  // equivalent of GCC's -ffinite-math-only flag.
  if (TM.Options.NoInfsFPMath && TM.Options.NoNaNsFPMath)
    ATS.emitAttribute(ARMBuildAttrs::ABI_FP_number_model,
                      ARMBuildAttrs::Allowed);
  else
    ATS.emitAttribute(ARMBuildAttrs::ABI_FP_number_model,
                      ARMBuildAttrs::AllowIEE754);

  if (STI.allowsUnalignedMem())
    ATS.emitAttribute(ARMBuildAttrs::CPU_unaligned_access,
                      ARMBuildAttrs::Allowed);
  else
    ATS.emitAttribute(ARMBuildAttrs::CPU_unaligned_access,
                      ARMBuildAttrs::Not_Allowed);

  // FIXME: add more flags to ARMBuildAttributes.h
  // 8-bytes alignment stuff.
  ATS.emitAttribute(ARMBuildAttrs::ABI_align_needed, 1);
  ATS.emitAttribute(ARMBuildAttrs::ABI_align_preserved, 1);

  // ABI_HardFP_use attribute to indicate single precision FP.
  if (STI.isFPOnlySP())
    ATS.emitAttribute(ARMBuildAttrs::ABI_HardFP_use,
                      ARMBuildAttrs::HardFPSinglePrecision);

  // Hard float.  Use both S and D registers and conform to AAPCS-VFP.
  if (STI.isAAPCS_ABI() && TM.Options.FloatABIType == FloatABI::Hard)
    ATS.emitAttribute(ARMBuildAttrs::ABI_VFP_args, ARMBuildAttrs::HardFPAAPCS);

  // FIXME: Should we signal R9 usage?

  if (STI.hasFP16())
    ATS.emitAttribute(ARMBuildAttrs::FP_HP_extension, ARMBuildAttrs::AllowHPFP);

  // FIXME: To support emitting this build attribute as GCC does, the
  // -mfp16-format option and associated plumbing must be
  // supported. For now the __fp16 type is exposed by default, so this
  // attribute should be emitted with value 1.
  ATS.emitAttribute(ARMBuildAttrs::ABI_FP_16bit_format,
                    ARMBuildAttrs::FP16FormatIEEE);

  if (STI.hasMPExtension())
    ATS.emitAttribute(ARMBuildAttrs::MPextension_use, ARMBuildAttrs::AllowMP);

  // Hardware divide in ARM mode is part of base arch, starting from ARMv8.
  // If only Thumb hwdiv is present, it must also be in base arch (ARMv7-R/M).
  // It is not possible to produce DisallowDIV: if hwdiv is present in the base
  // arch, supplying -hwdiv downgrades the effective arch, via ClearImpliedBits.
  // AllowDIVExt is only emitted if hwdiv isn't available in the base arch;
  // otherwise, the default value (AllowDIVIfExists) applies.
  if (STI.hasDivideInARMMode() && !STI.hasV8Ops())
    ATS.emitAttribute(ARMBuildAttrs::DIV_use, ARMBuildAttrs::AllowDIVExt);

  if (STI.hasDSP() && isV8M(&STI))
    ATS.emitAttribute(ARMBuildAttrs::DSP_extension, ARMBuildAttrs::Allowed);

  if (MMI) {
    if (const Module *SourceModule = MMI->getModule()) {
      // ABI_PCS_wchar_t to indicate wchar_t width
      // FIXME: There is no way to emit value 0 (wchar_t prohibited).
      if (auto WCharWidthValue = mdconst::extract_or_null<ConstantInt>(
              SourceModule->getModuleFlag("wchar_size"))) {
        int WCharWidth = WCharWidthValue->getZExtValue();
        assert((WCharWidth == 2 || WCharWidth == 4) &&
               "wchar_t width must be 2 or 4 bytes");
        ATS.emitAttribute(ARMBuildAttrs::ABI_PCS_wchar_t, WCharWidth);
      }

      // ABI_enum_size to indicate enum width
      // FIXME: There is no way to emit value 0 (enums prohibited) or value 3
      //        (all enums contain a value needing 32 bits to encode).
      if (auto EnumWidthValue = mdconst::extract_or_null<ConstantInt>(
              SourceModule->getModuleFlag("min_enum_size"))) {
        int EnumWidth = EnumWidthValue->getZExtValue();
        assert((EnumWidth == 1 || EnumWidth == 4) &&
               "Minimum enum width must be 1 or 4 bytes");
        int EnumBuildAttr = EnumWidth == 1 ? 1 : 2;
        ATS.emitAttribute(ARMBuildAttrs::ABI_enum_size, EnumBuildAttr);
      }
    }
  }

  // We currently do not support using R9 as the TLS pointer.
  if (STI.isRWPI())
    ATS.emitAttribute(ARMBuildAttrs::ABI_PCS_R9_use,
                      ARMBuildAttrs::R9IsSB);
  else if (STI.isR9Reserved())
    ATS.emitAttribute(ARMBuildAttrs::ABI_PCS_R9_use,
                      ARMBuildAttrs::R9Reserved);
  else
    ATS.emitAttribute(ARMBuildAttrs::ABI_PCS_R9_use,
                      ARMBuildAttrs::R9IsGPR);

  if (STI.hasTrustZone() && STI.hasVirtualization())
    ATS.emitAttribute(ARMBuildAttrs::Virtualization_use,
                      ARMBuildAttrs::AllowTZVirtualization);
  else if (STI.hasTrustZone())
    ATS.emitAttribute(ARMBuildAttrs::Virtualization_use,
                      ARMBuildAttrs::AllowTZ);
  else if (STI.hasVirtualization())
    ATS.emitAttribute(ARMBuildAttrs::Virtualization_use,
                      ARMBuildAttrs::AllowVirtualization);
}

//===----------------------------------------------------------------------===//

static MCSymbol *getPICLabel(const char *Prefix, unsigned FunctionNumber,
                             unsigned LabelId, MCContext &Ctx) {

  MCSymbol *Label = Ctx.getOrCreateSymbol(Twine(Prefix)
                       + "PC" + Twine(FunctionNumber) + "_" + Twine(LabelId));
  return Label;
}

static MCSymbolRefExpr::VariantKind
getModifierVariantKind(ARMCP::ARMCPModifier Modifier) {
  switch (Modifier) {
  case ARMCP::no_modifier:
    return MCSymbolRefExpr::VK_None;
  case ARMCP::TLSGD:
    return MCSymbolRefExpr::VK_TLSGD;
  case ARMCP::TPOFF:
    return MCSymbolRefExpr::VK_TPOFF;
  case ARMCP::GOTTPOFF:
    return MCSymbolRefExpr::VK_GOTTPOFF;
  case ARMCP::SBREL:
    return MCSymbolRefExpr::VK_ARM_SBREL;
  case ARMCP::GOT_PREL:
    return MCSymbolRefExpr::VK_ARM_GOT_PREL;
  case ARMCP::SECREL:
    return MCSymbolRefExpr::VK_SECREL;
  }
  llvm_unreachable("Invalid ARMCPModifier!");
}

MCSymbol *ARMAsmPrinter::GetARMGVSymbol(const GlobalValue *GV,
                                        unsigned char TargetFlags) {
  if (Subtarget->isTargetMachO()) {
    bool IsIndirect =
        (TargetFlags & ARMII::MO_NONLAZY) && Subtarget->isGVIndirectSymbol(GV);

    if (!IsIndirect)
      return getSymbol(GV);

    // FIXME: Remove this when Darwin transition to @GOT like syntax.
    MCSymbol *MCSym = getSymbolWithGlobalValueBase(GV, "$non_lazy_ptr");
    MachineModuleInfoMachO &MMIMachO =
      MMI->getObjFileInfo<MachineModuleInfoMachO>();
    MachineModuleInfoImpl::StubValueTy &StubSym =
        GV->isThreadLocal() ? MMIMachO.getThreadLocalGVStubEntry(MCSym)
                            : MMIMachO.getGVStubEntry(MCSym);

    if (!StubSym.getPointer())
      StubSym = MachineModuleInfoImpl::StubValueTy(getSymbol(GV),
                                                   !GV->hasInternalLinkage());
    return MCSym;
  } else if (Subtarget->isTargetCOFF()) {
    assert(Subtarget->isTargetWindows() &&
           "Windows is the only supported COFF target");

    bool IsIndirect = (TargetFlags & ARMII::MO_DLLIMPORT);
    if (!IsIndirect)
      return getSymbol(GV);

    SmallString<128> Name;
    Name = "__imp_";
    getNameWithPrefix(Name, GV);

    return OutContext.getOrCreateSymbol(Name);
  } else if (Subtarget->isTargetELF()) {
    return getSymbol(GV);
  }
  llvm_unreachable("unexpected target");
}

void ARMAsmPrinter::
EmitMachineConstantPoolValue(MachineConstantPoolValue *MCPV) {
  const DataLayout &DL = getDataLayout();
  int Size = DL.getTypeAllocSize(MCPV->getType());

  ARMConstantPoolValue *ACPV = static_cast<ARMConstantPoolValue*>(MCPV);

  MCSymbol *MCSym;
  if (ACPV->isLSDA()) {
    MCSym = getCurExceptionSym();
  } else if (ACPV->isBlockAddress()) {
    const BlockAddress *BA =
      cast<ARMConstantPoolConstant>(ACPV)->getBlockAddress();
    MCSym = GetBlockAddressSymbol(BA);
  } else if (ACPV->isGlobalValue()) {
    const GlobalValue *GV = cast<ARMConstantPoolConstant>(ACPV)->getGV();

    // On Darwin, const-pool entries may get the "FOO$non_lazy_ptr" mangling, so
    // flag the global as MO_NONLAZY.
    unsigned char TF = Subtarget->isTargetMachO() ? ARMII::MO_NONLAZY : 0;
    MCSym = GetARMGVSymbol(GV, TF);
  } else if (ACPV->isMachineBasicBlock()) {
    const MachineBasicBlock *MBB = cast<ARMConstantPoolMBB>(ACPV)->getMBB();
    MCSym = MBB->getSymbol();
  } else {
    assert(ACPV->isExtSymbol() && "unrecognized constant pool value");
    const char *Sym = cast<ARMConstantPoolSymbol>(ACPV)->getSymbol();
    MCSym = GetExternalSymbolSymbol(Sym);
  }

  // Create an MCSymbol for the reference.
  const MCExpr *Expr =
    MCSymbolRefExpr::create(MCSym, getModifierVariantKind(ACPV->getModifier()),
                            OutContext);

  if (ACPV->getPCAdjustment()) {
    MCSymbol *PCLabel =
        getPICLabel(DL.getPrivateGlobalPrefix(), getFunctionNumber(),
                    ACPV->getLabelId(), OutContext);
    const MCExpr *PCRelExpr = MCSymbolRefExpr::create(PCLabel, OutContext);
    PCRelExpr =
      MCBinaryExpr::createAdd(PCRelExpr,
                              MCConstantExpr::create(ACPV->getPCAdjustment(),
                                                     OutContext),
                              OutContext);
    if (ACPV->mustAddCurrentAddress()) {
      // We want "(<expr> - .)", but MC doesn't have a concept of the '.'
      // label, so just emit a local label end reference that instead.
      MCSymbol *DotSym = OutContext.createTempSymbol();
      OutStreamer->EmitLabel(DotSym);
      const MCExpr *DotExpr = MCSymbolRefExpr::create(DotSym, OutContext);
      PCRelExpr = MCBinaryExpr::createSub(PCRelExpr, DotExpr, OutContext);
    }
    Expr = MCBinaryExpr::createSub(Expr, PCRelExpr, OutContext);
  }
  OutStreamer->EmitValue(Expr, Size);
}

void ARMAsmPrinter::EmitJumpTableAddrs(const MachineInstr *MI) {
  const MachineOperand &MO1 = MI->getOperand(1);
  unsigned JTI = MO1.getIndex();

  // Make sure the Thumb jump table is 4-byte aligned. This will be a nop for
  // ARM mode tables.
  EmitAlignment(2);

  // Emit a label for the jump table.
  MCSymbol *JTISymbol = GetARMJTIPICJumpTableLabel(JTI);
  OutStreamer->EmitLabel(JTISymbol);

  // Mark the jump table as data-in-code.
  OutStreamer->EmitDataRegion(MCDR_DataRegionJT32);

  // Emit each entry of the table.
  const MachineJumpTableInfo *MJTI = MF->getJumpTableInfo();
  const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
  const std::vector<MachineBasicBlock*> &JTBBs = JT[JTI].MBBs;

  for (unsigned i = 0, e = JTBBs.size(); i != e; ++i) {
    MachineBasicBlock *MBB = JTBBs[i];
    // Construct an MCExpr for the entry. We want a value of the form:
    // (BasicBlockAddr - TableBeginAddr)
    //
    // For example, a table with entries jumping to basic blocks BB0 and BB1
    // would look like:
    // LJTI_0_0:
    //    .word (LBB0 - LJTI_0_0)
    //    .word (LBB1 - LJTI_0_0)
    const MCExpr *Expr = MCSymbolRefExpr::create(MBB->getSymbol(), OutContext);

    if (isPositionIndependent() || Subtarget->isROPI())
      Expr = MCBinaryExpr::createSub(Expr, MCSymbolRefExpr::create(JTISymbol,
                                                                   OutContext),
                                     OutContext);
    // If we're generating a table of Thumb addresses in static relocation
    // model, we need to add one to keep interworking correctly.
    else if (AFI->isThumbFunction())
      Expr = MCBinaryExpr::createAdd(Expr, MCConstantExpr::create(1,OutContext),
                                     OutContext);
    OutStreamer->EmitValue(Expr, 4);
  }
  // Mark the end of jump table data-in-code region.
  OutStreamer->EmitDataRegion(MCDR_DataRegionEnd);
}

void ARMAsmPrinter::EmitJumpTableInsts(const MachineInstr *MI) {
  const MachineOperand &MO1 = MI->getOperand(1);
  unsigned JTI = MO1.getIndex();

  MCSymbol *JTISymbol = GetARMJTIPICJumpTableLabel(JTI);
  OutStreamer->EmitLabel(JTISymbol);

  // Emit each entry of the table.
  const MachineJumpTableInfo *MJTI = MF->getJumpTableInfo();
  const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
  const std::vector<MachineBasicBlock*> &JTBBs = JT[JTI].MBBs;

  for (unsigned i = 0, e = JTBBs.size(); i != e; ++i) {
    MachineBasicBlock *MBB = JTBBs[i];
    const MCExpr *MBBSymbolExpr = MCSymbolRefExpr::create(MBB->getSymbol(),
                                                          OutContext);
    // If this isn't a TBB or TBH, the entries are direct branch instructions.
    EmitToStreamer(*OutStreamer, MCInstBuilder(ARM::t2B)
        .addExpr(MBBSymbolExpr)
        .addImm(ARMCC::AL)
        .addReg(0));
  }
}

void ARMAsmPrinter::EmitJumpTableTBInst(const MachineInstr *MI,
                                        unsigned OffsetWidth) {
  assert((OffsetWidth == 1 || OffsetWidth == 2) && "invalid tbb/tbh width");
  const MachineOperand &MO1 = MI->getOperand(1);
  unsigned JTI = MO1.getIndex();

  MCSymbol *JTISymbol = GetARMJTIPICJumpTableLabel(JTI);
  OutStreamer->EmitLabel(JTISymbol);

  // Emit each entry of the table.
  const MachineJumpTableInfo *MJTI = MF->getJumpTableInfo();
  const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
  const std::vector<MachineBasicBlock*> &JTBBs = JT[JTI].MBBs;

  // Mark the jump table as data-in-code.
  OutStreamer->EmitDataRegion(OffsetWidth == 1 ? MCDR_DataRegionJT8
                                               : MCDR_DataRegionJT16);

  for (auto MBB : JTBBs) {
    const MCExpr *MBBSymbolExpr = MCSymbolRefExpr::create(MBB->getSymbol(),
                                                          OutContext);
    // Otherwise it's an offset from the dispatch instruction. Construct an
    // MCExpr for the entry. We want a value of the form:
    // (BasicBlockAddr - TBBInstAddr + 4) / 2
    //
    // For example, a TBB table with entries jumping to basic blocks BB0 and BB1
    // would look like:
    // LJTI_0_0:
    //    .byte (LBB0 - (LCPI0_0 + 4)) / 2
    //    .byte (LBB1 - (LCPI0_0 + 4)) / 2
    // where LCPI0_0 is a label defined just before the TBB instruction using
    // this table.
    MCSymbol *TBInstPC = GetCPISymbol(MI->getOperand(0).getImm());
    const MCExpr *Expr = MCBinaryExpr::createAdd(
        MCSymbolRefExpr::create(TBInstPC, OutContext),
        MCConstantExpr::create(4, OutContext), OutContext);
    Expr = MCBinaryExpr::createSub(MBBSymbolExpr, Expr, OutContext);
    Expr = MCBinaryExpr::createDiv(Expr, MCConstantExpr::create(2, OutContext),
                                   OutContext);
    OutStreamer->EmitValue(Expr, OffsetWidth);
  }
  // Mark the end of jump table data-in-code region. 32-bit offsets use
  // actual branch instructions here, so we don't mark those as a data-region
  // at all.
  OutStreamer->EmitDataRegion(MCDR_DataRegionEnd);

  // Make sure the next instruction is 2-byte aligned.
  EmitAlignment(1);
}

void ARMAsmPrinter::EmitUnwindingInstruction(const MachineInstr *MI) {
  assert(MI->getFlag(MachineInstr::FrameSetup) &&
      "Only instruction which are involved into frame setup code are allowed");

  MCTargetStreamer &TS = *OutStreamer->getTargetStreamer();
  ARMTargetStreamer &ATS = static_cast<ARMTargetStreamer &>(TS);
  const MachineFunction &MF = *MI->getParent()->getParent();
  const TargetRegisterInfo *RegInfo = MF.getSubtarget().getRegisterInfo();
  const ARMFunctionInfo &AFI = *MF.getInfo<ARMFunctionInfo>();

  unsigned FramePtr = RegInfo->getFrameRegister(MF);
  unsigned Opc = MI->getOpcode();
  unsigned SrcReg, DstReg;

  if (Opc == ARM::tPUSH || Opc == ARM::tLDRpci) {
    // Two special cases:
    // 1) tPUSH does not have src/dst regs.
    // 2) for Thumb1 code we sometimes materialize the constant via constpool
    // load. Yes, this is pretty fragile, but for now I don't see better
    // way... :(
    SrcReg = DstReg = ARM::SP;
  } else {
    SrcReg = MI->getOperand(1).getReg();
    DstReg = MI->getOperand(0).getReg();
  }

  // Try to figure out the unwinding opcode out of src / dst regs.
  if (MI->mayStore()) {
    // Register saves.
    assert(DstReg == ARM::SP &&
           "Only stack pointer as a destination reg is supported");

    SmallVector<unsigned, 4> RegList;
    // Skip src & dst reg, and pred ops.
    unsigned StartOp = 2 + 2;
    // Use all the operands.
    unsigned NumOffset = 0;

    switch (Opc) {
    default:
      MI->dump();
      llvm_unreachable("Unsupported opcode for unwinding information");
    case ARM::tPUSH:
      // Special case here: no src & dst reg, but two extra imp ops.
      StartOp = 2; NumOffset = 2;
    case ARM::STMDB_UPD:
    case ARM::t2STMDB_UPD:
    case ARM::VSTMDDB_UPD:
      assert(SrcReg == ARM::SP &&
             "Only stack pointer as a source reg is supported");
      for (unsigned i = StartOp, NumOps = MI->getNumOperands() - NumOffset;
           i != NumOps; ++i) {
        const MachineOperand &MO = MI->getOperand(i);
        // Actually, there should never be any impdef stuff here. Skip it
        // temporary to workaround PR11902.
        if (MO.isImplicit())
          continue;
        RegList.push_back(MO.getReg());
      }
      break;
    case ARM::STR_PRE_IMM:
    case ARM::STR_PRE_REG:
    case ARM::t2STR_PRE:
      assert(MI->getOperand(2).getReg() == ARM::SP &&
             "Only stack pointer as a source reg is supported");
      RegList.push_back(SrcReg);
      break;
    }
    if (MAI->getExceptionHandlingType() == ExceptionHandling::ARM)
      ATS.emitRegSave(RegList, Opc == ARM::VSTMDDB_UPD);
  } else {
    // Changes of stack / frame pointer.
    if (SrcReg == ARM::SP) {
      int64_t Offset = 0;
      switch (Opc) {
      default:
        MI->dump();
        llvm_unreachable("Unsupported opcode for unwinding information");
      case ARM::MOVr:
      case ARM::tMOVr:
        Offset = 0;
        break;
      case ARM::ADDri:
      case ARM::t2ADDri:
        Offset = -MI->getOperand(2).getImm();
        break;
      case ARM::SUBri:
      case ARM::t2SUBri:
        Offset = MI->getOperand(2).getImm();
        break;
      case ARM::tSUBspi:
        Offset = MI->getOperand(2).getImm()*4;
        break;
      case ARM::tADDspi:
      case ARM::tADDrSPi:
        Offset = -MI->getOperand(2).getImm()*4;
        break;
      case ARM::tLDRpci: {
        // Grab the constpool index and check, whether it corresponds to
        // original or cloned constpool entry.
        unsigned CPI = MI->getOperand(1).getIndex();
        const MachineConstantPool *MCP = MF.getConstantPool();
        if (CPI >= MCP->getConstants().size())
          CPI = AFI.getOriginalCPIdx(CPI);
        assert(CPI != -1U && "Invalid constpool index");

        // Derive the actual offset.
        const MachineConstantPoolEntry &CPE = MCP->getConstants()[CPI];
        assert(!CPE.isMachineConstantPoolEntry() && "Invalid constpool entry");
        // FIXME: Check for user, it should be "add" instruction!
        Offset = -cast<ConstantInt>(CPE.Val.ConstVal)->getSExtValue();
        break;
      }
      }

      if (MAI->getExceptionHandlingType() == ExceptionHandling::ARM) {
        if (DstReg == FramePtr && FramePtr != ARM::SP)
          // Set-up of the frame pointer. Positive values correspond to "add"
          // instruction.
          ATS.emitSetFP(FramePtr, ARM::SP, -Offset);
        else if (DstReg == ARM::SP) {
          // Change of SP by an offset. Positive values correspond to "sub"
          // instruction.
          ATS.emitPad(Offset);
        } else {
          // Move of SP to a register.  Positive values correspond to an "add"
          // instruction.
          ATS.emitMovSP(DstReg, -Offset);
        }
      }
    } else if (DstReg == ARM::SP) {
      MI->dump();
      llvm_unreachable("Unsupported opcode for unwinding information");
    }
    else {
      MI->dump();
      llvm_unreachable("Unsupported opcode for unwinding information");
    }
  }
}

// Simple pseudo-instructions have their lowering (with expansion to real
// instructions) auto-generated.
#include "ARMGenMCPseudoLowering.inc"

void ARMAsmPrinter::EmitInstruction(const MachineInstr *MI) {
  const DataLayout &DL = getDataLayout();
  MCTargetStreamer &TS = *OutStreamer->getTargetStreamer();
  ARMTargetStreamer &ATS = static_cast<ARMTargetStreamer &>(TS);

  // If we just ended a constant pool, mark it as such.
  if (InConstantPool && MI->getOpcode() != ARM::CONSTPOOL_ENTRY) {
    OutStreamer->EmitDataRegion(MCDR_DataRegionEnd);
    InConstantPool = false;
  }

  // Emit unwinding stuff for frame-related instructions
  if (Subtarget->isTargetEHABICompatible() &&
       MI->getFlag(MachineInstr::FrameSetup))
    EmitUnwindingInstruction(MI);

  // Do any auto-generated pseudo lowerings.
  if (emitPseudoExpansionLowering(*OutStreamer, MI))
    return;

  assert(!convertAddSubFlagsOpcode(MI->getOpcode()) &&
         "Pseudo flag setting opcode should be expanded early");

  // Check for manual lowerings.
  unsigned Opc = MI->getOpcode();
  switch (Opc) {
  case ARM::t2MOVi32imm: llvm_unreachable("Should be lowered by thumb2it pass");
  case ARM::DBG_VALUE: llvm_unreachable("Should be handled by generic printing");
  case ARM::LEApcrel:
  case ARM::tLEApcrel:
  case ARM::t2LEApcrel: {
    // FIXME: Need to also handle globals and externals
    MCSymbol *CPISymbol = GetCPISymbol(MI->getOperand(1).getIndex());
    EmitToStreamer(*OutStreamer, MCInstBuilder(MI->getOpcode() ==
                                               ARM::t2LEApcrel ? ARM::t2ADR
                  : (MI->getOpcode() == ARM::tLEApcrel ? ARM::tADR
                     : ARM::ADR))
      .addReg(MI->getOperand(0).getReg())
      .addExpr(MCSymbolRefExpr::create(CPISymbol, OutContext))
      // Add predicate operands.
      .addImm(MI->getOperand(2).getImm())
      .addReg(MI->getOperand(3).getReg()));
    return;
  }
  case ARM::LEApcrelJT:
  case ARM::tLEApcrelJT:
  case ARM::t2LEApcrelJT: {
    MCSymbol *JTIPICSymbol =
      GetARMJTIPICJumpTableLabel(MI->getOperand(1).getIndex());
    EmitToStreamer(*OutStreamer, MCInstBuilder(MI->getOpcode() ==
                                               ARM::t2LEApcrelJT ? ARM::t2ADR
                  : (MI->getOpcode() == ARM::tLEApcrelJT ? ARM::tADR
                     : ARM::ADR))
      .addReg(MI->getOperand(0).getReg())
      .addExpr(MCSymbolRefExpr::create(JTIPICSymbol, OutContext))
      // Add predicate operands.
      .addImm(MI->getOperand(2).getImm())
      .addReg(MI->getOperand(3).getReg()));
    return;
  }
  // Darwin call instructions are just normal call instructions with different
  // clobber semantics (they clobber R9).
  case ARM::BX_CALL: {
    EmitToStreamer(*OutStreamer, MCInstBuilder(ARM::MOVr)
      .addReg(ARM::LR)
      .addReg(ARM::PC)
      // Add predicate operands.
      .addImm(ARMCC::AL)
      .addReg(0)
      // Add 's' bit operand (always reg0 for this)
      .addReg(0));

    EmitToStreamer(*OutStreamer, MCInstBuilder(ARM::BX)
      .addReg(MI->getOperand(0).getReg()));
    return;
  }
  case ARM::tBX_CALL: {
    if (Subtarget->hasV5TOps())
      llvm_unreachable("Expected BLX to be selected for v5t+");

    // On ARM v4t, when doing a call from thumb mode, we need to ensure
    // that the saved lr has its LSB set correctly (the arch doesn't
    // have blx).
    // So here we generate a bl to a small jump pad that does bx rN.
    // The jump pads are emitted after the function body.

    unsigned TReg = MI->getOperand(0).getReg();
    MCSymbol *TRegSym = nullptr;
    for (unsigned i = 0, e = ThumbIndirectPads.size(); i < e; i++) {
      if (ThumbIndirectPads[i].first == TReg) {
        TRegSym = ThumbIndirectPads[i].second;
        break;
      }
    }

    if (!TRegSym) {
      TRegSym = OutContext.createTempSymbol();
      ThumbIndirectPads.push_back(std::make_pair(TReg, TRegSym));
    }

    // Create a link-saving branch to the Reg Indirect Jump Pad.
    EmitToStreamer(*OutStreamer, MCInstBuilder(ARM::tBL)
        // Predicate comes first here.
        .addImm(ARMCC::AL).addReg(0)
        .addExpr(MCSymbolRefExpr::create(TRegSym, OutContext)));
    return;
  }
  case ARM::BMOVPCRX_CALL: {
    EmitToStreamer(*OutStreamer, MCInstBuilder(ARM::MOVr)
      .addReg(ARM::LR)
      .addReg(ARM::PC)
      // Add predicate operands.
      .addImm(ARMCC::AL)
      .addReg(0)
      // Add 's' bit operand (always reg0 for this)
      .addReg(0));

    EmitToStreamer(*OutStreamer, MCInstBuilder(ARM::MOVr)
      .addReg(ARM::PC)
      .addReg(MI->getOperand(0).getReg())
      // Add predicate operands.
      .addImm(ARMCC::AL)
      .addReg(0)
      // Add 's' bit operand (always reg0 for this)
      .addReg(0));
    return;
  }
  case ARM::BMOVPCB_CALL: {
    EmitToStreamer(*OutStreamer, MCInstBuilder(ARM::MOVr)
      .addReg(ARM::LR)
      .addReg(ARM::PC)
      // Add predicate operands.
      .addImm(ARMCC::AL)
      .addReg(0)
      // Add 's' bit operand (always reg0 for this)
      .addReg(0));

    const MachineOperand &Op = MI->getOperand(0);
    const GlobalValue *GV = Op.getGlobal();
    const unsigned TF = Op.getTargetFlags();
    MCSymbol *GVSym = GetARMGVSymbol(GV, TF);
    const MCExpr *GVSymExpr = MCSymbolRefExpr::create(GVSym, OutContext);
    EmitToStreamer(*OutStreamer, MCInstBuilder(ARM::Bcc)
      .addExpr(GVSymExpr)
      // Add predicate operands.
      .addImm(ARMCC::AL)
      .addReg(0));
    return;
  }
  case ARM::MOVi16_ga_pcrel:
  case ARM::t2MOVi16_ga_pcrel: {
    MCInst TmpInst;
    TmpInst.setOpcode(Opc == ARM::MOVi16_ga_pcrel? ARM::MOVi16 : ARM::t2MOVi16);
    TmpInst.addOperand(MCOperand::createReg(MI->getOperand(0).getReg()));

    unsigned TF = MI->getOperand(1).getTargetFlags();
    const GlobalValue *GV = MI->getOperand(1).getGlobal();
    MCSymbol *GVSym = GetARMGVSymbol(GV, TF);
    const MCExpr *GVSymExpr = MCSymbolRefExpr::create(GVSym, OutContext);

    MCSymbol *LabelSym =
        getPICLabel(DL.getPrivateGlobalPrefix(), getFunctionNumber(),
                    MI->getOperand(2).getImm(), OutContext);
    const MCExpr *LabelSymExpr= MCSymbolRefExpr::create(LabelSym, OutContext);
    unsigned PCAdj = (Opc == ARM::MOVi16_ga_pcrel) ? 8 : 4;
    const MCExpr *PCRelExpr =
      ARMMCExpr::createLower16(MCBinaryExpr::createSub(GVSymExpr,
                                      MCBinaryExpr::createAdd(LabelSymExpr,
                                      MCConstantExpr::create(PCAdj, OutContext),
                                      OutContext), OutContext), OutContext);
      TmpInst.addOperand(MCOperand::createExpr(PCRelExpr));

    // Add predicate operands.
    TmpInst.addOperand(MCOperand::createImm(ARMCC::AL));
    TmpInst.addOperand(MCOperand::createReg(0));
    // Add 's' bit operand (always reg0 for this)
    TmpInst.addOperand(MCOperand::createReg(0));
    EmitToStreamer(*OutStreamer, TmpInst);
    return;
  }
  case ARM::MOVTi16_ga_pcrel:
  case ARM::t2MOVTi16_ga_pcrel: {
    MCInst TmpInst;
    TmpInst.setOpcode(Opc == ARM::MOVTi16_ga_pcrel
                      ? ARM::MOVTi16 : ARM::t2MOVTi16);
    TmpInst.addOperand(MCOperand::createReg(MI->getOperand(0).getReg()));
    TmpInst.addOperand(MCOperand::createReg(MI->getOperand(1).getReg()));

    unsigned TF = MI->getOperand(2).getTargetFlags();
    const GlobalValue *GV = MI->getOperand(2).getGlobal();
    MCSymbol *GVSym = GetARMGVSymbol(GV, TF);
    const MCExpr *GVSymExpr = MCSymbolRefExpr::create(GVSym, OutContext);

    MCSymbol *LabelSym =
        getPICLabel(DL.getPrivateGlobalPrefix(), getFunctionNumber(),
                    MI->getOperand(3).getImm(), OutContext);
    const MCExpr *LabelSymExpr= MCSymbolRefExpr::create(LabelSym, OutContext);
    unsigned PCAdj = (Opc == ARM::MOVTi16_ga_pcrel) ? 8 : 4;
    const MCExpr *PCRelExpr =
        ARMMCExpr::createUpper16(MCBinaryExpr::createSub(GVSymExpr,
                                   MCBinaryExpr::createAdd(LabelSymExpr,
                                      MCConstantExpr::create(PCAdj, OutContext),
                                          OutContext), OutContext), OutContext);
      TmpInst.addOperand(MCOperand::createExpr(PCRelExpr));
    // Add predicate operands.
    TmpInst.addOperand(MCOperand::createImm(ARMCC::AL));
    TmpInst.addOperand(MCOperand::createReg(0));
    // Add 's' bit operand (always reg0 for this)
    TmpInst.addOperand(MCOperand::createReg(0));
    EmitToStreamer(*OutStreamer, TmpInst);
    return;
  }
  case ARM::tPICADD: {
    // This is a pseudo op for a label + instruction sequence, which looks like:
    // LPC0:
    //     add r0, pc
    // This adds the address of LPC0 to r0.

    // Emit the label.
    OutStreamer->EmitLabel(getPICLabel(DL.getPrivateGlobalPrefix(),
                                       getFunctionNumber(),
                                       MI->getOperand(2).getImm(), OutContext));

    // Form and emit the add.
    EmitToStreamer(*OutStreamer, MCInstBuilder(ARM::tADDhirr)
      .addReg(MI->getOperand(0).getReg())
      .addReg(MI->getOperand(0).getReg())
      .addReg(ARM::PC)
      // Add predicate operands.
      .addImm(ARMCC::AL)
      .addReg(0));
    return;
  }
  case ARM::PICADD: {
    // This is a pseudo op for a label + instruction sequence, which looks like:
    // LPC0:
    //     add r0, pc, r0
    // This adds the address of LPC0 to r0.

    // Emit the label.
    OutStreamer->EmitLabel(getPICLabel(DL.getPrivateGlobalPrefix(),
                                       getFunctionNumber(),
                                       MI->getOperand(2).getImm(), OutContext));

    // Form and emit the add.
    EmitToStreamer(*OutStreamer, MCInstBuilder(ARM::ADDrr)
      .addReg(MI->getOperand(0).getReg())
      .addReg(ARM::PC)
      .addReg(MI->getOperand(1).getReg())
      // Add predicate operands.
      .addImm(MI->getOperand(3).getImm())
      .addReg(MI->getOperand(4).getReg())
      // Add 's' bit operand (always reg0 for this)
      .addReg(0));
    return;
  }
  case ARM::PICSTR:
  case ARM::PICSTRB:
  case ARM::PICSTRH:
  case ARM::PICLDR:
  case ARM::PICLDRB:
  case ARM::PICLDRH:
  case ARM::PICLDRSB:
  case ARM::PICLDRSH: {
    // This is a pseudo op for a label + instruction sequence, which looks like:
    // LPC0:
    //     OP r0, [pc, r0]
    // The LCP0 label is referenced by a constant pool entry in order to get
    // a PC-relative address at the ldr instruction.

    // Emit the label.
    OutStreamer->EmitLabel(getPICLabel(DL.getPrivateGlobalPrefix(),
                                       getFunctionNumber(),
                                       MI->getOperand(2).getImm(), OutContext));

    // Form and emit the load
    unsigned Opcode;
    switch (MI->getOpcode()) {
    default:
      llvm_unreachable("Unexpected opcode!");
    case ARM::PICSTR:   Opcode = ARM::STRrs; break;
    case ARM::PICSTRB:  Opcode = ARM::STRBrs; break;
    case ARM::PICSTRH:  Opcode = ARM::STRH; break;
    case ARM::PICLDR:   Opcode = ARM::LDRrs; break;
    case ARM::PICLDRB:  Opcode = ARM::LDRBrs; break;
    case ARM::PICLDRH:  Opcode = ARM::LDRH; break;
    case ARM::PICLDRSB: Opcode = ARM::LDRSB; break;
    case ARM::PICLDRSH: Opcode = ARM::LDRSH; break;
    }
    EmitToStreamer(*OutStreamer, MCInstBuilder(Opcode)
      .addReg(MI->getOperand(0).getReg())
      .addReg(ARM::PC)
      .addReg(MI->getOperand(1).getReg())
      .addImm(0)
      // Add predicate operands.
      .addImm(MI->getOperand(3).getImm())
      .addReg(MI->getOperand(4).getReg()));

    return;
  }
  case ARM::CONSTPOOL_ENTRY: {
    /// CONSTPOOL_ENTRY - This instruction represents a floating constant pool
    /// in the function.  The first operand is the ID# for this instruction, the
    /// second is the index into the MachineConstantPool that this is, the third
    /// is the size in bytes of this constant pool entry.
    /// The required alignment is specified on the basic block holding this MI.
    unsigned LabelId = (unsigned)MI->getOperand(0).getImm();
    unsigned CPIdx   = (unsigned)MI->getOperand(1).getIndex();

    // If this is the first entry of the pool, mark it.
    if (!InConstantPool) {
      OutStreamer->EmitDataRegion(MCDR_DataRegion);
      InConstantPool = true;
    }

    OutStreamer->EmitLabel(GetCPISymbol(LabelId));

    const MachineConstantPoolEntry &MCPE = MCP->getConstants()[CPIdx];
    if (MCPE.isMachineConstantPoolEntry())
      EmitMachineConstantPoolValue(MCPE.Val.MachineCPVal);
    else
      EmitGlobalConstant(DL, MCPE.Val.ConstVal);
    return;
  }
  case ARM::JUMPTABLE_ADDRS:
    EmitJumpTableAddrs(MI);
    return;
  case ARM::JUMPTABLE_INSTS:
    EmitJumpTableInsts(MI);
    return;
  case ARM::JUMPTABLE_TBB:
  case ARM::JUMPTABLE_TBH:
    EmitJumpTableTBInst(MI, MI->getOpcode() == ARM::JUMPTABLE_TBB ? 1 : 2);
    return;
  case ARM::t2BR_JT: {
    // Lower and emit the instruction itself, then the jump table following it.
    EmitToStreamer(*OutStreamer, MCInstBuilder(ARM::tMOVr)
      .addReg(ARM::PC)
      .addReg(MI->getOperand(0).getReg())
      // Add predicate operands.
      .addImm(ARMCC::AL)
      .addReg(0));
    return;
  }
  case ARM::t2TBB_JT:
  case ARM::t2TBH_JT: {
    unsigned Opc = MI->getOpcode() == ARM::t2TBB_JT ? ARM::t2TBB : ARM::t2TBH;
    // Lower and emit the PC label, then the instruction itself.
    OutStreamer->EmitLabel(GetCPISymbol(MI->getOperand(3).getImm()));
    EmitToStreamer(*OutStreamer, MCInstBuilder(Opc)
                                     .addReg(MI->getOperand(0).getReg())
                                     .addReg(MI->getOperand(1).getReg())
                                     // Add predicate operands.
                                     .addImm(ARMCC::AL)
                                     .addReg(0));
    return;
  }
  case ARM::tBR_JTr:
  case ARM::BR_JTr: {
    // Lower and emit the instruction itself, then the jump table following it.
    // mov pc, target
    MCInst TmpInst;
    unsigned Opc = MI->getOpcode() == ARM::BR_JTr ?
      ARM::MOVr : ARM::tMOVr;
    TmpInst.setOpcode(Opc);
    TmpInst.addOperand(MCOperand::createReg(ARM::PC));
    TmpInst.addOperand(MCOperand::createReg(MI->getOperand(0).getReg()));
    // Add predicate operands.
    TmpInst.addOperand(MCOperand::createImm(ARMCC::AL));
    TmpInst.addOperand(MCOperand::createReg(0));
    // Add 's' bit operand (always reg0 for this)
    if (Opc == ARM::MOVr)
      TmpInst.addOperand(MCOperand::createReg(0));
    EmitToStreamer(*OutStreamer, TmpInst);
    return;
  }
  case ARM::BR_JTm: {
    // Lower and emit the instruction itself, then the jump table following it.
    // ldr pc, target
    MCInst TmpInst;
    if (MI->getOperand(1).getReg() == 0) {
      // literal offset
      TmpInst.setOpcode(ARM::LDRi12);
      TmpInst.addOperand(MCOperand::createReg(ARM::PC));
      TmpInst.addOperand(MCOperand::createReg(MI->getOperand(0).getReg()));
      TmpInst.addOperand(MCOperand::createImm(MI->getOperand(2).getImm()));
    } else {
      TmpInst.setOpcode(ARM::LDRrs);
      TmpInst.addOperand(MCOperand::createReg(ARM::PC));
      TmpInst.addOperand(MCOperand::createReg(MI->getOperand(0).getReg()));
      TmpInst.addOperand(MCOperand::createReg(MI->getOperand(1).getReg()));
      TmpInst.addOperand(MCOperand::createImm(0));
    }
    // Add predicate operands.
    TmpInst.addOperand(MCOperand::createImm(ARMCC::AL));
    TmpInst.addOperand(MCOperand::createReg(0));
    EmitToStreamer(*OutStreamer, TmpInst);
    return;
  }
  case ARM::BR_JTadd: {
    // Lower and emit the instruction itself, then the jump table following it.
    // add pc, target, idx
    EmitToStreamer(*OutStreamer, MCInstBuilder(ARM::ADDrr)
      .addReg(ARM::PC)
      .addReg(MI->getOperand(0).getReg())
      .addReg(MI->getOperand(1).getReg())
      // Add predicate operands.
      .addImm(ARMCC::AL)
      .addReg(0)
      // Add 's' bit operand (always reg0 for this)
      .addReg(0));
    return;
  }
  case ARM::SPACE:
    OutStreamer->EmitZeros(MI->getOperand(1).getImm());
    return;
  case ARM::TRAP: {
    // Non-Darwin binutils don't yet support the "trap" mnemonic.
    // FIXME: Remove this special case when they do.
    if (!Subtarget->isTargetMachO()) {
      uint32_t Val = 0xe7ffdefeUL;
      OutStreamer->AddComment("trap");
      ATS.emitInst(Val);
      return;
    }
    break;
  }
  case ARM::TRAPNaCl: {
    uint32_t Val = 0xe7fedef0UL;
    OutStreamer->AddComment("trap");
    ATS.emitInst(Val);
    return;
  }
  case ARM::tTRAP: {
    // Non-Darwin binutils don't yet support the "trap" mnemonic.
    // FIXME: Remove this special case when they do.
    if (!Subtarget->isTargetMachO()) {
      uint16_t Val = 0xdefe;
      OutStreamer->AddComment("trap");
      ATS.emitInst(Val, 'n');
      return;
    }
    break;
  }
  case ARM::t2Int_eh_sjlj_setjmp:
  case ARM::t2Int_eh_sjlj_setjmp_nofp:
  case ARM::tInt_eh_sjlj_setjmp: {
    // Two incoming args: GPR:$src, GPR:$val
    // mov $val, pc
    // adds $val, #7
    // str $val, [$src, #4]
    // movs r0, #0
    // b LSJLJEH
    // movs r0, #1
    // LSJLJEH:
    unsigned SrcReg = MI->getOperand(0).getReg();
    unsigned ValReg = MI->getOperand(1).getReg();
    MCSymbol *Label = OutContext.createTempSymbol("SJLJEH", false, true);
    OutStreamer->AddComment("eh_setjmp begin");
    EmitToStreamer(*OutStreamer, MCInstBuilder(ARM::tMOVr)
      .addReg(ValReg)
      .addReg(ARM::PC)
      // Predicate.
      .addImm(ARMCC::AL)
      .addReg(0));

    EmitToStreamer(*OutStreamer, MCInstBuilder(ARM::tADDi3)
      .addReg(ValReg)
      // 's' bit operand
      .addReg(ARM::CPSR)
      .addReg(ValReg)
      .addImm(7)
      // Predicate.
      .addImm(ARMCC::AL)
      .addReg(0));

    EmitToStreamer(*OutStreamer, MCInstBuilder(ARM::tSTRi)
      .addReg(ValReg)
      .addReg(SrcReg)
      // The offset immediate is #4. The operand value is scaled by 4 for the
      // tSTR instruction.
      .addImm(1)
      // Predicate.
      .addImm(ARMCC::AL)
      .addReg(0));

    EmitToStreamer(*OutStreamer, MCInstBuilder(ARM::tMOVi8)
      .addReg(ARM::R0)
      .addReg(ARM::CPSR)
      .addImm(0)
      // Predicate.
      .addImm(ARMCC::AL)
      .addReg(0));

    const MCExpr *SymbolExpr = MCSymbolRefExpr::create(Label, OutContext);
    EmitToStreamer(*OutStreamer, MCInstBuilder(ARM::tB)
      .addExpr(SymbolExpr)
      .addImm(ARMCC::AL)
      .addReg(0));

    OutStreamer->AddComment("eh_setjmp end");
    EmitToStreamer(*OutStreamer, MCInstBuilder(ARM::tMOVi8)
      .addReg(ARM::R0)
      .addReg(ARM::CPSR)
      .addImm(1)
      // Predicate.
      .addImm(ARMCC::AL)
      .addReg(0));

    OutStreamer->EmitLabel(Label);
    return;
  }

  case ARM::Int_eh_sjlj_setjmp_nofp:
  case ARM::Int_eh_sjlj_setjmp: {
    // Two incoming args: GPR:$src, GPR:$val
    // add $val, pc, #8
    // str $val, [$src, #+4]
    // mov r0, #0
    // add pc, pc, #0
    // mov r0, #1
    unsigned SrcReg = MI->getOperand(0).getReg();
    unsigned ValReg = MI->getOperand(1).getReg();

    OutStreamer->AddComment("eh_setjmp begin");
    EmitToStreamer(*OutStreamer, MCInstBuilder(ARM::ADDri)
      .addReg(ValReg)
      .addReg(ARM::PC)
      .addImm(8)
      // Predicate.
      .addImm(ARMCC::AL)
      .addReg(0)
      // 's' bit operand (always reg0 for this).
      .addReg(0));

    EmitToStreamer(*OutStreamer, MCInstBuilder(ARM::STRi12)
      .addReg(ValReg)
      .addReg(SrcReg)
      .addImm(4)
      // Predicate.
      .addImm(ARMCC::AL)
      .addReg(0));

    EmitToStreamer(*OutStreamer, MCInstBuilder(ARM::MOVi)
      .addReg(ARM::R0)
      .addImm(0)
      // Predicate.
      .addImm(ARMCC::AL)
      .addReg(0)
      // 's' bit operand (always reg0 for this).
      .addReg(0));

    EmitToStreamer(*OutStreamer, MCInstBuilder(ARM::ADDri)
      .addReg(ARM::PC)
      .addReg(ARM::PC)
      .addImm(0)
      // Predicate.
      .addImm(ARMCC::AL)
      .addReg(0)
      // 's' bit operand (always reg0 for this).
      .addReg(0));

    OutStreamer->AddComment("eh_setjmp end");
    EmitToStreamer(*OutStreamer, MCInstBuilder(ARM::MOVi)
      .addReg(ARM::R0)
      .addImm(1)
      // Predicate.
      .addImm(ARMCC::AL)
      .addReg(0)
      // 's' bit operand (always reg0 for this).
      .addReg(0));
    return;
  }
  case ARM::Int_eh_sjlj_longjmp: {
    // ldr sp, [$src, #8]
    // ldr $scratch, [$src, #4]
    // ldr r7, [$src]
    // bx $scratch
    unsigned SrcReg = MI->getOperand(0).getReg();
    unsigned ScratchReg = MI->getOperand(1).getReg();
    EmitToStreamer(*OutStreamer, MCInstBuilder(ARM::LDRi12)
      .addReg(ARM::SP)
      .addReg(SrcReg)
      .addImm(8)
      // Predicate.
      .addImm(ARMCC::AL)
      .addReg(0));

    EmitToStreamer(*OutStreamer, MCInstBuilder(ARM::LDRi12)
      .addReg(ScratchReg)
      .addReg(SrcReg)
      .addImm(4)
      // Predicate.
      .addImm(ARMCC::AL)
      .addReg(0));

    EmitToStreamer(*OutStreamer, MCInstBuilder(ARM::LDRi12)
      .addReg(ARM::R7)
      .addReg(SrcReg)
      .addImm(0)
      // Predicate.
      .addImm(ARMCC::AL)
      .addReg(0));

    EmitToStreamer(*OutStreamer, MCInstBuilder(ARM::BX)
      .addReg(ScratchReg)
      // Predicate.
      .addImm(ARMCC::AL)
      .addReg(0));
    return;
  }
  case ARM::tInt_eh_sjlj_longjmp: {
    // ldr $scratch, [$src, #8]
    // mov sp, $scratch
    // ldr $scratch, [$src, #4]
    // ldr r7, [$src]
    // bx $scratch
    unsigned SrcReg = MI->getOperand(0).getReg();
    unsigned ScratchReg = MI->getOperand(1).getReg();

    EmitToStreamer(*OutStreamer, MCInstBuilder(ARM::tLDRi)
      .addReg(ScratchReg)
      .addReg(SrcReg)
      // The offset immediate is #8. The operand value is scaled by 4 for the
      // tLDR instruction.
      .addImm(2)
      // Predicate.
      .addImm(ARMCC::AL)
      .addReg(0));

    EmitToStreamer(*OutStreamer, MCInstBuilder(ARM::tMOVr)
      .addReg(ARM::SP)
      .addReg(ScratchReg)
      // Predicate.
      .addImm(ARMCC::AL)
      .addReg(0));

    EmitToStreamer(*OutStreamer, MCInstBuilder(ARM::tLDRi)
      .addReg(ScratchReg)
      .addReg(SrcReg)
      .addImm(1)
      // Predicate.
      .addImm(ARMCC::AL)
      .addReg(0));

    EmitToStreamer(*OutStreamer, MCInstBuilder(ARM::tLDRi)
      .addReg(ARM::R7)
      .addReg(SrcReg)
      .addImm(0)
      // Predicate.
      .addImm(ARMCC::AL)
      .addReg(0));

    EmitToStreamer(*OutStreamer, MCInstBuilder(ARM::tBX)
      .addReg(ScratchReg)
      // Predicate.
      .addImm(ARMCC::AL)
      .addReg(0));
    return;
  }
  case ARM::tInt_WIN_eh_sjlj_longjmp: {
    // ldr.w r11, [$src, #0]
    // ldr.w  sp, [$src, #8]
    // ldr.w  pc, [$src, #4]

    unsigned SrcReg = MI->getOperand(0).getReg();

    EmitToStreamer(*OutStreamer, MCInstBuilder(ARM::t2LDRi12)
                                     .addReg(ARM::R11)
                                     .addReg(SrcReg)
                                     .addImm(0)
                                     // Predicate
                                     .addImm(ARMCC::AL)
                                     .addReg(0));
    EmitToStreamer(*OutStreamer, MCInstBuilder(ARM::t2LDRi12)
                                     .addReg(ARM::SP)
                                     .addReg(SrcReg)
                                     .addImm(8)
                                     // Predicate
                                     .addImm(ARMCC::AL)
                                     .addReg(0));
    EmitToStreamer(*OutStreamer, MCInstBuilder(ARM::t2LDRi12)
                                     .addReg(ARM::PC)
                                     .addReg(SrcReg)
                                     .addImm(4)
                                     // Predicate
                                     .addImm(ARMCC::AL)
                                     .addReg(0));
    return;
  }
  }

  MCInst TmpInst;
  LowerARMMachineInstrToMCInst(MI, TmpInst, *this);

  EmitToStreamer(*OutStreamer, TmpInst);
}

//===----------------------------------------------------------------------===//
// Target Registry Stuff
//===----------------------------------------------------------------------===//

// Force static initialization.
extern "C" void LLVMInitializeARMAsmPrinter() {
  RegisterAsmPrinter<ARMAsmPrinter> X(TheARMLETarget);
  RegisterAsmPrinter<ARMAsmPrinter> Y(TheARMBETarget);
  RegisterAsmPrinter<ARMAsmPrinter> A(TheThumbLETarget);
  RegisterAsmPrinter<ARMAsmPrinter> B(TheThumbBETarget);
}
