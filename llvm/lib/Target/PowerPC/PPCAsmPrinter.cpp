//===-- PPCAsmPrinter.cpp - Print machine instrs to PowerPC assembly ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to PowerPC assembly language. This printer is
// the output mechanism used by `llc'.
//
// Documentation at http://developer.apple.com/documentation/DeveloperTools/
// Reference/Assembler/ASMIntroduction/chapter_1_section_1.html
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/PPCInstPrinter.h"
#include "MCTargetDesc/PPCMCExpr.h"
#include "MCTargetDesc/PPCMCTargetDesc.h"
#include "MCTargetDesc/PPCPredicates.h"
#include "PPC.h"
#include "PPCInstrInfo.h"
#include "PPCMachineFunctionInfo.h"
#include "PPCSubtarget.h"
#include "PPCTargetMachine.h"
#include "PPCTargetStreamer.h"
#include "TargetInfo/PowerPCTargetInfo.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/StackMaps.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDirectives.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCSectionXCOFF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/MC/MCSymbolXCOFF.h"
#include "llvm/MC/SectionKind.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <new>

using namespace llvm;

#define DEBUG_TYPE "asmprinter"

namespace {

class PPCAsmPrinter : public AsmPrinter {
protected:
  MapVector<const MCSymbol *, MCSymbol *> TOC;
  const PPCSubtarget *Subtarget = nullptr;
  StackMaps SM;

public:
  explicit PPCAsmPrinter(TargetMachine &TM,
                         std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer)), SM(*this) {}

  StringRef getPassName() const override { return "PowerPC Assembly Printer"; }

  MCSymbol *lookUpOrCreateTOCEntry(const MCSymbol *Sym);

  bool doInitialization(Module &M) override {
    if (!TOC.empty())
      TOC.clear();
    return AsmPrinter::doInitialization(M);
  }

  void emitInstruction(const MachineInstr *MI) override;

  /// This function is for PrintAsmOperand and PrintAsmMemoryOperand,
  /// invoked by EmitMSInlineAsmStr and EmitGCCInlineAsmStr only.
  /// The \p MI would be INLINEASM ONLY.
  void printOperand(const MachineInstr *MI, unsigned OpNo, raw_ostream &O);

  void PrintSymbolOperand(const MachineOperand &MO, raw_ostream &O) override;
  bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                       const char *ExtraCode, raw_ostream &O) override;
  bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                             const char *ExtraCode, raw_ostream &O) override;

  void emitEndOfAsmFile(Module &M) override;

  void LowerSTACKMAP(StackMaps &SM, const MachineInstr &MI);
  void LowerPATCHPOINT(StackMaps &SM, const MachineInstr &MI);
  void EmitTlsCall(const MachineInstr *MI, MCSymbolRefExpr::VariantKind VK);
  bool runOnMachineFunction(MachineFunction &MF) override {
    Subtarget = &MF.getSubtarget<PPCSubtarget>();
    bool Changed = AsmPrinter::runOnMachineFunction(MF);
    emitXRayTable();
    return Changed;
  }
};

/// PPCLinuxAsmPrinter - PowerPC assembly printer, customized for Linux
class PPCLinuxAsmPrinter : public PPCAsmPrinter {
public:
  explicit PPCLinuxAsmPrinter(TargetMachine &TM,
                              std::unique_ptr<MCStreamer> Streamer)
      : PPCAsmPrinter(TM, std::move(Streamer)) {}

  StringRef getPassName() const override {
    return "Linux PPC Assembly Printer";
  }

  void emitStartOfAsmFile(Module &M) override;
  void emitEndOfAsmFile(Module &) override;

  void emitFunctionEntryLabel() override;

  void emitFunctionBodyStart() override;
  void emitFunctionBodyEnd() override;
  void emitInstruction(const MachineInstr *MI) override;
};

class PPCAIXAsmPrinter : public PPCAsmPrinter {
private:
  /// Symbols lowered from ExternalSymbolSDNodes, we will need to emit extern
  /// linkage for them in AIX.
  SmallPtrSet<MCSymbol *, 8> ExtSymSDNodeSymbols;

  /// A format indicator and unique trailing identifier to form part of the
  /// sinit/sterm function names.
  std::string FormatIndicatorAndUniqueModId;

  static void ValidateGV(const GlobalVariable *GV);
  // Record a list of GlobalAlias associated with a GlobalObject.
  // This is used for AIX's extra-label-at-definition aliasing strategy.
  DenseMap<const GlobalObject *, SmallVector<const GlobalAlias *, 1>>
      GOAliasMap;

public:
  PPCAIXAsmPrinter(TargetMachine &TM, std::unique_ptr<MCStreamer> Streamer)
      : PPCAsmPrinter(TM, std::move(Streamer)) {
    if (MAI->isLittleEndian())
      report_fatal_error(
          "cannot create AIX PPC Assembly Printer for a little-endian target");
  }

  StringRef getPassName() const override { return "AIX PPC Assembly Printer"; }

  bool doInitialization(Module &M) override;

  void emitXXStructorList(const DataLayout &DL, const Constant *List,
                          bool IsCtor) override;

  void SetupMachineFunction(MachineFunction &MF) override;

  void emitGlobalVariable(const GlobalVariable *GV) override;

  void emitFunctionDescriptor() override;

  void emitFunctionEntryLabel() override;

  void emitEndOfAsmFile(Module &) override;

  void emitLinkage(const GlobalValue *GV, MCSymbol *GVSym) const override;

  void emitInstruction(const MachineInstr *MI) override;

  bool doFinalization(Module &M) override;
};

} // end anonymous namespace

void PPCAsmPrinter::PrintSymbolOperand(const MachineOperand &MO,
                                       raw_ostream &O) {
  // Computing the address of a global symbol, not calling it.
  const GlobalValue *GV = MO.getGlobal();
  getSymbol(GV)->print(O, MAI);
  printOffset(MO.getOffset(), O);
}

void PPCAsmPrinter::printOperand(const MachineInstr *MI, unsigned OpNo,
                                 raw_ostream &O) {
  const DataLayout &DL = getDataLayout();
  const MachineOperand &MO = MI->getOperand(OpNo);

  switch (MO.getType()) {
  case MachineOperand::MO_Register: {
    // The MI is INLINEASM ONLY and UseVSXReg is always false.
    const char *RegName = PPCInstPrinter::getRegisterName(MO.getReg());

    // Linux assembler (Others?) does not take register mnemonics.
    // FIXME - What about special registers used in mfspr/mtspr?
    O << PPCRegisterInfo::stripRegisterPrefix(RegName);
    return;
  }
  case MachineOperand::MO_Immediate:
    O << MO.getImm();
    return;

  case MachineOperand::MO_MachineBasicBlock:
    MO.getMBB()->getSymbol()->print(O, MAI);
    return;
  case MachineOperand::MO_ConstantPoolIndex:
    O << DL.getPrivateGlobalPrefix() << "CPI" << getFunctionNumber() << '_'
      << MO.getIndex();
    return;
  case MachineOperand::MO_BlockAddress:
    GetBlockAddressSymbol(MO.getBlockAddress())->print(O, MAI);
    return;
  case MachineOperand::MO_GlobalAddress: {
    PrintSymbolOperand(MO, O);
    return;
  }

  default:
    O << "<unknown operand type: " << (unsigned)MO.getType() << ">";
    return;
  }
}

/// PrintAsmOperand - Print out an operand for an inline asm expression.
///
bool PPCAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                    const char *ExtraCode, raw_ostream &O) {
  // Does this asm operand have a single letter operand modifier?
  if (ExtraCode && ExtraCode[0]) {
    if (ExtraCode[1] != 0) return true; // Unknown modifier.

    switch (ExtraCode[0]) {
    default:
      // See if this is a generic print operand
      return AsmPrinter::PrintAsmOperand(MI, OpNo, ExtraCode, O);
    case 'L': // Write second word of DImode reference.
      // Verify that this operand has two consecutive registers.
      if (!MI->getOperand(OpNo).isReg() ||
          OpNo+1 == MI->getNumOperands() ||
          !MI->getOperand(OpNo+1).isReg())
        return true;
      ++OpNo;   // Return the high-part.
      break;
    case 'I':
      // Write 'i' if an integer constant, otherwise nothing.  Used to print
      // addi vs add, etc.
      if (MI->getOperand(OpNo).isImm())
        O << "i";
      return false;
    case 'x':
      if(!MI->getOperand(OpNo).isReg())
        return true;
      // This operand uses VSX numbering.
      // If the operand is a VMX register, convert it to a VSX register.
      Register Reg = MI->getOperand(OpNo).getReg();
      if (PPCInstrInfo::isVRRegister(Reg))
        Reg = PPC::VSX32 + (Reg - PPC::V0);
      else if (PPCInstrInfo::isVFRegister(Reg))
        Reg = PPC::VSX32 + (Reg - PPC::VF0);
      const char *RegName;
      RegName = PPCInstPrinter::getRegisterName(Reg);
      RegName = PPCRegisterInfo::stripRegisterPrefix(RegName);
      O << RegName;
      return false;
    }
  }

  printOperand(MI, OpNo, O);
  return false;
}

// At the moment, all inline asm memory operands are a single register.
// In any case, the output of this routine should always be just one
// assembler operand.

bool PPCAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                                          const char *ExtraCode,
                                          raw_ostream &O) {
  if (ExtraCode && ExtraCode[0]) {
    if (ExtraCode[1] != 0) return true; // Unknown modifier.

    switch (ExtraCode[0]) {
    default: return true;  // Unknown modifier.
    case 'L': // A memory reference to the upper word of a double word op.
      O << getDataLayout().getPointerSize() << "(";
      printOperand(MI, OpNo, O);
      O << ")";
      return false;
    case 'y': // A memory reference for an X-form instruction
      O << "0, ";
      printOperand(MI, OpNo, O);
      return false;
    case 'U': // Print 'u' for update form.
    case 'X': // Print 'x' for indexed form.
      // FIXME: Currently for PowerPC memory operands are always loaded
      // into a register, so we never get an update or indexed form.
      // This is bad even for offset forms, since even if we know we
      // have a value in -16(r1), we will generate a load into r<n>
      // and then load from 0(r<n>).  Until that issue is fixed,
      // tolerate 'U' and 'X' but don't output anything.
      assert(MI->getOperand(OpNo).isReg());
      return false;
    }
  }

  assert(MI->getOperand(OpNo).isReg());
  O << "0(";
  printOperand(MI, OpNo, O);
  O << ")";
  return false;
}

/// lookUpOrCreateTOCEntry -- Given a symbol, look up whether a TOC entry
/// exists for it.  If not, create one.  Then return a symbol that references
/// the TOC entry.
MCSymbol *PPCAsmPrinter::lookUpOrCreateTOCEntry(const MCSymbol *Sym) {
  MCSymbol *&TOCEntry = TOC[Sym];
  if (!TOCEntry)
    TOCEntry = createTempSymbol("C");
  return TOCEntry;
}

void PPCAsmPrinter::emitEndOfAsmFile(Module &M) {
  emitStackMaps(SM);
}

void PPCAsmPrinter::LowerSTACKMAP(StackMaps &SM, const MachineInstr &MI) {
  unsigned NumNOPBytes = MI.getOperand(1).getImm();
  
  auto &Ctx = OutStreamer->getContext();
  MCSymbol *MILabel = Ctx.createTempSymbol();
  OutStreamer->emitLabel(MILabel);

  SM.recordStackMap(*MILabel, MI);
  assert(NumNOPBytes % 4 == 0 && "Invalid number of NOP bytes requested!");

  // Scan ahead to trim the shadow.
  const MachineBasicBlock &MBB = *MI.getParent();
  MachineBasicBlock::const_iterator MII(MI);
  ++MII;
  while (NumNOPBytes > 0) {
    if (MII == MBB.end() || MII->isCall() ||
        MII->getOpcode() == PPC::DBG_VALUE ||
        MII->getOpcode() == TargetOpcode::PATCHPOINT ||
        MII->getOpcode() == TargetOpcode::STACKMAP)
      break;
    ++MII;
    NumNOPBytes -= 4;
  }

  // Emit nops.
  for (unsigned i = 0; i < NumNOPBytes; i += 4)
    EmitToStreamer(*OutStreamer, MCInstBuilder(PPC::NOP));
}

// Lower a patchpoint of the form:
// [<def>], <id>, <numBytes>, <target>, <numArgs>
void PPCAsmPrinter::LowerPATCHPOINT(StackMaps &SM, const MachineInstr &MI) {
  auto &Ctx = OutStreamer->getContext();
  MCSymbol *MILabel = Ctx.createTempSymbol();
  OutStreamer->emitLabel(MILabel);

  SM.recordPatchPoint(*MILabel, MI);
  PatchPointOpers Opers(&MI);

  unsigned EncodedBytes = 0;
  const MachineOperand &CalleeMO = Opers.getCallTarget();

  if (CalleeMO.isImm()) {
    int64_t CallTarget = CalleeMO.getImm();
    if (CallTarget) {
      assert((CallTarget & 0xFFFFFFFFFFFF) == CallTarget &&
             "High 16 bits of call target should be zero.");
      Register ScratchReg = MI.getOperand(Opers.getNextScratchIdx()).getReg();
      EncodedBytes = 0;
      // Materialize the jump address:
      EmitToStreamer(*OutStreamer, MCInstBuilder(PPC::LI8)
                                      .addReg(ScratchReg)
                                      .addImm((CallTarget >> 32) & 0xFFFF));
      ++EncodedBytes;
      EmitToStreamer(*OutStreamer, MCInstBuilder(PPC::RLDIC)
                                      .addReg(ScratchReg)
                                      .addReg(ScratchReg)
                                      .addImm(32).addImm(16));
      ++EncodedBytes;
      EmitToStreamer(*OutStreamer, MCInstBuilder(PPC::ORIS8)
                                      .addReg(ScratchReg)
                                      .addReg(ScratchReg)
                                      .addImm((CallTarget >> 16) & 0xFFFF));
      ++EncodedBytes;
      EmitToStreamer(*OutStreamer, MCInstBuilder(PPC::ORI8)
                                      .addReg(ScratchReg)
                                      .addReg(ScratchReg)
                                      .addImm(CallTarget & 0xFFFF));

      // Save the current TOC pointer before the remote call.
      int TOCSaveOffset = Subtarget->getFrameLowering()->getTOCSaveOffset();
      EmitToStreamer(*OutStreamer, MCInstBuilder(PPC::STD)
                                      .addReg(PPC::X2)
                                      .addImm(TOCSaveOffset)
                                      .addReg(PPC::X1));
      ++EncodedBytes;

      // If we're on ELFv1, then we need to load the actual function pointer
      // from the function descriptor.
      if (!Subtarget->isELFv2ABI()) {
        // Load the new TOC pointer and the function address, but not r11
        // (needing this is rare, and loading it here would prevent passing it
        // via a 'nest' parameter.
        EmitToStreamer(*OutStreamer, MCInstBuilder(PPC::LD)
                                        .addReg(PPC::X2)
                                        .addImm(8)
                                        .addReg(ScratchReg));
        ++EncodedBytes;
        EmitToStreamer(*OutStreamer, MCInstBuilder(PPC::LD)
                                        .addReg(ScratchReg)
                                        .addImm(0)
                                        .addReg(ScratchReg));
        ++EncodedBytes;
      }

      EmitToStreamer(*OutStreamer, MCInstBuilder(PPC::MTCTR8)
                                      .addReg(ScratchReg));
      ++EncodedBytes;
      EmitToStreamer(*OutStreamer, MCInstBuilder(PPC::BCTRL8));
      ++EncodedBytes;

      // Restore the TOC pointer after the call.
      EmitToStreamer(*OutStreamer, MCInstBuilder(PPC::LD)
                                      .addReg(PPC::X2)
                                      .addImm(TOCSaveOffset)
                                      .addReg(PPC::X1));
      ++EncodedBytes;
    }
  } else if (CalleeMO.isGlobal()) {
    const GlobalValue *GValue = CalleeMO.getGlobal();
    MCSymbol *MOSymbol = getSymbol(GValue);
    const MCExpr *SymVar = MCSymbolRefExpr::create(MOSymbol, OutContext);

    EmitToStreamer(*OutStreamer, MCInstBuilder(PPC::BL8_NOP)
                                    .addExpr(SymVar));
    EncodedBytes += 2;
  }

  // Each instruction is 4 bytes.
  EncodedBytes *= 4;

  // Emit padding.
  unsigned NumBytes = Opers.getNumPatchBytes();
  assert(NumBytes >= EncodedBytes &&
         "Patchpoint can't request size less than the length of a call.");
  assert((NumBytes - EncodedBytes) % 4 == 0 &&
         "Invalid number of NOP bytes requested!");
  for (unsigned i = EncodedBytes; i < NumBytes; i += 4)
    EmitToStreamer(*OutStreamer, MCInstBuilder(PPC::NOP));
}

/// EmitTlsCall -- Given a GETtls[ld]ADDR[32] instruction, print a
/// call to __tls_get_addr to the current output stream.
void PPCAsmPrinter::EmitTlsCall(const MachineInstr *MI,
                                MCSymbolRefExpr::VariantKind VK) {
  StringRef Name = "__tls_get_addr";
  MCSymbol *TlsGetAddr = OutContext.getOrCreateSymbol(Name);
  MCSymbolRefExpr::VariantKind Kind = MCSymbolRefExpr::VK_None;
  unsigned Opcode = PPC::BL8_NOP_TLS;

  assert(MI->getNumOperands() >= 3 && "Expecting at least 3 operands from MI");
  if (MI->getOperand(2).getTargetFlags() == PPCII::MO_GOT_TLSGD_PCREL_FLAG) {
    Kind = MCSymbolRefExpr::VK_PPC_NOTOC;
    Opcode = PPC::BL8_NOTOC_TLS;
  }
  const Module *M = MF->getFunction().getParent();

  assert(MI->getOperand(0).isReg() &&
         ((Subtarget->isPPC64() && MI->getOperand(0).getReg() == PPC::X3) ||
          (!Subtarget->isPPC64() && MI->getOperand(0).getReg() == PPC::R3)) &&
         "GETtls[ld]ADDR[32] must define GPR3");
  assert(MI->getOperand(1).isReg() &&
         ((Subtarget->isPPC64() && MI->getOperand(1).getReg() == PPC::X3) ||
          (!Subtarget->isPPC64() && MI->getOperand(1).getReg() == PPC::R3)) &&
         "GETtls[ld]ADDR[32] must read GPR3");

  if (Subtarget->is32BitELFABI() && isPositionIndependent())
    Kind = MCSymbolRefExpr::VK_PLT;

  const MCExpr *TlsRef =
    MCSymbolRefExpr::create(TlsGetAddr, Kind, OutContext);

  // Add 32768 offset to the symbol so we follow up the latest GOT/PLT ABI.
  if (Kind == MCSymbolRefExpr::VK_PLT && Subtarget->isSecurePlt() &&
      M->getPICLevel() == PICLevel::BigPIC)
    TlsRef = MCBinaryExpr::createAdd(
        TlsRef, MCConstantExpr::create(32768, OutContext), OutContext);
  const MachineOperand &MO = MI->getOperand(2);
  const GlobalValue *GValue = MO.getGlobal();
  MCSymbol *MOSymbol = getSymbol(GValue);
  const MCExpr *SymVar = MCSymbolRefExpr::create(MOSymbol, VK, OutContext);
  EmitToStreamer(*OutStreamer,
                 MCInstBuilder(Subtarget->isPPC64() ? Opcode
                                                    : (unsigned)PPC::BL_TLS)
                     .addExpr(TlsRef)
                     .addExpr(SymVar));
}

/// Map a machine operand for a TOC pseudo-machine instruction to its
/// corresponding MCSymbol.
static MCSymbol *getMCSymbolForTOCPseudoMO(const MachineOperand &MO,
                                           AsmPrinter &AP) {
  switch (MO.getType()) {
  case MachineOperand::MO_GlobalAddress:
    return AP.getSymbol(MO.getGlobal());
  case MachineOperand::MO_ConstantPoolIndex:
    return AP.GetCPISymbol(MO.getIndex());
  case MachineOperand::MO_JumpTableIndex:
    return AP.GetJTISymbol(MO.getIndex());
  case MachineOperand::MO_BlockAddress:
    return AP.GetBlockAddressSymbol(MO.getBlockAddress());
  default:
    llvm_unreachable("Unexpected operand type to get symbol.");
  }
}

/// EmitInstruction -- Print out a single PowerPC MI in Darwin syntax to
/// the current output stream.
///
void PPCAsmPrinter::emitInstruction(const MachineInstr *MI) {
  MCInst TmpInst;
  const bool IsPPC64 = Subtarget->isPPC64();
  const bool IsAIX = Subtarget->isAIXABI();
  const Module *M = MF->getFunction().getParent();
  PICLevel::Level PL = M->getPICLevel();

#ifndef NDEBUG
  // Validate that SPE and FPU are mutually exclusive in codegen
  if (!MI->isInlineAsm()) {
    for (const MachineOperand &MO: MI->operands()) {
      if (MO.isReg()) {
        Register Reg = MO.getReg();
        if (Subtarget->hasSPE()) {
          if (PPC::F4RCRegClass.contains(Reg) ||
              PPC::F8RCRegClass.contains(Reg) ||
              PPC::VFRCRegClass.contains(Reg) ||
              PPC::VRRCRegClass.contains(Reg) ||
              PPC::VSFRCRegClass.contains(Reg) ||
              PPC::VSSRCRegClass.contains(Reg)
              )
            llvm_unreachable("SPE targets cannot have FPRegs!");
        } else {
          if (PPC::SPERCRegClass.contains(Reg))
            llvm_unreachable("SPE register found in FPU-targeted code!");
        }
      }
    }
  }
#endif

  auto getTOCRelocAdjustedExprForXCOFF = [this](const MCExpr *Expr,
                                                ptrdiff_t OriginalOffset) {
    // Apply an offset to the TOC-based expression such that the adjusted
    // notional offset from the TOC base (to be encoded into the instruction's D
    // or DS field) is the signed 16-bit truncation of the original notional
    // offset from the TOC base.
    // This is consistent with the treatment used both by XL C/C++ and
    // by AIX ld -r.
    ptrdiff_t Adjustment =
        OriginalOffset - llvm::SignExtend32<16>(OriginalOffset);
    return MCBinaryExpr::createAdd(
        Expr, MCConstantExpr::create(-Adjustment, OutContext), OutContext);
  };

  auto getTOCEntryLoadingExprForXCOFF =
      [IsPPC64, getTOCRelocAdjustedExprForXCOFF,
       this](const MCSymbol *MOSymbol, const MCExpr *Expr) -> const MCExpr * {
    const unsigned EntryByteSize = IsPPC64 ? 8 : 4;
    const auto TOCEntryIter = TOC.find(MOSymbol);
    assert(TOCEntryIter != TOC.end() &&
           "Could not find the TOC entry for this symbol.");
    const ptrdiff_t EntryDistanceFromTOCBase =
        (TOCEntryIter - TOC.begin()) * EntryByteSize;
    constexpr int16_t PositiveTOCRange = INT16_MAX;

    if (EntryDistanceFromTOCBase > PositiveTOCRange)
      return getTOCRelocAdjustedExprForXCOFF(Expr, EntryDistanceFromTOCBase);

    return Expr;
  };

  // Lower multi-instruction pseudo operations.
  switch (MI->getOpcode()) {
  default: break;
  case TargetOpcode::DBG_VALUE:
    llvm_unreachable("Should be handled target independently");
  case TargetOpcode::STACKMAP:
    return LowerSTACKMAP(SM, *MI);
  case TargetOpcode::PATCHPOINT:
    return LowerPATCHPOINT(SM, *MI);

  case PPC::MoveGOTtoLR: {
    // Transform %lr = MoveGOTtoLR
    // Into this: bl _GLOBAL_OFFSET_TABLE_@local-4
    // _GLOBAL_OFFSET_TABLE_@local-4 (instruction preceding
    // _GLOBAL_OFFSET_TABLE_) has exactly one instruction:
    //      blrl
    // This will return the pointer to _GLOBAL_OFFSET_TABLE_@local
    MCSymbol *GOTSymbol =
      OutContext.getOrCreateSymbol(StringRef("_GLOBAL_OFFSET_TABLE_"));
    const MCExpr *OffsExpr =
      MCBinaryExpr::createSub(MCSymbolRefExpr::create(GOTSymbol,
                                                      MCSymbolRefExpr::VK_PPC_LOCAL,
                                                      OutContext),
                              MCConstantExpr::create(4, OutContext),
                              OutContext);

    // Emit the 'bl'.
    EmitToStreamer(*OutStreamer, MCInstBuilder(PPC::BL).addExpr(OffsExpr));
    return;
  }
  case PPC::MovePCtoLR:
  case PPC::MovePCtoLR8: {
    // Transform %lr = MovePCtoLR
    // Into this, where the label is the PIC base:
    //     bl L1$pb
    // L1$pb:
    MCSymbol *PICBase = MF->getPICBaseSymbol();

    // Emit the 'bl'.
    EmitToStreamer(*OutStreamer,
                   MCInstBuilder(PPC::BL)
                       // FIXME: We would like an efficient form for this, so we
                       // don't have to do a lot of extra uniquing.
                       .addExpr(MCSymbolRefExpr::create(PICBase, OutContext)));

    // Emit the label.
    OutStreamer->emitLabel(PICBase);
    return;
  }
  case PPC::UpdateGBR: {
    // Transform %rd = UpdateGBR(%rt, %ri)
    // Into: lwz %rt, .L0$poff - .L0$pb(%ri)
    //       add %rd, %rt, %ri
    // or into (if secure plt mode is on):
    //       addis r30, r30, {.LTOC,_GLOBAL_OFFSET_TABLE} - .L0$pb@ha
    //       addi r30, r30, {.LTOC,_GLOBAL_OFFSET_TABLE} - .L0$pb@l
    // Get the offset from the GOT Base Register to the GOT
    LowerPPCMachineInstrToMCInst(MI, TmpInst, *this);
    if (Subtarget->isSecurePlt() && isPositionIndependent() ) {
      unsigned PICR = TmpInst.getOperand(0).getReg();
      MCSymbol *BaseSymbol = OutContext.getOrCreateSymbol(
          M->getPICLevel() == PICLevel::SmallPIC ? "_GLOBAL_OFFSET_TABLE_"
                                                 : ".LTOC");
      const MCExpr *PB =
          MCSymbolRefExpr::create(MF->getPICBaseSymbol(), OutContext);

      const MCExpr *DeltaExpr = MCBinaryExpr::createSub(
          MCSymbolRefExpr::create(BaseSymbol, OutContext), PB, OutContext);

      const MCExpr *DeltaHi = PPCMCExpr::createHa(DeltaExpr, OutContext);
      EmitToStreamer(
          *OutStreamer,
          MCInstBuilder(PPC::ADDIS).addReg(PICR).addReg(PICR).addExpr(DeltaHi));

      const MCExpr *DeltaLo = PPCMCExpr::createLo(DeltaExpr, OutContext);
      EmitToStreamer(
          *OutStreamer,
          MCInstBuilder(PPC::ADDI).addReg(PICR).addReg(PICR).addExpr(DeltaLo));
      return;
    } else {
      MCSymbol *PICOffset =
        MF->getInfo<PPCFunctionInfo>()->getPICOffsetSymbol(*MF);
      TmpInst.setOpcode(PPC::LWZ);
      const MCExpr *Exp =
        MCSymbolRefExpr::create(PICOffset, MCSymbolRefExpr::VK_None, OutContext);
      const MCExpr *PB =
        MCSymbolRefExpr::create(MF->getPICBaseSymbol(),
                                MCSymbolRefExpr::VK_None,
                                OutContext);
      const MCOperand TR = TmpInst.getOperand(1);
      const MCOperand PICR = TmpInst.getOperand(0);

      // Step 1: lwz %rt, .L$poff - .L$pb(%ri)
      TmpInst.getOperand(1) =
          MCOperand::createExpr(MCBinaryExpr::createSub(Exp, PB, OutContext));
      TmpInst.getOperand(0) = TR;
      TmpInst.getOperand(2) = PICR;
      EmitToStreamer(*OutStreamer, TmpInst);

      TmpInst.setOpcode(PPC::ADD4);
      TmpInst.getOperand(0) = PICR;
      TmpInst.getOperand(1) = TR;
      TmpInst.getOperand(2) = PICR;
      EmitToStreamer(*OutStreamer, TmpInst);
      return;
    }
  }
  case PPC::LWZtoc: {
    // Transform %rN = LWZtoc @op1, %r2
    LowerPPCMachineInstrToMCInst(MI, TmpInst, *this);

    // Change the opcode to LWZ.
    TmpInst.setOpcode(PPC::LWZ);

    const MachineOperand &MO = MI->getOperand(1);
    assert((MO.isGlobal() || MO.isCPI() || MO.isJTI() || MO.isBlockAddress()) &&
           "Invalid operand for LWZtoc.");

    // Map the operand to its corresponding MCSymbol.
    const MCSymbol *const MOSymbol = getMCSymbolForTOCPseudoMO(MO, *this);

    // Create a reference to the GOT entry for the symbol. The GOT entry will be
    // synthesized later.
    if (PL == PICLevel::SmallPIC && !IsAIX) {
      const MCExpr *Exp =
        MCSymbolRefExpr::create(MOSymbol, MCSymbolRefExpr::VK_GOT,
                                OutContext);
      TmpInst.getOperand(1) = MCOperand::createExpr(Exp);
      EmitToStreamer(*OutStreamer, TmpInst);
      return;
    }

    // Otherwise, use the TOC. 'TOCEntry' is a label used to reference the
    // storage allocated in the TOC which contains the address of
    // 'MOSymbol'. Said TOC entry will be synthesized later.
    MCSymbol *TOCEntry = lookUpOrCreateTOCEntry(MOSymbol);
    const MCExpr *Exp =
        MCSymbolRefExpr::create(TOCEntry, MCSymbolRefExpr::VK_None, OutContext);

    // AIX uses the label directly as the lwz displacement operand for
    // references into the toc section. The displacement value will be generated
    // relative to the toc-base.
    if (IsAIX) {
      assert(
          TM.getCodeModel() == CodeModel::Small &&
          "This pseudo should only be selected for 32-bit small code model.");
      Exp = getTOCEntryLoadingExprForXCOFF(MOSymbol, Exp);
      TmpInst.getOperand(1) = MCOperand::createExpr(Exp);
      EmitToStreamer(*OutStreamer, TmpInst);
      return;
    }

    // Create an explicit subtract expression between the local symbol and
    // '.LTOC' to manifest the toc-relative offset.
    const MCExpr *PB = MCSymbolRefExpr::create(
        OutContext.getOrCreateSymbol(Twine(".LTOC")), OutContext);
    Exp = MCBinaryExpr::createSub(Exp, PB, OutContext);
    TmpInst.getOperand(1) = MCOperand::createExpr(Exp);
    EmitToStreamer(*OutStreamer, TmpInst);
    return;
  }
  case PPC::LDtocJTI:
  case PPC::LDtocCPT:
  case PPC::LDtocBA:
  case PPC::LDtoc: {
    // Transform %x3 = LDtoc @min1, %x2
    LowerPPCMachineInstrToMCInst(MI, TmpInst, *this);

    // Change the opcode to LD.
    TmpInst.setOpcode(PPC::LD);

    const MachineOperand &MO = MI->getOperand(1);
    assert((MO.isGlobal() || MO.isCPI() || MO.isJTI() || MO.isBlockAddress()) &&
           "Invalid operand!");

    // Map the operand to its corresponding MCSymbol.
    const MCSymbol *const MOSymbol = getMCSymbolForTOCPseudoMO(MO, *this);

    // Map the machine operand to its corresponding MCSymbol, then map the
    // global address operand to be a reference to the TOC entry we will
    // synthesize later.
    MCSymbol *TOCEntry = lookUpOrCreateTOCEntry(MOSymbol);

    const MCSymbolRefExpr::VariantKind VK =
        IsAIX ? MCSymbolRefExpr::VK_None : MCSymbolRefExpr::VK_PPC_TOC;
    const MCExpr *Exp =
        MCSymbolRefExpr::create(TOCEntry, VK, OutContext);
    TmpInst.getOperand(1) = MCOperand::createExpr(
        IsAIX ? getTOCEntryLoadingExprForXCOFF(MOSymbol, Exp) : Exp);
    EmitToStreamer(*OutStreamer, TmpInst);
    return;
  }
  case PPC::ADDIStocHA: {
    assert((IsAIX && !IsPPC64 && TM.getCodeModel() == CodeModel::Large) &&
           "This pseudo should only be selected for 32-bit large code model on"
           " AIX.");

    // Transform %rd = ADDIStocHA %rA, @sym(%r2)
    LowerPPCMachineInstrToMCInst(MI, TmpInst, *this);

    // Change the opcode to ADDIS.
    TmpInst.setOpcode(PPC::ADDIS);

    const MachineOperand &MO = MI->getOperand(2);
    assert((MO.isGlobal() || MO.isCPI() || MO.isJTI() || MO.isBlockAddress()) &&
           "Invalid operand for ADDIStocHA.");

    // Map the machine operand to its corresponding MCSymbol.
    MCSymbol *MOSymbol = getMCSymbolForTOCPseudoMO(MO, *this);

    // Always use TOC on AIX. Map the global address operand to be a reference
    // to the TOC entry we will synthesize later. 'TOCEntry' is a label used to
    // reference the storage allocated in the TOC which contains the address of
    // 'MOSymbol'.
    MCSymbol *TOCEntry = lookUpOrCreateTOCEntry(MOSymbol);
    const MCExpr *Exp = MCSymbolRefExpr::create(TOCEntry,
                                                MCSymbolRefExpr::VK_PPC_U,
                                                OutContext);
    TmpInst.getOperand(2) = MCOperand::createExpr(Exp);
    EmitToStreamer(*OutStreamer, TmpInst);
    return;
  }
  case PPC::LWZtocL: {
    assert(IsAIX && !IsPPC64 && TM.getCodeModel() == CodeModel::Large &&
           "This pseudo should only be selected for 32-bit large code model on"
           " AIX.");

    // Transform %rd = LWZtocL @sym, %rs.
    LowerPPCMachineInstrToMCInst(MI, TmpInst, *this);

    // Change the opcode to lwz.
    TmpInst.setOpcode(PPC::LWZ);

    const MachineOperand &MO = MI->getOperand(1);
    assert((MO.isGlobal() || MO.isCPI() || MO.isJTI() || MO.isBlockAddress()) &&
           "Invalid operand for LWZtocL.");

    // Map the machine operand to its corresponding MCSymbol.
    MCSymbol *MOSymbol = getMCSymbolForTOCPseudoMO(MO, *this);

    // Always use TOC on AIX. Map the global address operand to be a reference
    // to the TOC entry we will synthesize later. 'TOCEntry' is a label used to
    // reference the storage allocated in the TOC which contains the address of
    // 'MOSymbol'.
    MCSymbol *TOCEntry = lookUpOrCreateTOCEntry(MOSymbol);
    const MCExpr *Exp = MCSymbolRefExpr::create(TOCEntry,
                                                MCSymbolRefExpr::VK_PPC_L,
                                                OutContext);
    TmpInst.getOperand(1) = MCOperand::createExpr(Exp);
    EmitToStreamer(*OutStreamer, TmpInst);
    return;
  }
  case PPC::ADDIStocHA8: {
    // Transform %xd = ADDIStocHA8 %x2, @sym
    LowerPPCMachineInstrToMCInst(MI, TmpInst, *this);

    // Change the opcode to ADDIS8. If the global address is the address of
    // an external symbol, is a jump table address, is a block address, or is a
    // constant pool index with large code model enabled, then generate a TOC
    // entry and reference that. Otherwise, reference the symbol directly.
    TmpInst.setOpcode(PPC::ADDIS8);

    const MachineOperand &MO = MI->getOperand(2);
    assert((MO.isGlobal() || MO.isCPI() || MO.isJTI() || MO.isBlockAddress()) &&
           "Invalid operand for ADDIStocHA8!");

    const MCSymbol *MOSymbol = getMCSymbolForTOCPseudoMO(MO, *this);

    const bool GlobalToc =
        MO.isGlobal() && Subtarget->isGVIndirectSymbol(MO.getGlobal());
    if (GlobalToc || MO.isJTI() || MO.isBlockAddress() ||
        (MO.isCPI() && TM.getCodeModel() == CodeModel::Large))
      MOSymbol = lookUpOrCreateTOCEntry(MOSymbol);

    const MCSymbolRefExpr::VariantKind VK =
        IsAIX ? MCSymbolRefExpr::VK_PPC_U : MCSymbolRefExpr::VK_PPC_TOC_HA;

    const MCExpr *Exp =
        MCSymbolRefExpr::create(MOSymbol, VK, OutContext);

    if (!MO.isJTI() && MO.getOffset())
      Exp = MCBinaryExpr::createAdd(Exp,
                                    MCConstantExpr::create(MO.getOffset(),
                                                           OutContext),
                                    OutContext);

    TmpInst.getOperand(2) = MCOperand::createExpr(Exp);
    EmitToStreamer(*OutStreamer, TmpInst);
    return;
  }
  case PPC::LDtocL: {
    // Transform %xd = LDtocL @sym, %xs
    LowerPPCMachineInstrToMCInst(MI, TmpInst, *this);

    // Change the opcode to LD. If the global address is the address of
    // an external symbol, is a jump table address, is a block address, or is
    // a constant pool index with large code model enabled, then generate a
    // TOC entry and reference that. Otherwise, reference the symbol directly.
    TmpInst.setOpcode(PPC::LD);

    const MachineOperand &MO = MI->getOperand(1);
    assert((MO.isGlobal() || MO.isCPI() || MO.isJTI() ||
            MO.isBlockAddress()) &&
           "Invalid operand for LDtocL!");

    LLVM_DEBUG(assert(
        (!MO.isGlobal() || Subtarget->isGVIndirectSymbol(MO.getGlobal())) &&
        "LDtocL used on symbol that could be accessed directly is "
        "invalid. Must match ADDIStocHA8."));

    const MCSymbol *MOSymbol = getMCSymbolForTOCPseudoMO(MO, *this);

    if (!MO.isCPI() || TM.getCodeModel() == CodeModel::Large)
      MOSymbol = lookUpOrCreateTOCEntry(MOSymbol);

    const MCSymbolRefExpr::VariantKind VK =
        IsAIX ? MCSymbolRefExpr::VK_PPC_L : MCSymbolRefExpr::VK_PPC_TOC_LO;
    const MCExpr *Exp =
        MCSymbolRefExpr::create(MOSymbol, VK, OutContext);
    TmpInst.getOperand(1) = MCOperand::createExpr(Exp);
    EmitToStreamer(*OutStreamer, TmpInst);
    return;
  }
  case PPC::ADDItocL: {
    // Transform %xd = ADDItocL %xs, @sym
    LowerPPCMachineInstrToMCInst(MI, TmpInst, *this);

    // Change the opcode to ADDI8. If the global address is external, then
    // generate a TOC entry and reference that. Otherwise, reference the
    // symbol directly.
    TmpInst.setOpcode(PPC::ADDI8);

    const MachineOperand &MO = MI->getOperand(2);
    assert((MO.isGlobal() || MO.isCPI()) && "Invalid operand for ADDItocL.");

    LLVM_DEBUG(assert(
        !(MO.isGlobal() && Subtarget->isGVIndirectSymbol(MO.getGlobal())) &&
        "Interposable definitions must use indirect access."));

    const MCExpr *Exp =
        MCSymbolRefExpr::create(getMCSymbolForTOCPseudoMO(MO, *this),
                                MCSymbolRefExpr::VK_PPC_TOC_LO, OutContext);
    TmpInst.getOperand(2) = MCOperand::createExpr(Exp);
    EmitToStreamer(*OutStreamer, TmpInst);
    return;
  }
  case PPC::ADDISgotTprelHA: {
    // Transform: %xd = ADDISgotTprelHA %x2, @sym
    // Into:      %xd = ADDIS8 %x2, sym@got@tlsgd@ha
    assert(IsPPC64 && "Not supported for 32-bit PowerPC");
    const MachineOperand &MO = MI->getOperand(2);
    const GlobalValue *GValue = MO.getGlobal();
    MCSymbol *MOSymbol = getSymbol(GValue);
    const MCExpr *SymGotTprel =
        MCSymbolRefExpr::create(MOSymbol, MCSymbolRefExpr::VK_PPC_GOT_TPREL_HA,
                                OutContext);
    EmitToStreamer(*OutStreamer, MCInstBuilder(PPC::ADDIS8)
                                 .addReg(MI->getOperand(0).getReg())
                                 .addReg(MI->getOperand(1).getReg())
                                 .addExpr(SymGotTprel));
    return;
  }
  case PPC::LDgotTprelL:
  case PPC::LDgotTprelL32: {
    // Transform %xd = LDgotTprelL @sym, %xs
    LowerPPCMachineInstrToMCInst(MI, TmpInst, *this);

    // Change the opcode to LD.
    TmpInst.setOpcode(IsPPC64 ? PPC::LD : PPC::LWZ);
    const MachineOperand &MO = MI->getOperand(1);
    const GlobalValue *GValue = MO.getGlobal();
    MCSymbol *MOSymbol = getSymbol(GValue);
    const MCExpr *Exp = MCSymbolRefExpr::create(
        MOSymbol, IsPPC64 ? MCSymbolRefExpr::VK_PPC_GOT_TPREL_LO
                          : MCSymbolRefExpr::VK_PPC_GOT_TPREL,
        OutContext);
    TmpInst.getOperand(1) = MCOperand::createExpr(Exp);
    EmitToStreamer(*OutStreamer, TmpInst);
    return;
  }

  case PPC::PPC32PICGOT: {
    MCSymbol *GOTSymbol = OutContext.getOrCreateSymbol(StringRef("_GLOBAL_OFFSET_TABLE_"));
    MCSymbol *GOTRef = OutContext.createTempSymbol();
    MCSymbol *NextInstr = OutContext.createTempSymbol();

    EmitToStreamer(*OutStreamer, MCInstBuilder(PPC::BL)
      // FIXME: We would like an efficient form for this, so we don't have to do
      // a lot of extra uniquing.
      .addExpr(MCSymbolRefExpr::create(NextInstr, OutContext)));
    const MCExpr *OffsExpr =
      MCBinaryExpr::createSub(MCSymbolRefExpr::create(GOTSymbol, OutContext),
                                MCSymbolRefExpr::create(GOTRef, OutContext),
        OutContext);
    OutStreamer->emitLabel(GOTRef);
    OutStreamer->emitValue(OffsExpr, 4);
    OutStreamer->emitLabel(NextInstr);
    EmitToStreamer(*OutStreamer, MCInstBuilder(PPC::MFLR)
                                 .addReg(MI->getOperand(0).getReg()));
    EmitToStreamer(*OutStreamer, MCInstBuilder(PPC::LWZ)
                                 .addReg(MI->getOperand(1).getReg())
                                 .addImm(0)
                                 .addReg(MI->getOperand(0).getReg()));
    EmitToStreamer(*OutStreamer, MCInstBuilder(PPC::ADD4)
                                 .addReg(MI->getOperand(0).getReg())
                                 .addReg(MI->getOperand(1).getReg())
                                 .addReg(MI->getOperand(0).getReg()));
    return;
  }
  case PPC::PPC32GOT: {
    MCSymbol *GOTSymbol =
        OutContext.getOrCreateSymbol(StringRef("_GLOBAL_OFFSET_TABLE_"));
    const MCExpr *SymGotTlsL = MCSymbolRefExpr::create(
        GOTSymbol, MCSymbolRefExpr::VK_PPC_LO, OutContext);
    const MCExpr *SymGotTlsHA = MCSymbolRefExpr::create(
        GOTSymbol, MCSymbolRefExpr::VK_PPC_HA, OutContext);
    EmitToStreamer(*OutStreamer, MCInstBuilder(PPC::LI)
                                 .addReg(MI->getOperand(0).getReg())
                                 .addExpr(SymGotTlsL));
    EmitToStreamer(*OutStreamer, MCInstBuilder(PPC::ADDIS)
                                 .addReg(MI->getOperand(0).getReg())
                                 .addReg(MI->getOperand(0).getReg())
                                 .addExpr(SymGotTlsHA));
    return;
  }
  case PPC::ADDIStlsgdHA: {
    // Transform: %xd = ADDIStlsgdHA %x2, @sym
    // Into:      %xd = ADDIS8 %x2, sym@got@tlsgd@ha
    assert(IsPPC64 && "Not supported for 32-bit PowerPC");
    const MachineOperand &MO = MI->getOperand(2);
    const GlobalValue *GValue = MO.getGlobal();
    MCSymbol *MOSymbol = getSymbol(GValue);
    const MCExpr *SymGotTlsGD =
      MCSymbolRefExpr::create(MOSymbol, MCSymbolRefExpr::VK_PPC_GOT_TLSGD_HA,
                              OutContext);
    EmitToStreamer(*OutStreamer, MCInstBuilder(PPC::ADDIS8)
                                 .addReg(MI->getOperand(0).getReg())
                                 .addReg(MI->getOperand(1).getReg())
                                 .addExpr(SymGotTlsGD));
    return;
  }
  case PPC::ADDItlsgdL:
    // Transform: %xd = ADDItlsgdL %xs, @sym
    // Into:      %xd = ADDI8 %xs, sym@got@tlsgd@l
  case PPC::ADDItlsgdL32: {
    // Transform: %rd = ADDItlsgdL32 %rs, @sym
    // Into:      %rd = ADDI %rs, sym@got@tlsgd
    const MachineOperand &MO = MI->getOperand(2);
    const GlobalValue *GValue = MO.getGlobal();
    MCSymbol *MOSymbol = getSymbol(GValue);
    const MCExpr *SymGotTlsGD = MCSymbolRefExpr::create(
        MOSymbol, IsPPC64 ? MCSymbolRefExpr::VK_PPC_GOT_TLSGD_LO
                          : MCSymbolRefExpr::VK_PPC_GOT_TLSGD,
        OutContext);
    EmitToStreamer(*OutStreamer,
                   MCInstBuilder(IsPPC64 ? PPC::ADDI8 : PPC::ADDI)
                   .addReg(MI->getOperand(0).getReg())
                   .addReg(MI->getOperand(1).getReg())
                   .addExpr(SymGotTlsGD));
    return;
  }
  case PPC::GETtlsADDR:
    // Transform: %x3 = GETtlsADDR %x3, @sym
    // Into: BL8_NOP_TLS __tls_get_addr(sym at tlsgd)
  case PPC::GETtlsADDR32: {
    // Transform: %r3 = GETtlsADDR32 %r3, @sym
    // Into: BL_TLS __tls_get_addr(sym at tlsgd)@PLT
    EmitTlsCall(MI, MCSymbolRefExpr::VK_PPC_TLSGD);
    return;
  }
  case PPC::ADDIStlsldHA: {
    // Transform: %xd = ADDIStlsldHA %x2, @sym
    // Into:      %xd = ADDIS8 %x2, sym@got@tlsld@ha
    assert(IsPPC64 && "Not supported for 32-bit PowerPC");
    const MachineOperand &MO = MI->getOperand(2);
    const GlobalValue *GValue = MO.getGlobal();
    MCSymbol *MOSymbol = getSymbol(GValue);
    const MCExpr *SymGotTlsLD =
      MCSymbolRefExpr::create(MOSymbol, MCSymbolRefExpr::VK_PPC_GOT_TLSLD_HA,
                              OutContext);
    EmitToStreamer(*OutStreamer, MCInstBuilder(PPC::ADDIS8)
                                 .addReg(MI->getOperand(0).getReg())
                                 .addReg(MI->getOperand(1).getReg())
                                 .addExpr(SymGotTlsLD));
    return;
  }
  case PPC::ADDItlsldL:
    // Transform: %xd = ADDItlsldL %xs, @sym
    // Into:      %xd = ADDI8 %xs, sym@got@tlsld@l
  case PPC::ADDItlsldL32: {
    // Transform: %rd = ADDItlsldL32 %rs, @sym
    // Into:      %rd = ADDI %rs, sym@got@tlsld
    const MachineOperand &MO = MI->getOperand(2);
    const GlobalValue *GValue = MO.getGlobal();
    MCSymbol *MOSymbol = getSymbol(GValue);
    const MCExpr *SymGotTlsLD = MCSymbolRefExpr::create(
        MOSymbol, IsPPC64 ? MCSymbolRefExpr::VK_PPC_GOT_TLSLD_LO
                          : MCSymbolRefExpr::VK_PPC_GOT_TLSLD,
        OutContext);
    EmitToStreamer(*OutStreamer,
                   MCInstBuilder(IsPPC64 ? PPC::ADDI8 : PPC::ADDI)
                       .addReg(MI->getOperand(0).getReg())
                       .addReg(MI->getOperand(1).getReg())
                       .addExpr(SymGotTlsLD));
    return;
  }
  case PPC::GETtlsldADDR:
    // Transform: %x3 = GETtlsldADDR %x3, @sym
    // Into: BL8_NOP_TLS __tls_get_addr(sym at tlsld)
  case PPC::GETtlsldADDR32: {
    // Transform: %r3 = GETtlsldADDR32 %r3, @sym
    // Into: BL_TLS __tls_get_addr(sym at tlsld)@PLT
    EmitTlsCall(MI, MCSymbolRefExpr::VK_PPC_TLSLD);
    return;
  }
  case PPC::ADDISdtprelHA:
    // Transform: %xd = ADDISdtprelHA %xs, @sym
    // Into:      %xd = ADDIS8 %xs, sym@dtprel@ha
  case PPC::ADDISdtprelHA32: {
    // Transform: %rd = ADDISdtprelHA32 %rs, @sym
    // Into:      %rd = ADDIS %rs, sym@dtprel@ha
    const MachineOperand &MO = MI->getOperand(2);
    const GlobalValue *GValue = MO.getGlobal();
    MCSymbol *MOSymbol = getSymbol(GValue);
    const MCExpr *SymDtprel =
      MCSymbolRefExpr::create(MOSymbol, MCSymbolRefExpr::VK_PPC_DTPREL_HA,
                              OutContext);
    EmitToStreamer(
        *OutStreamer,
        MCInstBuilder(IsPPC64 ? PPC::ADDIS8 : PPC::ADDIS)
            .addReg(MI->getOperand(0).getReg())
            .addReg(MI->getOperand(1).getReg())
            .addExpr(SymDtprel));
    return;
  }
  case PPC::ADDIdtprelL:
    // Transform: %xd = ADDIdtprelL %xs, @sym
    // Into:      %xd = ADDI8 %xs, sym@dtprel@l
  case PPC::ADDIdtprelL32: {
    // Transform: %rd = ADDIdtprelL32 %rs, @sym
    // Into:      %rd = ADDI %rs, sym@dtprel@l
    const MachineOperand &MO = MI->getOperand(2);
    const GlobalValue *GValue = MO.getGlobal();
    MCSymbol *MOSymbol = getSymbol(GValue);
    const MCExpr *SymDtprel =
      MCSymbolRefExpr::create(MOSymbol, MCSymbolRefExpr::VK_PPC_DTPREL_LO,
                              OutContext);
    EmitToStreamer(*OutStreamer,
                   MCInstBuilder(IsPPC64 ? PPC::ADDI8 : PPC::ADDI)
                       .addReg(MI->getOperand(0).getReg())
                       .addReg(MI->getOperand(1).getReg())
                       .addExpr(SymDtprel));
    return;
  }
  case PPC::MFOCRF:
  case PPC::MFOCRF8:
    if (!Subtarget->hasMFOCRF()) {
      // Transform: %r3 = MFOCRF %cr7
      // Into:      %r3 = MFCR   ;; cr7
      unsigned NewOpcode =
        MI->getOpcode() == PPC::MFOCRF ? PPC::MFCR : PPC::MFCR8;
      OutStreamer->AddComment(PPCInstPrinter::
                              getRegisterName(MI->getOperand(1).getReg()));
      EmitToStreamer(*OutStreamer, MCInstBuilder(NewOpcode)
                                  .addReg(MI->getOperand(0).getReg()));
      return;
    }
    break;
  case PPC::MTOCRF:
  case PPC::MTOCRF8:
    if (!Subtarget->hasMFOCRF()) {
      // Transform: %cr7 = MTOCRF %r3
      // Into:      MTCRF mask, %r3 ;; cr7
      unsigned NewOpcode =
        MI->getOpcode() == PPC::MTOCRF ? PPC::MTCRF : PPC::MTCRF8;
      unsigned Mask = 0x80 >> OutContext.getRegisterInfo()
                              ->getEncodingValue(MI->getOperand(0).getReg());
      OutStreamer->AddComment(PPCInstPrinter::
                              getRegisterName(MI->getOperand(0).getReg()));
      EmitToStreamer(*OutStreamer, MCInstBuilder(NewOpcode)
                                     .addImm(Mask)
                                     .addReg(MI->getOperand(1).getReg()));
      return;
    }
    break;
  case PPC::LD:
  case PPC::STD:
  case PPC::LWA_32:
  case PPC::LWA: {
    // Verify alignment is legal, so we don't create relocations
    // that can't be supported.
    // FIXME:  This test is currently disabled for Darwin.  The test
    // suite shows a handful of test cases that fail this check for
    // Darwin.  Those need to be investigated before this sanity test
    // can be enabled for those subtargets.
    unsigned OpNum = (MI->getOpcode() == PPC::STD) ? 2 : 1;
    const MachineOperand &MO = MI->getOperand(OpNum);
    if (MO.isGlobal()) {
      const DataLayout &DL = MO.getGlobal()->getParent()->getDataLayout();
      if (MO.getGlobal()->getPointerAlignment(DL) < 4)
        llvm_unreachable("Global must be word-aligned for LD, STD, LWA!");
    }
    // Now process the instruction normally.
    break;
  }
  }

  LowerPPCMachineInstrToMCInst(MI, TmpInst, *this);
  EmitToStreamer(*OutStreamer, TmpInst);
}

void PPCLinuxAsmPrinter::emitInstruction(const MachineInstr *MI) {
  if (!Subtarget->isPPC64())
    return PPCAsmPrinter::emitInstruction(MI);

  switch (MI->getOpcode()) {
  default:
    return PPCAsmPrinter::emitInstruction(MI);
  case TargetOpcode::PATCHABLE_FUNCTION_ENTER: {
    // .begin:
    //   b .end # lis 0, FuncId[16..32]
    //   nop    # li  0, FuncId[0..15]
    //   std 0, -8(1)
    //   mflr 0
    //   bl __xray_FunctionEntry
    //   mtlr 0
    // .end:
    //
    // Update compiler-rt/lib/xray/xray_powerpc64.cc accordingly when number
    // of instructions change.
    MCSymbol *BeginOfSled = OutContext.createTempSymbol();
    MCSymbol *EndOfSled = OutContext.createTempSymbol();
    OutStreamer->emitLabel(BeginOfSled);
    EmitToStreamer(*OutStreamer,
                   MCInstBuilder(PPC::B).addExpr(
                       MCSymbolRefExpr::create(EndOfSled, OutContext)));
    EmitToStreamer(*OutStreamer, MCInstBuilder(PPC::NOP));
    EmitToStreamer(
        *OutStreamer,
        MCInstBuilder(PPC::STD).addReg(PPC::X0).addImm(-8).addReg(PPC::X1));
    EmitToStreamer(*OutStreamer, MCInstBuilder(PPC::MFLR8).addReg(PPC::X0));
    EmitToStreamer(*OutStreamer,
                   MCInstBuilder(PPC::BL8_NOP)
                       .addExpr(MCSymbolRefExpr::create(
                           OutContext.getOrCreateSymbol("__xray_FunctionEntry"),
                           OutContext)));
    EmitToStreamer(*OutStreamer, MCInstBuilder(PPC::MTLR8).addReg(PPC::X0));
    OutStreamer->emitLabel(EndOfSled);
    recordSled(BeginOfSled, *MI, SledKind::FUNCTION_ENTER, 2);
    break;
  }
  case TargetOpcode::PATCHABLE_RET: {
    unsigned RetOpcode = MI->getOperand(0).getImm();
    MCInst RetInst;
    RetInst.setOpcode(RetOpcode);
    for (const auto &MO :
         make_range(std::next(MI->operands_begin()), MI->operands_end())) {
      MCOperand MCOp;
      if (LowerPPCMachineOperandToMCOperand(MO, MCOp, *this))
        RetInst.addOperand(MCOp);
    }

    bool IsConditional;
    if (RetOpcode == PPC::BCCLR) {
      IsConditional = true;
    } else if (RetOpcode == PPC::TCRETURNdi8 || RetOpcode == PPC::TCRETURNri8 ||
               RetOpcode == PPC::TCRETURNai8) {
      break;
    } else if (RetOpcode == PPC::BLR8 || RetOpcode == PPC::TAILB8) {
      IsConditional = false;
    } else {
      EmitToStreamer(*OutStreamer, RetInst);
      break;
    }

    MCSymbol *FallthroughLabel;
    if (IsConditional) {
      // Before:
      //   bgtlr cr0
      //
      // After:
      //   ble cr0, .end
      // .p2align 3
      // .begin:
      //   blr    # lis 0, FuncId[16..32]
      //   nop    # li  0, FuncId[0..15]
      //   std 0, -8(1)
      //   mflr 0
      //   bl __xray_FunctionExit
      //   mtlr 0
      //   blr
      // .end:
      //
      // Update compiler-rt/lib/xray/xray_powerpc64.cc accordingly when number
      // of instructions change.
      FallthroughLabel = OutContext.createTempSymbol();
      EmitToStreamer(
          *OutStreamer,
          MCInstBuilder(PPC::BCC)
              .addImm(PPC::InvertPredicate(
                  static_cast<PPC::Predicate>(MI->getOperand(1).getImm())))
              .addReg(MI->getOperand(2).getReg())
              .addExpr(MCSymbolRefExpr::create(FallthroughLabel, OutContext)));
      RetInst = MCInst();
      RetInst.setOpcode(PPC::BLR8);
    }
    // .p2align 3
    // .begin:
    //   b(lr)? # lis 0, FuncId[16..32]
    //   nop    # li  0, FuncId[0..15]
    //   std 0, -8(1)
    //   mflr 0
    //   bl __xray_FunctionExit
    //   mtlr 0
    //   b(lr)?
    //
    // Update compiler-rt/lib/xray/xray_powerpc64.cc accordingly when number
    // of instructions change.
    OutStreamer->emitCodeAlignment(8);
    MCSymbol *BeginOfSled = OutContext.createTempSymbol();
    OutStreamer->emitLabel(BeginOfSled);
    EmitToStreamer(*OutStreamer, RetInst);
    EmitToStreamer(*OutStreamer, MCInstBuilder(PPC::NOP));
    EmitToStreamer(
        *OutStreamer,
        MCInstBuilder(PPC::STD).addReg(PPC::X0).addImm(-8).addReg(PPC::X1));
    EmitToStreamer(*OutStreamer, MCInstBuilder(PPC::MFLR8).addReg(PPC::X0));
    EmitToStreamer(*OutStreamer,
                   MCInstBuilder(PPC::BL8_NOP)
                       .addExpr(MCSymbolRefExpr::create(
                           OutContext.getOrCreateSymbol("__xray_FunctionExit"),
                           OutContext)));
    EmitToStreamer(*OutStreamer, MCInstBuilder(PPC::MTLR8).addReg(PPC::X0));
    EmitToStreamer(*OutStreamer, RetInst);
    if (IsConditional)
      OutStreamer->emitLabel(FallthroughLabel);
    recordSled(BeginOfSled, *MI, SledKind::FUNCTION_EXIT, 2);
    break;
  }
  case TargetOpcode::PATCHABLE_FUNCTION_EXIT:
    llvm_unreachable("PATCHABLE_FUNCTION_EXIT should never be emitted");
  case TargetOpcode::PATCHABLE_TAIL_CALL:
    // TODO: Define a trampoline `__xray_FunctionTailExit` and differentiate a
    // normal function exit from a tail exit.
    llvm_unreachable("Tail call is handled in the normal case. See comments "
                     "around this assert.");
  }
}

void PPCLinuxAsmPrinter::emitStartOfAsmFile(Module &M) {
  if (static_cast<const PPCTargetMachine &>(TM).isELFv2ABI()) {
    PPCTargetStreamer *TS =
      static_cast<PPCTargetStreamer *>(OutStreamer->getTargetStreamer());

    if (TS)
      TS->emitAbiVersion(2);
  }

  if (static_cast<const PPCTargetMachine &>(TM).isPPC64() ||
      !isPositionIndependent())
    return AsmPrinter::emitStartOfAsmFile(M);

  if (M.getPICLevel() == PICLevel::SmallPIC)
    return AsmPrinter::emitStartOfAsmFile(M);

  OutStreamer->SwitchSection(OutContext.getELFSection(
      ".got2", ELF::SHT_PROGBITS, ELF::SHF_WRITE | ELF::SHF_ALLOC));

  MCSymbol *TOCSym = OutContext.getOrCreateSymbol(Twine(".LTOC"));
  MCSymbol *CurrentPos = OutContext.createTempSymbol();

  OutStreamer->emitLabel(CurrentPos);

  // The GOT pointer points to the middle of the GOT, in order to reference the
  // entire 64kB range.  0x8000 is the midpoint.
  const MCExpr *tocExpr =
    MCBinaryExpr::createAdd(MCSymbolRefExpr::create(CurrentPos, OutContext),
                            MCConstantExpr::create(0x8000, OutContext),
                            OutContext);

  OutStreamer->emitAssignment(TOCSym, tocExpr);

  OutStreamer->SwitchSection(getObjFileLowering().getTextSection());
}

void PPCLinuxAsmPrinter::emitFunctionEntryLabel() {
  // linux/ppc32 - Normal entry label.
  if (!Subtarget->isPPC64() &&
      (!isPositionIndependent() ||
       MF->getFunction().getParent()->getPICLevel() == PICLevel::SmallPIC))
    return AsmPrinter::emitFunctionEntryLabel();

  if (!Subtarget->isPPC64()) {
    const PPCFunctionInfo *PPCFI = MF->getInfo<PPCFunctionInfo>();
    if (PPCFI->usesPICBase() && !Subtarget->isSecurePlt()) {
      MCSymbol *RelocSymbol = PPCFI->getPICOffsetSymbol(*MF);
      MCSymbol *PICBase = MF->getPICBaseSymbol();
      OutStreamer->emitLabel(RelocSymbol);

      const MCExpr *OffsExpr =
        MCBinaryExpr::createSub(
          MCSymbolRefExpr::create(OutContext.getOrCreateSymbol(Twine(".LTOC")),
                                                               OutContext),
                                  MCSymbolRefExpr::create(PICBase, OutContext),
          OutContext);
      OutStreamer->emitValue(OffsExpr, 4);
      OutStreamer->emitLabel(CurrentFnSym);
      return;
    } else
      return AsmPrinter::emitFunctionEntryLabel();
  }

  // ELFv2 ABI - Normal entry label.
  if (Subtarget->isELFv2ABI()) {
    // In the Large code model, we allow arbitrary displacements between
    // the text section and its associated TOC section.  We place the
    // full 8-byte offset to the TOC in memory immediately preceding
    // the function global entry point.
    if (TM.getCodeModel() == CodeModel::Large
        && !MF->getRegInfo().use_empty(PPC::X2)) {
      const PPCFunctionInfo *PPCFI = MF->getInfo<PPCFunctionInfo>();

      MCSymbol *TOCSymbol = OutContext.getOrCreateSymbol(StringRef(".TOC."));
      MCSymbol *GlobalEPSymbol = PPCFI->getGlobalEPSymbol(*MF);
      const MCExpr *TOCDeltaExpr =
        MCBinaryExpr::createSub(MCSymbolRefExpr::create(TOCSymbol, OutContext),
                                MCSymbolRefExpr::create(GlobalEPSymbol,
                                                        OutContext),
                                OutContext);

      OutStreamer->emitLabel(PPCFI->getTOCOffsetSymbol(*MF));
      OutStreamer->emitValue(TOCDeltaExpr, 8);
    }
    return AsmPrinter::emitFunctionEntryLabel();
  }

  // Emit an official procedure descriptor.
  MCSectionSubPair Current = OutStreamer->getCurrentSection();
  MCSectionELF *Section = OutStreamer->getContext().getELFSection(
      ".opd", ELF::SHT_PROGBITS, ELF::SHF_WRITE | ELF::SHF_ALLOC);
  OutStreamer->SwitchSection(Section);
  OutStreamer->emitLabel(CurrentFnSym);
  OutStreamer->emitValueToAlignment(8);
  MCSymbol *Symbol1 = CurrentFnSymForSize;
  // Generates a R_PPC64_ADDR64 (from FK_DATA_8) relocation for the function
  // entry point.
  OutStreamer->emitValue(MCSymbolRefExpr::create(Symbol1, OutContext),
                         8 /*size*/);
  MCSymbol *Symbol2 = OutContext.getOrCreateSymbol(StringRef(".TOC."));
  // Generates a R_PPC64_TOC relocation for TOC base insertion.
  OutStreamer->emitValue(
    MCSymbolRefExpr::create(Symbol2, MCSymbolRefExpr::VK_PPC_TOCBASE, OutContext),
    8/*size*/);
  // Emit a null environment pointer.
  OutStreamer->emitIntValue(0, 8 /* size */);
  OutStreamer->SwitchSection(Current.first, Current.second);
}

void PPCLinuxAsmPrinter::emitEndOfAsmFile(Module &M) {
  const DataLayout &DL = getDataLayout();

  bool isPPC64 = DL.getPointerSizeInBits() == 64;

  PPCTargetStreamer *TS =
      static_cast<PPCTargetStreamer *>(OutStreamer->getTargetStreamer());

  if (!TOC.empty()) {
    const char *Name = isPPC64 ? ".toc" : ".got2";
    MCSectionELF *Section = OutContext.getELFSection(
        Name, ELF::SHT_PROGBITS, ELF::SHF_WRITE | ELF::SHF_ALLOC);
    OutStreamer->SwitchSection(Section);
    if (!isPPC64)
      OutStreamer->emitValueToAlignment(4);

    for (const auto &TOCMapPair : TOC) {
      const MCSymbol *const TOCEntryTarget = TOCMapPair.first;
      MCSymbol *const TOCEntryLabel = TOCMapPair.second;

      OutStreamer->emitLabel(TOCEntryLabel);
      if (isPPC64 && TS != nullptr)
        TS->emitTCEntry(*TOCEntryTarget);
      else
        OutStreamer->emitSymbolValue(TOCEntryTarget, 4);
    }
  }

  PPCAsmPrinter::emitEndOfAsmFile(M);
}

/// EmitFunctionBodyStart - Emit a global entry point prefix for ELFv2.
void PPCLinuxAsmPrinter::emitFunctionBodyStart() {
  // In the ELFv2 ABI, in functions that use the TOC register, we need to
  // provide two entry points.  The ABI guarantees that when calling the
  // local entry point, r2 is set up by the caller to contain the TOC base
  // for this function, and when calling the global entry point, r12 is set
  // up by the caller to hold the address of the global entry point.  We
  // thus emit a prefix sequence along the following lines:
  //
  // func:
  // .Lfunc_gepNN:
  //         # global entry point
  //         addis r2,r12,(.TOC.-.Lfunc_gepNN)@ha
  //         addi  r2,r2,(.TOC.-.Lfunc_gepNN)@l
  // .Lfunc_lepNN:
  //         .localentry func, .Lfunc_lepNN-.Lfunc_gepNN
  //         # local entry point, followed by function body
  //
  // For the Large code model, we create
  //
  // .Lfunc_tocNN:
  //         .quad .TOC.-.Lfunc_gepNN      # done by EmitFunctionEntryLabel
  // func:
  // .Lfunc_gepNN:
  //         # global entry point
  //         ld    r2,.Lfunc_tocNN-.Lfunc_gepNN(r12)
  //         add   r2,r2,r12
  // .Lfunc_lepNN:
  //         .localentry func, .Lfunc_lepNN-.Lfunc_gepNN
  //         # local entry point, followed by function body
  //
  // This ensures we have r2 set up correctly while executing the function
  // body, no matter which entry point is called.
  const PPCFunctionInfo *PPCFI = MF->getInfo<PPCFunctionInfo>();
  const bool UsesX2OrR2 = !MF->getRegInfo().use_empty(PPC::X2) ||
                          !MF->getRegInfo().use_empty(PPC::R2);
  const bool PCrelGEPRequired = Subtarget->isUsingPCRelativeCalls() &&
                                UsesX2OrR2 && PPCFI->usesTOCBasePtr();
  const bool NonPCrelGEPRequired = !Subtarget->isUsingPCRelativeCalls() &&
                                   Subtarget->isELFv2ABI() && UsesX2OrR2;

  // Only do all that if the function uses R2 as the TOC pointer
  // in the first place. We don't need the global entry point if the
  // function uses R2 as an allocatable register.
  if (NonPCrelGEPRequired || PCrelGEPRequired) {
    // Note: The logic here must be synchronized with the code in the
    // branch-selection pass which sets the offset of the first block in the
    // function. This matters because it affects the alignment.
    MCSymbol *GlobalEntryLabel = PPCFI->getGlobalEPSymbol(*MF);
    OutStreamer->emitLabel(GlobalEntryLabel);
    const MCSymbolRefExpr *GlobalEntryLabelExp =
      MCSymbolRefExpr::create(GlobalEntryLabel, OutContext);

    if (TM.getCodeModel() != CodeModel::Large) {
      MCSymbol *TOCSymbol = OutContext.getOrCreateSymbol(StringRef(".TOC."));
      const MCExpr *TOCDeltaExpr =
        MCBinaryExpr::createSub(MCSymbolRefExpr::create(TOCSymbol, OutContext),
                                GlobalEntryLabelExp, OutContext);

      const MCExpr *TOCDeltaHi = PPCMCExpr::createHa(TOCDeltaExpr, OutContext);
      EmitToStreamer(*OutStreamer, MCInstBuilder(PPC::ADDIS)
                                   .addReg(PPC::X2)
                                   .addReg(PPC::X12)
                                   .addExpr(TOCDeltaHi));

      const MCExpr *TOCDeltaLo = PPCMCExpr::createLo(TOCDeltaExpr, OutContext);
      EmitToStreamer(*OutStreamer, MCInstBuilder(PPC::ADDI)
                                   .addReg(PPC::X2)
                                   .addReg(PPC::X2)
                                   .addExpr(TOCDeltaLo));
    } else {
      MCSymbol *TOCOffset = PPCFI->getTOCOffsetSymbol(*MF);
      const MCExpr *TOCOffsetDeltaExpr =
        MCBinaryExpr::createSub(MCSymbolRefExpr::create(TOCOffset, OutContext),
                                GlobalEntryLabelExp, OutContext);

      EmitToStreamer(*OutStreamer, MCInstBuilder(PPC::LD)
                                   .addReg(PPC::X2)
                                   .addExpr(TOCOffsetDeltaExpr)
                                   .addReg(PPC::X12));
      EmitToStreamer(*OutStreamer, MCInstBuilder(PPC::ADD8)
                                   .addReg(PPC::X2)
                                   .addReg(PPC::X2)
                                   .addReg(PPC::X12));
    }

    MCSymbol *LocalEntryLabel = PPCFI->getLocalEPSymbol(*MF);
    OutStreamer->emitLabel(LocalEntryLabel);
    const MCSymbolRefExpr *LocalEntryLabelExp =
       MCSymbolRefExpr::create(LocalEntryLabel, OutContext);
    const MCExpr *LocalOffsetExp =
      MCBinaryExpr::createSub(LocalEntryLabelExp,
                              GlobalEntryLabelExp, OutContext);

    PPCTargetStreamer *TS =
      static_cast<PPCTargetStreamer *>(OutStreamer->getTargetStreamer());

    if (TS)
      TS->emitLocalEntry(cast<MCSymbolELF>(CurrentFnSym), LocalOffsetExp);
  } else if (Subtarget->isUsingPCRelativeCalls()) {
    // When generating the entry point for a function we have a few scenarios
    // based on whether or not that function uses R2 and whether or not that
    // function makes calls (or is a leaf function).
    // 1) A leaf function that does not use R2 (or treats it as callee-saved
    //    and preserves it). In this case st_other=0 and both
    //    the local and global entry points for the function are the same.
    //    No special entry point code is required.
    // 2) A function uses the TOC pointer R2. This function may or may not have
    //    calls. In this case st_other=[2,6] and the global and local entry
    //    points are different. Code to correctly setup the TOC pointer in R2
    //    is put between the global and local entry points. This case is
    //    covered by the if statatement above.
    // 3) A function does not use the TOC pointer R2 but does have calls.
    //    In this case st_other=1 since we do not know whether or not any
    //    of the callees clobber R2. This case is dealt with in this else if
    //    block. Tail calls are considered calls and the st_other should also
    //    be set to 1 in that case as well.
    // 4) The function does not use the TOC pointer but R2 is used inside
    //    the function. In this case st_other=1 once again.
    // 5) This function uses inline asm. We mark R2 as reserved if the function
    //    has inline asm as we have to assume that it may be used.
    if (MF->getFrameInfo().hasCalls() || MF->getFrameInfo().hasTailCall() ||
        MF->hasInlineAsm() || (!PPCFI->usesTOCBasePtr() && UsesX2OrR2)) {
      PPCTargetStreamer *TS =
          static_cast<PPCTargetStreamer *>(OutStreamer->getTargetStreamer());
      if (TS)
        TS->emitLocalEntry(cast<MCSymbolELF>(CurrentFnSym),
                           MCConstantExpr::create(1, OutContext));
    }
  }
}

/// EmitFunctionBodyEnd - Print the traceback table before the .size
/// directive.
///
void PPCLinuxAsmPrinter::emitFunctionBodyEnd() {
  // Only the 64-bit target requires a traceback table.  For now,
  // we only emit the word of zeroes that GDB requires to find
  // the end of the function, and zeroes for the eight-byte
  // mandatory fields.
  // FIXME: We should fill in the eight-byte mandatory fields as described in
  // the PPC64 ELF ABI (this is a low-priority item because GDB does not
  // currently make use of these fields).
  if (Subtarget->isPPC64()) {
    OutStreamer->emitIntValue(0, 4/*size*/);
    OutStreamer->emitIntValue(0, 8/*size*/);
  }
}

void PPCAIXAsmPrinter::emitLinkage(const GlobalValue *GV,
                                   MCSymbol *GVSym) const {

  assert(MAI->hasVisibilityOnlyWithLinkage() &&
         "AIX's linkage directives take a visibility setting.");

  MCSymbolAttr LinkageAttr = MCSA_Invalid;
  switch (GV->getLinkage()) {
  case GlobalValue::ExternalLinkage:
    LinkageAttr = GV->isDeclaration() ? MCSA_Extern : MCSA_Global;
    break;
  case GlobalValue::LinkOnceAnyLinkage:
  case GlobalValue::LinkOnceODRLinkage:
  case GlobalValue::WeakAnyLinkage:
  case GlobalValue::WeakODRLinkage:
  case GlobalValue::ExternalWeakLinkage:
    LinkageAttr = MCSA_Weak;
    break;
  case GlobalValue::AvailableExternallyLinkage:
    LinkageAttr = MCSA_Extern;
    break;
  case GlobalValue::PrivateLinkage:
    return;
  case GlobalValue::InternalLinkage:
    assert(GV->getVisibility() == GlobalValue::DefaultVisibility &&
           "InternalLinkage should not have other visibility setting.");
    LinkageAttr = MCSA_LGlobal;
    break;
  case GlobalValue::AppendingLinkage:
    llvm_unreachable("Should never emit this");
  case GlobalValue::CommonLinkage:
    llvm_unreachable("CommonLinkage of XCOFF should not come to this path");
  }

  assert(LinkageAttr != MCSA_Invalid && "LinkageAttr should not MCSA_Invalid.");

  MCSymbolAttr VisibilityAttr = MCSA_Invalid;
  switch (GV->getVisibility()) {

  // TODO: "exported" and "internal" Visibility needs to go here.
  case GlobalValue::DefaultVisibility:
    break;
  case GlobalValue::HiddenVisibility:
    VisibilityAttr = MAI->getHiddenVisibilityAttr();
    break;
  case GlobalValue::ProtectedVisibility:
    VisibilityAttr = MAI->getProtectedVisibilityAttr();
    break;
  }

  OutStreamer->emitXCOFFSymbolLinkageWithVisibility(GVSym, LinkageAttr,
                                                    VisibilityAttr);
}

void PPCAIXAsmPrinter::SetupMachineFunction(MachineFunction &MF) {
  // Setup CurrentFnDescSym and its containing csect.
  MCSectionXCOFF *FnDescSec =
      cast<MCSectionXCOFF>(getObjFileLowering().getSectionForFunctionDescriptor(
          &MF.getFunction(), TM));
  FnDescSec->setAlignment(Align(Subtarget->isPPC64() ? 8 : 4));

  CurrentFnDescSym = FnDescSec->getQualNameSymbol();

  return AsmPrinter::SetupMachineFunction(MF);
}

void PPCAIXAsmPrinter::ValidateGV(const GlobalVariable *GV) {
  // Early error checking limiting what is supported.
  if (GV->isThreadLocal())
    report_fatal_error("Thread local not yet supported on AIX.");

  if (GV->hasSection())
    report_fatal_error("Custom section for Data not yet supported.");

  if (GV->hasComdat())
    report_fatal_error("COMDAT not yet supported by AIX.");
}

static bool isSpecialLLVMGlobalArrayToSkip(const GlobalVariable *GV) {
  return GV->hasAppendingLinkage() &&
         StringSwitch<bool>(GV->getName())
             // TODO: Linker could still eliminate the GV if we just skip
             // handling llvm.used array. Skipping them for now until we or the
             // AIX OS team come up with a good solution.
             .Case("llvm.used", true)
             // It's correct to just skip llvm.compiler.used array here.
             .Case("llvm.compiler.used", true)
             .Default(false);
}

static bool isSpecialLLVMGlobalArrayForStaticInit(const GlobalVariable *GV) {
  return StringSwitch<bool>(GV->getName())
      .Cases("llvm.global_ctors", "llvm.global_dtors", true)
      .Default(false);
}

void PPCAIXAsmPrinter::emitGlobalVariable(const GlobalVariable *GV) {
  // Special LLVM global arrays have been handled at the initialization.
  if (isSpecialLLVMGlobalArrayToSkip(GV) || isSpecialLLVMGlobalArrayForStaticInit(GV))
    return;

  assert(!GV->getName().startswith("llvm.") &&
         "Unhandled intrinsic global variable.");
  ValidateGV(GV);

  MCSymbolXCOFF *GVSym = cast<MCSymbolXCOFF>(getSymbol(GV));

  if (GV->isDeclarationForLinker()) {
    emitLinkage(GV, GVSym);
    return;
  }

  SectionKind GVKind = getObjFileLowering().getKindForGlobal(GV, TM);
  if (!GVKind.isGlobalWriteableData() && !GVKind.isReadOnly())
    report_fatal_error("Encountered a global variable kind that is "
                       "not supported yet.");

  MCSectionXCOFF *Csect = cast<MCSectionXCOFF>(
      getObjFileLowering().SectionForGlobal(GV, GVKind, TM));

  // Switch to the containing csect.
  OutStreamer->SwitchSection(Csect);

  const DataLayout &DL = GV->getParent()->getDataLayout();

  // Handle common symbols.
  if (GVKind.isCommon() || GVKind.isBSSLocal()) {
    Align Alignment = GV->getAlign().getValueOr(DL.getPreferredAlign(GV));
    uint64_t Size = DL.getTypeAllocSize(GV->getType()->getElementType());
    GVSym->setStorageClass(
        TargetLoweringObjectFileXCOFF::getStorageClassForGlobal(GV));

    if (GVKind.isBSSLocal())
      OutStreamer->emitXCOFFLocalCommonSymbol(
          OutContext.getOrCreateSymbol(GVSym->getSymbolTableName()), Size,
          GVSym, Alignment.value());
    else
      OutStreamer->emitCommonSymbol(GVSym, Size, Alignment.value());
    return;
  }

  MCSymbol *EmittedInitSym = GVSym;
  emitLinkage(GV, EmittedInitSym);
  emitAlignment(getGVAlignment(GV, DL), GV);
  OutStreamer->emitLabel(EmittedInitSym);
  // Emit aliasing label for global variable.
  llvm::for_each(GOAliasMap[GV], [this](const GlobalAlias *Alias) {
    OutStreamer->emitLabel(getSymbol(Alias));
  });
  emitGlobalConstant(GV->getParent()->getDataLayout(), GV->getInitializer());
}

void PPCAIXAsmPrinter::emitFunctionDescriptor() {
  const DataLayout &DL = getDataLayout();
  const unsigned PointerSize = DL.getPointerSizeInBits() == 64 ? 8 : 4;

  MCSectionSubPair Current = OutStreamer->getCurrentSection();
  // Emit function descriptor.
  OutStreamer->SwitchSection(
      cast<MCSymbolXCOFF>(CurrentFnDescSym)->getRepresentedCsect());

  // Emit aliasing label for function descriptor csect.
  llvm::for_each(GOAliasMap[&MF->getFunction()],
                 [this](const GlobalAlias *Alias) {
                   OutStreamer->emitLabel(getSymbol(Alias));
                 });

  // Emit function entry point address.
  OutStreamer->emitValue(MCSymbolRefExpr::create(CurrentFnSym, OutContext),
                         PointerSize);
  // Emit TOC base address.
  const MCSymbol *TOCBaseSym =
      cast<MCSectionXCOFF>(getObjFileLowering().getTOCBaseSection())
          ->getQualNameSymbol();
  OutStreamer->emitValue(MCSymbolRefExpr::create(TOCBaseSym, OutContext),
                         PointerSize);
  // Emit a null environment pointer.
  OutStreamer->emitIntValue(0, PointerSize);

  OutStreamer->SwitchSection(Current.first, Current.second);
}

void PPCAIXAsmPrinter::emitFunctionEntryLabel() {
  // It's not necessary to emit the label when we have individual
  // function in its own csect.
  if (!TM.getFunctionSections())
    PPCAsmPrinter::emitFunctionEntryLabel();

  // Emit aliasing label for function entry point label.
  llvm::for_each(
      GOAliasMap[&MF->getFunction()], [this](const GlobalAlias *Alias) {
        OutStreamer->emitLabel(
            getObjFileLowering().getFunctionEntryPointSymbol(Alias, TM));
      });
}

void PPCAIXAsmPrinter::emitEndOfAsmFile(Module &M) {
  // If there are no functions in this module, we will never need to reference
  // the TOC base.
  if (M.empty())
    return;

  // Switch to section to emit TOC base.
  OutStreamer->SwitchSection(getObjFileLowering().getTOCBaseSection());

  PPCTargetStreamer *TS =
      static_cast<PPCTargetStreamer *>(OutStreamer->getTargetStreamer());

  for (auto &I : TOC) {
    // Setup the csect for the current TC entry.
    MCSectionXCOFF *TCEntry = cast<MCSectionXCOFF>(
        getObjFileLowering().getSectionForTOCEntry(I.first, TM));
    OutStreamer->SwitchSection(TCEntry);

    OutStreamer->emitLabel(I.second);
    if (TS != nullptr)
      TS->emitTCEntry(*I.first);
  }
}

bool PPCAIXAsmPrinter::doInitialization(Module &M) {
  const bool Result = PPCAsmPrinter::doInitialization(M);

  auto setCsectAlignment = [this](const GlobalObject *GO) {
    // Declarations have 0 alignment which is set by default.
    if (GO->isDeclarationForLinker())
      return;

    SectionKind GOKind = getObjFileLowering().getKindForGlobal(GO, TM);
    MCSectionXCOFF *Csect = cast<MCSectionXCOFF>(
        getObjFileLowering().SectionForGlobal(GO, GOKind, TM));

    Align GOAlign = getGVAlignment(GO, GO->getParent()->getDataLayout());
    if (GOAlign > Csect->getAlignment())
      Csect->setAlignment(GOAlign);
  };

  // We need to know, up front, the alignment of csects for the assembly path,
  // because once a .csect directive gets emitted, we could not change the
  // alignment value on it.
  for (const auto &G : M.globals()) {
    if (isSpecialLLVMGlobalArrayToSkip(&G))
      continue;

    if (isSpecialLLVMGlobalArrayForStaticInit(&G)) {
      // Generate a format indicator and a unique module id to be a part of
      // the sinit and sterm function names.
      if (FormatIndicatorAndUniqueModId.empty()) {
        std::string UniqueModuleId = getUniqueModuleId(&M);
        if (UniqueModuleId.compare("") != 0)
          // TODO: Use source file full path to generate the unique module id
          // and add a format indicator as a part of function name in case we
          // will support more than one format.
          FormatIndicatorAndUniqueModId = "clang_" + UniqueModuleId.substr(1);
        else
          // Use the Pid and current time as the unique module id when we cannot
          // generate one based on a module's strong external symbols.
          // FIXME: Adjust the comment accordingly after we use source file full
          // path instead.
          FormatIndicatorAndUniqueModId =
              "clangPidTime_" + llvm::itostr(sys::Process::getProcessId()) +
              "_" + llvm::itostr(time(nullptr));
      }

      emitSpecialLLVMGlobal(&G);
      continue;
    }

    setCsectAlignment(&G);
  }

  for (const auto &F : M)
    setCsectAlignment(&F);

  // Construct an aliasing list for each GlobalObject.
  for (const auto &Alias : M.aliases()) {
    const GlobalObject *Base = Alias.getBaseObject();
    if (!Base)
      report_fatal_error(
          "alias without a base object is not yet supported on AIX");
    GOAliasMap[Base].push_back(&Alias);
  }

  return Result;
}

void PPCAIXAsmPrinter::emitInstruction(const MachineInstr *MI) {
  switch (MI->getOpcode()) {
  default:
    break;
  case PPC::BL8:
  case PPC::BL:
  case PPC::BL8_NOP:
  case PPC::BL_NOP: {
    const MachineOperand &MO = MI->getOperand(0);
    if (MO.isSymbol()) {
      MCSymbolXCOFF *S =
          cast<MCSymbolXCOFF>(OutContext.getOrCreateSymbol(MO.getSymbolName()));
      ExtSymSDNodeSymbols.insert(S);
    }
  } break;
  case PPC::BL_TLS:
  case PPC::BL8_TLS:
  case PPC::BL8_TLS_:
  case PPC::BL8_NOP_TLS:
    report_fatal_error("TLS call not yet implemented");
  case PPC::TAILB:
  case PPC::TAILB8:
  case PPC::TAILBA:
  case PPC::TAILBA8:
  case PPC::TAILBCTR:
  case PPC::TAILBCTR8:
    if (MI->getOperand(0).isSymbol())
      report_fatal_error("Tail call for extern symbol not yet supported.");
    break;
  }
  return PPCAsmPrinter::emitInstruction(MI);
}

bool PPCAIXAsmPrinter::doFinalization(Module &M) {
  for (MCSymbol *Sym : ExtSymSDNodeSymbols)
    OutStreamer->emitSymbolAttribute(Sym, MCSA_Extern);
  return PPCAsmPrinter::doFinalization(M);
}

void PPCAIXAsmPrinter::emitXXStructorList(const DataLayout &DL,
                                          const Constant *List, bool IsCtor) {
  SmallVector<Structor, 8> Structors;
  preprocessXXStructorList(DL, List, Structors);
  if (Structors.empty())
    return;

  unsigned Index = 0;
  for (Structor &S : Structors) {
    if (S.Priority != 65535)
      report_fatal_error(
          "prioritized sinit and sterm functions are not yet supported on AIX");

    llvm::GlobalAlias::create(
        GlobalValue::ExternalLinkage,
        (IsCtor ? llvm::Twine("__sinit") : llvm::Twine("__sterm")) +
            llvm::Twine("80000000_", FormatIndicatorAndUniqueModId) +
            llvm::Twine("_", llvm::utostr(Index++)),
        cast<Function>(S.Func));
  }
}

/// createPPCAsmPrinterPass - Returns a pass that prints the PPC assembly code
/// for a MachineFunction to the given output stream, in a format that the
/// Darwin assembler can deal with.
///
static AsmPrinter *
createPPCAsmPrinterPass(TargetMachine &tm,
                        std::unique_ptr<MCStreamer> &&Streamer) {
  if (tm.getTargetTriple().isOSAIX())
    return new PPCAIXAsmPrinter(tm, std::move(Streamer));

  return new PPCLinuxAsmPrinter(tm, std::move(Streamer));
}

// Force static initialization.
extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializePowerPCAsmPrinter() {
  TargetRegistry::RegisterAsmPrinter(getThePPC32Target(),
                                     createPPCAsmPrinterPass);
  TargetRegistry::RegisterAsmPrinter(getThePPC64Target(),
                                     createPPCAsmPrinterPass);
  TargetRegistry::RegisterAsmPrinter(getThePPC64LETarget(),
                                     createPPCAsmPrinterPass);
}
