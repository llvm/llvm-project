//===-- X86MCInstLower.cpp - Convert X86 MachineInstr to an MCInst --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains code to lower X86 MachineInstrs to their corresponding
// MCInst records.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/X86ATTInstPrinter.h"
#include "MCTargetDesc/X86BaseInfo.h"
#include "MCTargetDesc/X86EncodingOptimization.h"
#include "MCTargetDesc/X86InstComments.h"
#include "MCTargetDesc/X86MCAsmInfo.h"
#include "MCTargetDesc/X86ShuffleDecode.h"
#include "MCTargetDesc/X86TargetStreamer.h"
#include "X86AsmPrinter.h"
#include "X86MachineFunctionInfo.h"
#include "X86RegisterInfo.h"
#include "X86ShuffleDecodeConstantPool.h"
#include "X86Subtarget.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/StackMaps.h"
#include "llvm/CodeGen/WinEHFuncInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Mangler.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/CFGuard.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizer.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizerCommon.h"
#include <string>

using namespace llvm;

static cl::opt<bool> EnableBranchHint("enable-branch-hint",
                                      cl::desc("Enable branch hint."),
                                      cl::init(false), cl::Hidden);
static cl::opt<unsigned> BranchHintProbabilityThreshold(
    "branch-hint-probability-threshold",
    cl::desc("The probability threshold of enabling branch hint."),
    cl::init(50), cl::Hidden);

namespace {

/// X86MCInstLower - This class is used to lower an MachineInstr into an MCInst.
class X86MCInstLower {
  MCContext &Ctx;
  const MachineFunction &MF;
  const TargetMachine &TM;
  const MCAsmInfo &MAI;
  X86AsmPrinter &AsmPrinter;

public:
  X86MCInstLower(const MachineFunction &MF, X86AsmPrinter &asmprinter);

  MCOperand LowerMachineOperand(const MachineInstr *MI,
                                const MachineOperand &MO) const;
  void Lower(const MachineInstr *MI, MCInst &OutMI) const;

  MCSymbol *GetSymbolFromOperand(const MachineOperand &MO) const;
  MCOperand LowerSymbolOperand(const MachineOperand &MO, MCSymbol *Sym) const;

private:
  MachineModuleInfoMachO &getMachOMMI() const;
};

} // end anonymous namespace

/// A RAII helper which defines a region of instructions which can't have
/// padding added between them for correctness.
struct NoAutoPaddingScope {
  MCStreamer &OS;
  const bool OldAllowAutoPadding;
  NoAutoPaddingScope(MCStreamer &OS)
      : OS(OS), OldAllowAutoPadding(OS.getAllowAutoPadding()) {
    changeAndComment(false);
  }
  ~NoAutoPaddingScope() { changeAndComment(OldAllowAutoPadding); }
  void changeAndComment(bool b) {
    if (b == OS.getAllowAutoPadding())
      return;
    OS.setAllowAutoPadding(b);
    if (b)
      OS.emitRawComment("autopadding");
    else
      OS.emitRawComment("noautopadding");
  }
};

// Emit a minimal sequence of nops spanning NumBytes bytes.
static void emitX86Nops(MCStreamer &OS, unsigned NumBytes,
                        const X86Subtarget *Subtarget);

void X86AsmPrinter::StackMapShadowTracker::count(const MCInst &Inst,
                                                 const MCSubtargetInfo &STI,
                                                 MCCodeEmitter *CodeEmitter) {
  if (InShadow) {
    SmallString<256> Code;
    SmallVector<MCFixup, 4> Fixups;
    CodeEmitter->encodeInstruction(Inst, Code, Fixups, STI);
    CurrentShadowSize += Code.size();
    if (CurrentShadowSize >= RequiredShadowSize)
      InShadow = false; // The shadow is big enough. Stop counting.
  }
}

void X86AsmPrinter::StackMapShadowTracker::emitShadowPadding(
    MCStreamer &OutStreamer, const MCSubtargetInfo &STI) {
  if (InShadow && CurrentShadowSize < RequiredShadowSize) {
    InShadow = false;
    emitX86Nops(OutStreamer, RequiredShadowSize - CurrentShadowSize,
                &MF->getSubtarget<X86Subtarget>());
  }
}

void X86AsmPrinter::EmitAndCountInstruction(MCInst &Inst) {
  OutStreamer->emitInstruction(Inst, getSubtargetInfo());
  SMShadowTracker.count(Inst, getSubtargetInfo(), CodeEmitter.get());
}

X86MCInstLower::X86MCInstLower(const MachineFunction &mf,
                               X86AsmPrinter &asmprinter)
    : Ctx(asmprinter.OutContext), MF(mf), TM(mf.getTarget()),
      MAI(*TM.getMCAsmInfo()), AsmPrinter(asmprinter) {}

MachineModuleInfoMachO &X86MCInstLower::getMachOMMI() const {
  return AsmPrinter.MMI->getObjFileInfo<MachineModuleInfoMachO>();
}

/// GetSymbolFromOperand - Lower an MO_GlobalAddress or MO_ExternalSymbol
/// operand to an MCSymbol.
MCSymbol *X86MCInstLower::GetSymbolFromOperand(const MachineOperand &MO) const {
  const Triple &TT = TM.getTargetTriple();
  if (MO.isGlobal() && TT.isOSBinFormatELF())
    return AsmPrinter.getSymbolPreferLocal(*MO.getGlobal());

  const DataLayout &DL = MF.getDataLayout();
  assert((MO.isGlobal() || MO.isSymbol() || MO.isMBB()) &&
         "Isn't a symbol reference");

  MCSymbol *Sym = nullptr;
  SmallString<128> Name;
  StringRef Suffix;

  switch (MO.getTargetFlags()) {
  case X86II::MO_DLLIMPORT:
    // Handle dllimport linkage.
    Name += "__imp_";
    break;
  case X86II::MO_COFFSTUB:
    Name += ".refptr.";
    break;
  case X86II::MO_DARWIN_NONLAZY:
  case X86II::MO_DARWIN_NONLAZY_PIC_BASE:
    Suffix = "$non_lazy_ptr";
    break;
  }

  if (!Suffix.empty())
    Name += DL.getPrivateGlobalPrefix();

  if (MO.isGlobal()) {
    const GlobalValue *GV = MO.getGlobal();
    AsmPrinter.getNameWithPrefix(Name, GV);
  } else if (MO.isSymbol()) {
    Mangler::getNameWithPrefix(Name, MO.getSymbolName(), DL);
  } else if (MO.isMBB()) {
    assert(Suffix.empty());
    Sym = MO.getMBB()->getSymbol();
  }

  Name += Suffix;
  if (!Sym)
    Sym = Ctx.getOrCreateSymbol(Name);

  // If the target flags on the operand changes the name of the symbol, do that
  // before we return the symbol.
  switch (MO.getTargetFlags()) {
  default:
    break;
  case X86II::MO_COFFSTUB: {
    MachineModuleInfoCOFF &MMICOFF =
        AsmPrinter.MMI->getObjFileInfo<MachineModuleInfoCOFF>();
    MachineModuleInfoImpl::StubValueTy &StubSym = MMICOFF.getGVStubEntry(Sym);
    if (!StubSym.getPointer()) {
      assert(MO.isGlobal() && "Extern symbol not handled yet");
      StubSym = MachineModuleInfoImpl::StubValueTy(
          AsmPrinter.getSymbol(MO.getGlobal()), true);
    }
    break;
  }
  case X86II::MO_DARWIN_NONLAZY:
  case X86II::MO_DARWIN_NONLAZY_PIC_BASE: {
    MachineModuleInfoImpl::StubValueTy &StubSym =
        getMachOMMI().getGVStubEntry(Sym);
    if (!StubSym.getPointer()) {
      assert(MO.isGlobal() && "Extern symbol not handled yet");
      StubSym = MachineModuleInfoImpl::StubValueTy(
          AsmPrinter.getSymbol(MO.getGlobal()),
          !MO.getGlobal()->hasInternalLinkage());
    }
    break;
  }
  }

  return Sym;
}

MCOperand X86MCInstLower::LowerSymbolOperand(const MachineOperand &MO,
                                             MCSymbol *Sym) const {
  // FIXME: We would like an efficient form for this, so we don't have to do a
  // lot of extra uniquing.
  const MCExpr *Expr = nullptr;
  uint16_t Specifier = X86::S_None;

  switch (MO.getTargetFlags()) {
  default:
    llvm_unreachable("Unknown target flag on GV operand");
  case X86II::MO_NO_FLAG: // No flag.
  // These affect the name of the symbol, not any suffix.
  case X86II::MO_DARWIN_NONLAZY:
  case X86II::MO_DLLIMPORT:
  case X86II::MO_COFFSTUB:
    break;

  case X86II::MO_TLVP:
    Specifier = X86::S_TLVP;
    break;
  case X86II::MO_TLVP_PIC_BASE:
    Expr = MCSymbolRefExpr::create(Sym, X86::S_TLVP, Ctx);
    // Subtract the pic base.
    Expr = MCBinaryExpr::createSub(
        Expr, MCSymbolRefExpr::create(MF.getPICBaseSymbol(), Ctx), Ctx);
    break;
  case X86II::MO_SECREL:
    Specifier = uint16_t(X86::S_COFF_SECREL);
    break;
  case X86II::MO_TLSGD:
    Specifier = X86::S_TLSGD;
    break;
  case X86II::MO_TLSLD:
    Specifier = X86::S_TLSLD;
    break;
  case X86II::MO_TLSLDM:
    Specifier = X86::S_TLSLDM;
    break;
  case X86II::MO_GOTTPOFF:
    Specifier = X86::S_GOTTPOFF;
    break;
  case X86II::MO_INDNTPOFF:
    Specifier = X86::S_INDNTPOFF;
    break;
  case X86II::MO_TPOFF:
    Specifier = X86::S_TPOFF;
    break;
  case X86II::MO_DTPOFF:
    Specifier = X86::S_DTPOFF;
    break;
  case X86II::MO_NTPOFF:
    Specifier = X86::S_NTPOFF;
    break;
  case X86II::MO_GOTNTPOFF:
    Specifier = X86::S_GOTNTPOFF;
    break;
  case X86II::MO_GOTPCREL:
    Specifier = X86::S_GOTPCREL;
    break;
  case X86II::MO_GOTPCREL_NORELAX:
    Specifier = X86::S_GOTPCREL_NORELAX;
    break;
  case X86II::MO_GOT:
    Specifier = X86::S_GOT;
    break;
  case X86II::MO_GOTOFF:
    Specifier = X86::S_GOTOFF;
    break;
  case X86II::MO_PLT:
    Specifier = X86::S_PLT;
    break;
  case X86II::MO_ABS8:
    Specifier = X86::S_ABS8;
    break;
  case X86II::MO_PIC_BASE_OFFSET:
  case X86II::MO_DARWIN_NONLAZY_PIC_BASE:
    Expr = MCSymbolRefExpr::create(Sym, Ctx);
    // Subtract the pic base.
    Expr = MCBinaryExpr::createSub(
        Expr, MCSymbolRefExpr::create(MF.getPICBaseSymbol(), Ctx), Ctx);
    if (MO.isJTI()) {
      assert(MAI.doesSetDirectiveSuppressReloc());
      // If .set directive is supported, use it to reduce the number of
      // relocations the assembler will generate for differences between
      // local labels. This is only safe when the symbols are in the same
      // section so we are restricting it to jumptable references.
      MCSymbol *Label = Ctx.createTempSymbol();
      AsmPrinter.OutStreamer->emitAssignment(Label, Expr);
      Expr = MCSymbolRefExpr::create(Label, Ctx);
    }
    break;
  }

  if (!Expr)
    Expr = MCSymbolRefExpr::create(Sym, Specifier, Ctx);

  if (!MO.isJTI() && !MO.isMBB() && MO.getOffset())
    Expr = MCBinaryExpr::createAdd(
        Expr, MCConstantExpr::create(MO.getOffset(), Ctx), Ctx);
  return MCOperand::createExpr(Expr);
}

static unsigned getRetOpcode(const X86Subtarget &Subtarget) {
  return Subtarget.is64Bit() ? X86::RET64 : X86::RET32;
}

MCOperand X86MCInstLower::LowerMachineOperand(const MachineInstr *MI,
                                              const MachineOperand &MO) const {
  switch (MO.getType()) {
  default:
    MI->print(errs());
    llvm_unreachable("unknown operand type");
  case MachineOperand::MO_Register:
    // Ignore all implicit register operands.
    if (MO.isImplicit())
      return MCOperand();
    return MCOperand::createReg(MO.getReg());
  case MachineOperand::MO_Immediate:
    return MCOperand::createImm(MO.getImm());
  case MachineOperand::MO_MachineBasicBlock:
  case MachineOperand::MO_GlobalAddress:
  case MachineOperand::MO_ExternalSymbol:
    return LowerSymbolOperand(MO, GetSymbolFromOperand(MO));
  case MachineOperand::MO_MCSymbol:
    return LowerSymbolOperand(MO, MO.getMCSymbol());
  case MachineOperand::MO_JumpTableIndex:
    return LowerSymbolOperand(MO, AsmPrinter.GetJTISymbol(MO.getIndex()));
  case MachineOperand::MO_ConstantPoolIndex:
    return LowerSymbolOperand(MO, AsmPrinter.GetCPISymbol(MO.getIndex()));
  case MachineOperand::MO_BlockAddress:
    return LowerSymbolOperand(
        MO, AsmPrinter.GetBlockAddressSymbol(MO.getBlockAddress()));
  case MachineOperand::MO_RegisterMask:
    // Ignore call clobbers.
    return MCOperand();
  }
}

// Replace TAILJMP opcodes with their equivalent opcodes that have encoding
// information.
static unsigned convertTailJumpOpcode(unsigned Opcode) {
  switch (Opcode) {
  case X86::TAILJMPr:
    Opcode = X86::JMP32r;
    break;
  case X86::TAILJMPm:
    Opcode = X86::JMP32m;
    break;
  case X86::TAILJMPr64:
    Opcode = X86::JMP64r;
    break;
  case X86::TAILJMPm64:
    Opcode = X86::JMP64m;
    break;
  case X86::TAILJMPr64_REX:
    Opcode = X86::JMP64r_REX;
    break;
  case X86::TAILJMPm64_REX:
    Opcode = X86::JMP64m_REX;
    break;
  case X86::TAILJMPd:
  case X86::TAILJMPd64:
    Opcode = X86::JMP_1;
    break;
  case X86::TAILJMPd_CC:
  case X86::TAILJMPd64_CC:
    Opcode = X86::JCC_1;
    break;
  }

  return Opcode;
}

void X86MCInstLower::Lower(const MachineInstr *MI, MCInst &OutMI) const {
  OutMI.setOpcode(MI->getOpcode());

  for (const MachineOperand &MO : MI->operands())
    if (auto Op = LowerMachineOperand(MI, MO); Op.isValid())
      OutMI.addOperand(Op);

  bool In64BitMode = AsmPrinter.getSubtarget().is64Bit();
  if (X86::optimizeInstFromVEX3ToVEX2(OutMI, MI->getDesc()) ||
      X86::optimizeShiftRotateWithImmediateOne(OutMI) ||
      X86::optimizeVPCMPWithImmediateOneOrSix(OutMI) ||
      X86::optimizeMOVSX(OutMI) || X86::optimizeINCDEC(OutMI, In64BitMode) ||
      X86::optimizeMOV(OutMI, In64BitMode) ||
      X86::optimizeToFixedRegisterOrShortImmediateForm(OutMI))
    return;

  // Handle a few special cases to eliminate operand modifiers.
  switch (OutMI.getOpcode()) {
  case X86::LEA64_32r:
  case X86::LEA64r:
  case X86::LEA16r:
  case X86::LEA32r:
    // LEA should have a segment register, but it must be empty.
    assert(OutMI.getNumOperands() == 1 + X86::AddrNumOperands &&
           "Unexpected # of LEA operands");
    assert(OutMI.getOperand(1 + X86::AddrSegmentReg).getReg() == 0 &&
           "LEA has segment specified!");
    break;
  case X86::MULX32Hrr:
  case X86::MULX32Hrm:
  case X86::MULX64Hrr:
  case X86::MULX64Hrm: {
    // Turn into regular MULX by duplicating the destination.
    unsigned NewOpc;
    switch (OutMI.getOpcode()) {
    default: llvm_unreachable("Invalid opcode");
    case X86::MULX32Hrr: NewOpc = X86::MULX32rr; break;
    case X86::MULX32Hrm: NewOpc = X86::MULX32rm; break;
    case X86::MULX64Hrr: NewOpc = X86::MULX64rr; break;
    case X86::MULX64Hrm: NewOpc = X86::MULX64rm; break;
    }
    OutMI.setOpcode(NewOpc);
    // Duplicate the destination.
    MCRegister DestReg = OutMI.getOperand(0).getReg();
    OutMI.insert(OutMI.begin(), MCOperand::createReg(DestReg));
    break;
  }
  // CALL64r, CALL64pcrel32 - These instructions used to have
  // register inputs modeled as normal uses instead of implicit uses.  As such,
  // they we used to truncate off all but the first operand (the callee). This
  // issue seems to have been fixed at some point. This assert verifies that.
  case X86::CALL64r:
  case X86::CALL64pcrel32:
    assert(OutMI.getNumOperands() == 1 && "Unexpected number of operands!");
    break;
  case X86::EH_RETURN:
  case X86::EH_RETURN64: {
    OutMI = MCInst();
    OutMI.setOpcode(getRetOpcode(AsmPrinter.getSubtarget()));
    break;
  }
  case X86::CLEANUPRET: {
    // Replace CLEANUPRET with the appropriate RET.
    OutMI = MCInst();
    OutMI.setOpcode(getRetOpcode(AsmPrinter.getSubtarget()));
    break;
  }
  case X86::CATCHRET: {
    // Replace CATCHRET with the appropriate RET.
    const X86Subtarget &Subtarget = AsmPrinter.getSubtarget();
    unsigned ReturnReg = In64BitMode ? X86::RAX : X86::EAX;
    OutMI = MCInst();
    OutMI.setOpcode(getRetOpcode(Subtarget));
    OutMI.addOperand(MCOperand::createReg(ReturnReg));
    break;
  }
  // TAILJMPd, TAILJMPd64, TailJMPd_cc - Lower to the correct jump
  // instruction.
  case X86::TAILJMPr:
  case X86::TAILJMPr64:
  case X86::TAILJMPr64_REX:
  case X86::TAILJMPd:
  case X86::TAILJMPd64:
    assert(OutMI.getNumOperands() == 1 && "Unexpected number of operands!");
    OutMI.setOpcode(convertTailJumpOpcode(OutMI.getOpcode()));
    break;
  case X86::TAILJMPd_CC:
  case X86::TAILJMPd64_CC:
    assert(OutMI.getNumOperands() == 2 && "Unexpected number of operands!");
    OutMI.setOpcode(convertTailJumpOpcode(OutMI.getOpcode()));
    break;
  case X86::TAILJMPm:
  case X86::TAILJMPm64:
  case X86::TAILJMPm64_REX:
    assert(OutMI.getNumOperands() == X86::AddrNumOperands &&
           "Unexpected number of operands!");
    OutMI.setOpcode(convertTailJumpOpcode(OutMI.getOpcode()));
    break;
  case X86::MASKMOVDQU:
  case X86::VMASKMOVDQU:
    if (In64BitMode)
      OutMI.setFlags(X86::IP_HAS_AD_SIZE);
    break;
  case X86::BSF16rm:
  case X86::BSF16rr:
  case X86::BSF32rm:
  case X86::BSF32rr:
  case X86::BSF64rm:
  case X86::BSF64rr: {
    // Add an REP prefix to BSF instructions so that new processors can
    // recognize as TZCNT, which has better performance than BSF.
    // BSF and TZCNT have different interpretations on ZF bit. So make sure
    // it won't be used later.
    const MachineOperand *FlagDef =
        MI->findRegisterDefOperand(X86::EFLAGS, /*TRI=*/nullptr);
    if (!MF.getFunction().hasOptSize() && FlagDef && FlagDef->isDead())
      OutMI.setFlags(X86::IP_HAS_REPEAT);
    break;
  }
  default:
    break;
  }
}

void X86AsmPrinter::LowerTlsAddr(X86MCInstLower &MCInstLowering,
                                 const MachineInstr &MI) {
  NoAutoPaddingScope NoPadScope(*OutStreamer);
  bool Is64Bits = getSubtarget().is64Bit();
  bool Is64BitsLP64 = getSubtarget().isTarget64BitLP64();
  MCContext &Ctx = OutStreamer->getContext();

  X86::Specifier Specifier;
  switch (MI.getOpcode()) {
  case X86::TLS_addr32:
  case X86::TLS_addr64:
  case X86::TLS_addrX32:
    Specifier = X86::S_TLSGD;
    break;
  case X86::TLS_base_addr32:
    Specifier = X86::S_TLSLDM;
    break;
  case X86::TLS_base_addr64:
  case X86::TLS_base_addrX32:
    Specifier = X86::S_TLSLD;
    break;
  case X86::TLS_desc32:
  case X86::TLS_desc64:
    Specifier = X86::S_TLSDESC;
    break;
  default:
    llvm_unreachable("unexpected opcode");
  }

  const MCSymbolRefExpr *Sym = MCSymbolRefExpr::create(
      MCInstLowering.GetSymbolFromOperand(MI.getOperand(3)), Specifier, Ctx);

  // Before binutils 2.41, ld has a bogus TLS relaxation error when the GD/LD
  // code sequence using R_X86_64_GOTPCREL (instead of R_X86_64_GOTPCRELX) is
  // attempted to be relaxed to IE/LE (binutils PR24784). Work around the bug by
  // only using GOT when GOTPCRELX is enabled.
  // TODO Delete the workaround when rustc no longer relies on the hack
  bool UseGot = MMI->getModule()->getRtLibUseGOT() &&
                Ctx.getTargetOptions()->X86RelaxRelocations;

  if (Specifier == X86::S_TLSDESC) {
    const MCSymbolRefExpr *Expr = MCSymbolRefExpr::create(
        MCInstLowering.GetSymbolFromOperand(MI.getOperand(3)), X86::S_TLSCALL,
        Ctx);
    EmitAndCountInstruction(
        MCInstBuilder(Is64BitsLP64 ? X86::LEA64r : X86::LEA32r)
            .addReg(Is64BitsLP64 ? X86::RAX : X86::EAX)
            .addReg(Is64Bits ? X86::RIP : X86::EBX)
            .addImm(1)
            .addReg(0)
            .addExpr(Sym)
            .addReg(0));
    EmitAndCountInstruction(
        MCInstBuilder(Is64Bits ? X86::CALL64m : X86::CALL32m)
            .addReg(Is64BitsLP64 ? X86::RAX : X86::EAX)
            .addImm(1)
            .addReg(0)
            .addExpr(Expr)
            .addReg(0));
  } else if (Is64Bits) {
    bool NeedsPadding = Specifier == X86::S_TLSGD;
    if (NeedsPadding && Is64BitsLP64)
      EmitAndCountInstruction(MCInstBuilder(X86::DATA16_PREFIX));
    EmitAndCountInstruction(MCInstBuilder(X86::LEA64r)
                                .addReg(X86::RDI)
                                .addReg(X86::RIP)
                                .addImm(1)
                                .addReg(0)
                                .addExpr(Sym)
                                .addReg(0));
    const MCSymbol *TlsGetAddr = Ctx.getOrCreateSymbol("__tls_get_addr");
    if (NeedsPadding) {
      if (!UseGot)
        EmitAndCountInstruction(MCInstBuilder(X86::DATA16_PREFIX));
      EmitAndCountInstruction(MCInstBuilder(X86::DATA16_PREFIX));
      EmitAndCountInstruction(MCInstBuilder(X86::REX64_PREFIX));
    }
    if (UseGot) {
      const MCExpr *Expr =
          MCSymbolRefExpr::create(TlsGetAddr, X86::S_GOTPCREL, Ctx);
      EmitAndCountInstruction(MCInstBuilder(X86::CALL64m)
                                  .addReg(X86::RIP)
                                  .addImm(1)
                                  .addReg(0)
                                  .addExpr(Expr)
                                  .addReg(0));
    } else {
      EmitAndCountInstruction(
          MCInstBuilder(X86::CALL64pcrel32)
              .addExpr(MCSymbolRefExpr::create(TlsGetAddr, X86::S_PLT, Ctx)));
    }
  } else {
    if (Specifier == X86::S_TLSGD && !UseGot) {
      EmitAndCountInstruction(MCInstBuilder(X86::LEA32r)
                                  .addReg(X86::EAX)
                                  .addReg(0)
                                  .addImm(1)
                                  .addReg(X86::EBX)
                                  .addExpr(Sym)
                                  .addReg(0));
    } else {
      EmitAndCountInstruction(MCInstBuilder(X86::LEA32r)
                                  .addReg(X86::EAX)
                                  .addReg(X86::EBX)
                                  .addImm(1)
                                  .addReg(0)
                                  .addExpr(Sym)
                                  .addReg(0));
    }

    const MCSymbol *TlsGetAddr = Ctx.getOrCreateSymbol("___tls_get_addr");
    if (UseGot) {
      const MCExpr *Expr = MCSymbolRefExpr::create(TlsGetAddr, X86::S_GOT, Ctx);
      EmitAndCountInstruction(MCInstBuilder(X86::CALL32m)
                                  .addReg(X86::EBX)
                                  .addImm(1)
                                  .addReg(0)
                                  .addExpr(Expr)
                                  .addReg(0));
    } else {
      EmitAndCountInstruction(
          MCInstBuilder(X86::CALLpcrel32)
              .addExpr(MCSymbolRefExpr::create(TlsGetAddr, X86::S_PLT, Ctx)));
    }
  }
}

/// Emit the largest nop instruction smaller than or equal to \p NumBytes
/// bytes.  Return the size of nop emitted.
static unsigned emitNop(MCStreamer &OS, unsigned NumBytes,
                        const X86Subtarget *Subtarget) {
  // Determine the longest nop which can be efficiently decoded for the given
  // target cpu.  15-bytes is the longest single NOP instruction, but some
  // platforms can't decode the longest forms efficiently.
  unsigned MaxNopLength = 1;
  if (Subtarget->is64Bit()) {
    // FIXME: We can use NOOPL on 32-bit targets with FeatureNOPL, but the
    // IndexReg/BaseReg below need to be updated.
    if (Subtarget->hasFeature(X86::TuningFast7ByteNOP))
      MaxNopLength = 7;
    else if (Subtarget->hasFeature(X86::TuningFast15ByteNOP))
      MaxNopLength = 15;
    else if (Subtarget->hasFeature(X86::TuningFast11ByteNOP))
      MaxNopLength = 11;
    else
      MaxNopLength = 10;
  } if (Subtarget->is32Bit())
    MaxNopLength = 2;

  // Cap a single nop emission at the profitable value for the target
  NumBytes = std::min(NumBytes, MaxNopLength);

  unsigned NopSize;
  unsigned Opc, BaseReg, ScaleVal, IndexReg, Displacement, SegmentReg;
  IndexReg = Displacement = SegmentReg = 0;
  BaseReg = X86::RAX;
  ScaleVal = 1;
  switch (NumBytes) {
  case 0:
    llvm_unreachable("Zero nops?");
    break;
  case 1:
    NopSize = 1;
    Opc = X86::NOOP;
    break;
  case 2:
    NopSize = 2;
    Opc = X86::XCHG16ar;
    break;
  case 3:
    NopSize = 3;
    Opc = X86::NOOPL;
    break;
  case 4:
    NopSize = 4;
    Opc = X86::NOOPL;
    Displacement = 8;
    break;
  case 5:
    NopSize = 5;
    Opc = X86::NOOPL;
    Displacement = 8;
    IndexReg = X86::RAX;
    break;
  case 6:
    NopSize = 6;
    Opc = X86::NOOPW;
    Displacement = 8;
    IndexReg = X86::RAX;
    break;
  case 7:
    NopSize = 7;
    Opc = X86::NOOPL;
    Displacement = 512;
    break;
  case 8:
    NopSize = 8;
    Opc = X86::NOOPL;
    Displacement = 512;
    IndexReg = X86::RAX;
    break;
  case 9:
    NopSize = 9;
    Opc = X86::NOOPW;
    Displacement = 512;
    IndexReg = X86::RAX;
    break;
  default:
    NopSize = 10;
    Opc = X86::NOOPW;
    Displacement = 512;
    IndexReg = X86::RAX;
    SegmentReg = X86::CS;
    break;
  }

  unsigned NumPrefixes = std::min(NumBytes - NopSize, 5U);
  NopSize += NumPrefixes;
  for (unsigned i = 0; i != NumPrefixes; ++i)
    OS.emitBytes("\x66");

  switch (Opc) {
  default: llvm_unreachable("Unexpected opcode");
  case X86::NOOP:
    OS.emitInstruction(MCInstBuilder(Opc), *Subtarget);
    break;
  case X86::XCHG16ar:
    OS.emitInstruction(MCInstBuilder(Opc).addReg(X86::AX).addReg(X86::AX),
                       *Subtarget);
    break;
  case X86::NOOPL:
  case X86::NOOPW:
    OS.emitInstruction(MCInstBuilder(Opc)
                           .addReg(BaseReg)
                           .addImm(ScaleVal)
                           .addReg(IndexReg)
                           .addImm(Displacement)
                           .addReg(SegmentReg),
                       *Subtarget);
    break;
  }
  assert(NopSize <= NumBytes && "We overemitted?");
  return NopSize;
}

/// Emit the optimal amount of multi-byte nops on X86.
static void emitX86Nops(MCStreamer &OS, unsigned NumBytes,
                        const X86Subtarget *Subtarget) {
  unsigned NopsToEmit = NumBytes;
  (void)NopsToEmit;
  while (NumBytes) {
    NumBytes -= emitNop(OS, NumBytes, Subtarget);
    assert(NopsToEmit >= NumBytes && "Emitted more than I asked for!");
  }
}

void X86AsmPrinter::LowerSTATEPOINT(const MachineInstr &MI,
                                    X86MCInstLower &MCIL) {
  assert(Subtarget->is64Bit() && "Statepoint currently only supports X86-64");

  NoAutoPaddingScope NoPadScope(*OutStreamer);

  StatepointOpers SOpers(&MI);
  if (unsigned PatchBytes = SOpers.getNumPatchBytes()) {
    emitX86Nops(*OutStreamer, PatchBytes, Subtarget);
  } else {
    // Lower call target and choose correct opcode
    const MachineOperand &CallTarget = SOpers.getCallTarget();
    MCOperand CallTargetMCOp;
    unsigned CallOpcode;
    switch (CallTarget.getType()) {
    case MachineOperand::MO_GlobalAddress:
    case MachineOperand::MO_ExternalSymbol:
      CallTargetMCOp = MCIL.LowerSymbolOperand(
          CallTarget, MCIL.GetSymbolFromOperand(CallTarget));
      CallOpcode = X86::CALL64pcrel32;
      // Currently, we only support relative addressing with statepoints.
      // Otherwise, we'll need a scratch register to hold the target
      // address.  You'll fail asserts during load & relocation if this
      // symbol is to far away. (TODO: support non-relative addressing)
      break;
    case MachineOperand::MO_Immediate:
      CallTargetMCOp = MCOperand::createImm(CallTarget.getImm());
      CallOpcode = X86::CALL64pcrel32;
      // Currently, we only support relative addressing with statepoints.
      // Otherwise, we'll need a scratch register to hold the target
      // immediate.  You'll fail asserts during load & relocation if this
      // address is to far away. (TODO: support non-relative addressing)
      break;
    case MachineOperand::MO_Register:
      // FIXME: Add retpoline support and remove this.
      if (Subtarget->useIndirectThunkCalls())
        report_fatal_error("Lowering register statepoints with thunks not "
                           "yet implemented.");
      CallTargetMCOp = MCOperand::createReg(CallTarget.getReg());
      CallOpcode = X86::CALL64r;
      break;
    default:
      llvm_unreachable("Unsupported operand type in statepoint call target");
      break;
    }

    // Emit call
    MCInst CallInst;
    CallInst.setOpcode(CallOpcode);
    CallInst.addOperand(CallTargetMCOp);
    OutStreamer->emitInstruction(CallInst, getSubtargetInfo());
    maybeEmitNopAfterCallForWindowsEH(&MI);
  }

  // Record our statepoint node in the same section used by STACKMAP
  // and PATCHPOINT
  auto &Ctx = OutStreamer->getContext();
  MCSymbol *MILabel = Ctx.createTempSymbol();
  OutStreamer->emitLabel(MILabel);
  SM.recordStatepoint(*MILabel, MI);
}

void X86AsmPrinter::LowerFAULTING_OP(const MachineInstr &FaultingMI,
                                     X86MCInstLower &MCIL) {
  // FAULTING_LOAD_OP <def>, <faltinf type>, <MBB handler>,
  //                  <opcode>, <operands>

  NoAutoPaddingScope NoPadScope(*OutStreamer);

  Register DefRegister = FaultingMI.getOperand(0).getReg();
  FaultMaps::FaultKind FK =
      static_cast<FaultMaps::FaultKind>(FaultingMI.getOperand(1).getImm());
  MCSymbol *HandlerLabel = FaultingMI.getOperand(2).getMBB()->getSymbol();
  unsigned Opcode = FaultingMI.getOperand(3).getImm();
  unsigned OperandsBeginIdx = 4;

  auto &Ctx = OutStreamer->getContext();
  MCSymbol *FaultingLabel = Ctx.createTempSymbol();
  OutStreamer->emitLabel(FaultingLabel);

  assert(FK < FaultMaps::FaultKindMax && "Invalid Faulting Kind!");
  FM.recordFaultingOp(FK, FaultingLabel, HandlerLabel);

  MCInst MI;
  MI.setOpcode(Opcode);

  if (DefRegister != X86::NoRegister)
    MI.addOperand(MCOperand::createReg(DefRegister));

  for (const MachineOperand &MO :
       llvm::drop_begin(FaultingMI.operands(), OperandsBeginIdx))
    if (auto Op = MCIL.LowerMachineOperand(&FaultingMI, MO); Op.isValid())
      MI.addOperand(Op);

  OutStreamer->AddComment("on-fault: " + HandlerLabel->getName());
  OutStreamer->emitInstruction(MI, getSubtargetInfo());
}

void X86AsmPrinter::LowerFENTRY_CALL(const MachineInstr &MI,
                                     X86MCInstLower &MCIL) {
  bool Is64Bits = Subtarget->is64Bit();
  MCContext &Ctx = OutStreamer->getContext();
  MCSymbol *fentry = Ctx.getOrCreateSymbol("__fentry__");
  const MCSymbolRefExpr *Op = MCSymbolRefExpr::create(fentry, Ctx);

  EmitAndCountInstruction(
      MCInstBuilder(Is64Bits ? X86::CALL64pcrel32 : X86::CALLpcrel32)
          .addExpr(Op));
}

void X86AsmPrinter::LowerKCFI_CHECK(const MachineInstr &MI) {
  assert(std::next(MI.getIterator())->isCall() &&
         "KCFI_CHECK not followed by a call instruction");

  // Adjust the offset for patchable-function-prefix. X86InstrInfo::getNop()
  // returns a 1-byte X86::NOOP, which means the offset is the same in
  // bytes.  This assumes that patchable-function-prefix is the same for all
  // functions.
  const MachineFunction &MF = *MI.getMF();
  int64_t PrefixNops = 0;
  (void)MF.getFunction()
      .getFnAttribute("patchable-function-prefix")
      .getValueAsString()
      .getAsInteger(10, PrefixNops);

  // KCFI allows indirect calls to any location that's preceded by a valid
  // type identifier. To avoid encoding the full constant into an instruction,
  // and thus emitting potential call target gadgets at each indirect call
  // site, load a negated constant to a register and compare that to the
  // expected value at the call target.
  const Register AddrReg = MI.getOperand(0).getReg();
  const uint32_t Type = MI.getOperand(1).getImm();
  // The check is immediately before the call. If the call target is in R10,
  // we can clobber R11 for the check instead.
  unsigned TempReg = AddrReg == X86::R10 ? X86::R11D : X86::R10D;
  EmitAndCountInstruction(
      MCInstBuilder(X86::MOV32ri).addReg(TempReg).addImm(-MaskKCFIType(Type)));
  EmitAndCountInstruction(MCInstBuilder(X86::ADD32rm)
                              .addReg(X86::NoRegister)
                              .addReg(TempReg)
                              .addReg(AddrReg)
                              .addImm(1)
                              .addReg(X86::NoRegister)
                              .addImm(-(PrefixNops + 4))
                              .addReg(X86::NoRegister));

  MCSymbol *Pass = OutContext.createTempSymbol();
  EmitAndCountInstruction(
      MCInstBuilder(X86::JCC_1)
          .addExpr(MCSymbolRefExpr::create(Pass, OutContext))
          .addImm(X86::COND_E));

  MCSymbol *Trap = OutContext.createTempSymbol();
  OutStreamer->emitLabel(Trap);
  EmitAndCountInstruction(MCInstBuilder(X86::TRAP));
  emitKCFITrapEntry(MF, Trap);
  OutStreamer->emitLabel(Pass);
}

void X86AsmPrinter::LowerASAN_CHECK_MEMACCESS(const MachineInstr &MI) {
  // FIXME: Make this work on non-ELF.
  if (!TM.getTargetTriple().isOSBinFormatELF()) {
    report_fatal_error("llvm.asan.check.memaccess only supported on ELF");
    return;
  }

  const auto &Reg = MI.getOperand(0).getReg();
  ASanAccessInfo AccessInfo(MI.getOperand(1).getImm());

  uint64_t ShadowBase;
  int MappingScale;
  bool OrShadowOffset;
  getAddressSanitizerParams(TM.getTargetTriple(), 64, AccessInfo.CompileKernel,
                            &ShadowBase, &MappingScale, &OrShadowOffset);

  StringRef Name = AccessInfo.IsWrite ? "store" : "load";
  StringRef Op = OrShadowOffset ? "or" : "add";
  std::string SymName = ("__asan_check_" + Name + "_" + Op + "_" +
                         Twine(1ULL << AccessInfo.AccessSizeIndex) + "_" +
                         TM.getMCRegisterInfo()->getName(Reg.asMCReg()))
                            .str();
  if (OrShadowOffset)
    report_fatal_error(
        "OrShadowOffset is not supported with optimized callbacks");

  EmitAndCountInstruction(
      MCInstBuilder(X86::CALL64pcrel32)
          .addExpr(MCSymbolRefExpr::create(
              OutContext.getOrCreateSymbol(SymName), OutContext)));
}

void X86AsmPrinter::LowerPATCHABLE_OP(const MachineInstr &MI,
                                      X86MCInstLower &MCIL) {
  // PATCHABLE_OP minsize

  NoAutoPaddingScope NoPadScope(*OutStreamer);

  auto NextMI = std::find_if(std::next(MI.getIterator()),
                             MI.getParent()->end().getInstrIterator(),
                             [](auto &II) { return !II.isMetaInstruction(); });

  SmallString<256> Code;
  unsigned MinSize = MI.getOperand(0).getImm();

  if (NextMI != MI.getParent()->end() && !NextMI->isInlineAsm()) {
    // Lower the next MachineInstr to find its byte size.
    // If the next instruction is inline assembly, we skip lowering it for now,
    // and assume we should always generate NOPs.
    MCInst MCI;
    MCIL.Lower(&*NextMI, MCI);

    SmallVector<MCFixup, 4> Fixups;
    CodeEmitter->encodeInstruction(MCI, Code, Fixups, getSubtargetInfo());
  }

  if (Code.size() < MinSize) {
    if (MinSize == 2 && Subtarget->is32Bit() &&
        Subtarget->isTargetWindowsMSVC() &&
        (Subtarget->getCPU().empty() || Subtarget->getCPU() == "pentium3")) {
      // For compatibility reasons, when targetting MSVC, it is important to
      // generate a 'legacy' NOP in the form of a 8B FF MOV EDI, EDI. Some tools
      // rely specifically on this pattern to be able to patch a function.
      // This is only for 32-bit targets, when using /arch:IA32 or /arch:SSE.
      OutStreamer->emitInstruction(
          MCInstBuilder(X86::MOV32rr_REV).addReg(X86::EDI).addReg(X86::EDI),
          *Subtarget);
    } else {
      unsigned NopSize = emitNop(*OutStreamer, MinSize, Subtarget);
      assert(NopSize == MinSize && "Could not implement MinSize!");
      (void)NopSize;
    }
  }
}

// Lower a stackmap of the form:
// <id>, <shadowBytes>, ...
void X86AsmPrinter::LowerSTACKMAP(const MachineInstr &MI) {
  SMShadowTracker.emitShadowPadding(*OutStreamer, getSubtargetInfo());

  auto &Ctx = OutStreamer->getContext();
  MCSymbol *MILabel = Ctx.createTempSymbol();
  OutStreamer->emitLabel(MILabel);

  SM.recordStackMap(*MILabel, MI);
  unsigned NumShadowBytes = MI.getOperand(1).getImm();
  SMShadowTracker.reset(NumShadowBytes);
}

// Lower a patchpoint of the form:
// [<def>], <id>, <numBytes>, <target>, <numArgs>, <cc>, ...
void X86AsmPrinter::LowerPATCHPOINT(const MachineInstr &MI,
                                    X86MCInstLower &MCIL) {
  assert(Subtarget->is64Bit() && "Patchpoint currently only supports X86-64");

  SMShadowTracker.emitShadowPadding(*OutStreamer, getSubtargetInfo());

  NoAutoPaddingScope NoPadScope(*OutStreamer);

  auto &Ctx = OutStreamer->getContext();
  MCSymbol *MILabel = Ctx.createTempSymbol();
  OutStreamer->emitLabel(MILabel);
  SM.recordPatchPoint(*MILabel, MI);

  PatchPointOpers opers(&MI);
  unsigned ScratchIdx = opers.getNextScratchIdx();
  unsigned EncodedBytes = 0;
  const MachineOperand &CalleeMO = opers.getCallTarget();

  // Check for null target. If target is non-null (i.e. is non-zero or is
  // symbolic) then emit a call.
  if (!(CalleeMO.isImm() && !CalleeMO.getImm())) {
    MCOperand CalleeMCOp;
    switch (CalleeMO.getType()) {
    default:
      /// FIXME: Add a verifier check for bad callee types.
      llvm_unreachable("Unrecognized callee operand type.");
    case MachineOperand::MO_Immediate:
      if (CalleeMO.getImm())
        CalleeMCOp = MCOperand::createImm(CalleeMO.getImm());
      break;
    case MachineOperand::MO_ExternalSymbol:
    case MachineOperand::MO_GlobalAddress:
      CalleeMCOp = MCIL.LowerSymbolOperand(CalleeMO,
                                           MCIL.GetSymbolFromOperand(CalleeMO));
      break;
    }

    // Emit MOV to materialize the target address and the CALL to target.
    // This is encoded with 12-13 bytes, depending on which register is used.
    Register ScratchReg = MI.getOperand(ScratchIdx).getReg();
    if (X86II::isX86_64ExtendedReg(ScratchReg))
      EncodedBytes = 13;
    else
      EncodedBytes = 12;

    EmitAndCountInstruction(
        MCInstBuilder(X86::MOV64ri).addReg(ScratchReg).addOperand(CalleeMCOp));
    // FIXME: Add retpoline support and remove this.
    if (Subtarget->useIndirectThunkCalls())
      report_fatal_error(
          "Lowering patchpoint with thunks not yet implemented.");
    EmitAndCountInstruction(MCInstBuilder(X86::CALL64r).addReg(ScratchReg));
  }

  // Emit padding.
  unsigned NumBytes = opers.getNumPatchBytes();
  assert(NumBytes >= EncodedBytes &&
         "Patchpoint can't request size less than the length of a call.");

  emitX86Nops(*OutStreamer, NumBytes - EncodedBytes, Subtarget);
}

void X86AsmPrinter::LowerPATCHABLE_EVENT_CALL(const MachineInstr &MI,
                                              X86MCInstLower &MCIL) {
  assert(Subtarget->is64Bit() && "XRay custom events only supports X86-64");

  NoAutoPaddingScope NoPadScope(*OutStreamer);

  // We want to emit the following pattern, which follows the x86 calling
  // convention to prepare for the trampoline call to be patched in.
  //
  //   .p2align 1, ...
  // .Lxray_event_sled_N:
  //   jmp +N                        // jump across the instrumentation sled
  //   ...                           // set up arguments in register
  //   callq __xray_CustomEvent@plt  // force dependency to symbol
  //   ...
  //   <jump here>
  //
  // After patching, it would look something like:
  //
  //   nopw (2-byte nop)
  //   ...
  //   callq __xrayCustomEvent  // already lowered
  //   ...
  //
  // ---
  // First we emit the label and the jump.
  auto CurSled = OutContext.createTempSymbol("xray_event_sled_", true);
  OutStreamer->AddComment("# XRay Custom Event Log");
  OutStreamer->emitCodeAlignment(Align(2), &getSubtargetInfo());
  OutStreamer->emitLabel(CurSled);

  // Use a two-byte `jmp`. This version of JMP takes an 8-bit relative offset as
  // an operand (computed as an offset from the jmp instruction).
  // FIXME: Find another less hacky way do force the relative jump.
  OutStreamer->emitBinaryData("\xeb\x0f");

  // The default C calling convention will place two arguments into %rcx and
  // %rdx -- so we only work with those.
  const Register DestRegs[] = {X86::RDI, X86::RSI};
  bool UsedMask[] = {false, false};
  // Filled out in loop.
  Register SrcRegs[] = {0, 0};

  // Then we put the operands in the %rdi and %rsi registers. We spill the
  // values in the register before we clobber them, and mark them as used in
  // UsedMask. In case the arguments are already in the correct register, we use
  // emit nops appropriately sized to keep the sled the same size in every
  // situation.
  for (unsigned I = 0; I < MI.getNumOperands(); ++I)
    if (auto Op = MCIL.LowerMachineOperand(&MI, MI.getOperand(I));
        Op.isValid()) {
      assert(Op.isReg() && "Only support arguments in registers");
      SrcRegs[I] = getX86SubSuperRegister(Op.getReg(), 64);
      assert(SrcRegs[I].isValid() && "Invalid operand");
      if (SrcRegs[I] != DestRegs[I]) {
        UsedMask[I] = true;
        EmitAndCountInstruction(
            MCInstBuilder(X86::PUSH64r).addReg(DestRegs[I]));
      } else {
        emitX86Nops(*OutStreamer, 4, Subtarget);
      }
    }

  // Now that the register values are stashed, mov arguments into place.
  // FIXME: This doesn't work if one of the later SrcRegs is equal to an
  // earlier DestReg. We will have already overwritten over the register before
  // we can copy from it.
  for (unsigned I = 0; I < MI.getNumOperands(); ++I)
    if (SrcRegs[I] != DestRegs[I])
      EmitAndCountInstruction(
          MCInstBuilder(X86::MOV64rr).addReg(DestRegs[I]).addReg(SrcRegs[I]));

  // We emit a hard dependency on the __xray_CustomEvent symbol, which is the
  // name of the trampoline to be implemented by the XRay runtime.
  auto TSym = OutContext.getOrCreateSymbol("__xray_CustomEvent");
  MachineOperand TOp = MachineOperand::CreateMCSymbol(TSym);
  if (isPositionIndependent())
    TOp.setTargetFlags(X86II::MO_PLT);

  // Emit the call instruction.
  EmitAndCountInstruction(MCInstBuilder(X86::CALL64pcrel32)
                              .addOperand(MCIL.LowerSymbolOperand(TOp, TSym)));

  // Restore caller-saved and used registers.
  for (unsigned I = sizeof UsedMask; I-- > 0;)
    if (UsedMask[I])
      EmitAndCountInstruction(MCInstBuilder(X86::POP64r).addReg(DestRegs[I]));
    else
      emitX86Nops(*OutStreamer, 1, Subtarget);

  OutStreamer->AddComment("xray custom event end.");

  // Record the sled version. Version 0 of this sled was spelled differently, so
  // we let the runtime handle the different offsets we're using. Version 2
  // changed the absolute address to a PC-relative address.
  recordSled(CurSled, MI, SledKind::CUSTOM_EVENT, 2);
}

void X86AsmPrinter::LowerPATCHABLE_TYPED_EVENT_CALL(const MachineInstr &MI,
                                                    X86MCInstLower &MCIL) {
  assert(Subtarget->is64Bit() && "XRay typed events only supports X86-64");

  NoAutoPaddingScope NoPadScope(*OutStreamer);

  // We want to emit the following pattern, which follows the x86 calling
  // convention to prepare for the trampoline call to be patched in.
  //
  //   .p2align 1, ...
  // .Lxray_event_sled_N:
  //   jmp +N                        // jump across the instrumentation sled
  //   ...                           // set up arguments in register
  //   callq __xray_TypedEvent@plt  // force dependency to symbol
  //   ...
  //   <jump here>
  //
  // After patching, it would look something like:
  //
  //   nopw (2-byte nop)
  //   ...
  //   callq __xrayTypedEvent  // already lowered
  //   ...
  //
  // ---
  // First we emit the label and the jump.
  auto CurSled = OutContext.createTempSymbol("xray_typed_event_sled_", true);
  OutStreamer->AddComment("# XRay Typed Event Log");
  OutStreamer->emitCodeAlignment(Align(2), &getSubtargetInfo());
  OutStreamer->emitLabel(CurSled);

  // Use a two-byte `jmp`. This version of JMP takes an 8-bit relative offset as
  // an operand (computed as an offset from the jmp instruction).
  // FIXME: Find another less hacky way do force the relative jump.
  OutStreamer->emitBinaryData("\xeb\x14");

  // An x86-64 convention may place three arguments into %rcx, %rdx, and R8,
  // so we'll work with those. Or we may be called via SystemV, in which case
  // we don't have to do any translation.
  const Register DestRegs[] = {X86::RDI, X86::RSI, X86::RDX};
  bool UsedMask[] = {false, false, false};

  // Will fill out src regs in the loop.
  Register SrcRegs[] = {0, 0, 0};

  // Then we put the operands in the SystemV registers. We spill the values in
  // the registers before we clobber them, and mark them as used in UsedMask.
  // In case the arguments are already in the correct register, we emit nops
  // appropriately sized to keep the sled the same size in every situation.
  for (unsigned I = 0; I < MI.getNumOperands(); ++I)
    if (auto Op = MCIL.LowerMachineOperand(&MI, MI.getOperand(I));
        Op.isValid()) {
      // TODO: Is register only support adequate?
      assert(Op.isReg() && "Only supports arguments in registers");
      SrcRegs[I] = getX86SubSuperRegister(Op.getReg(), 64);
      assert(SrcRegs[I].isValid() && "Invalid operand");
      if (SrcRegs[I] != DestRegs[I]) {
        UsedMask[I] = true;
        EmitAndCountInstruction(
            MCInstBuilder(X86::PUSH64r).addReg(DestRegs[I]));
      } else {
        emitX86Nops(*OutStreamer, 4, Subtarget);
      }
    }

  // In the above loop we only stash all of the destination registers or emit
  // nops if the arguments are already in the right place. Doing the actually
  // moving is postponed until after all the registers are stashed so nothing
  // is clobbers. We've already added nops to account for the size of mov and
  // push if the register is in the right place, so we only have to worry about
  // emitting movs.
  // FIXME: This doesn't work if one of the later SrcRegs is equal to an
  // earlier DestReg. We will have already overwritten over the register before
  // we can copy from it.
  for (unsigned I = 0; I < MI.getNumOperands(); ++I)
    if (UsedMask[I])
      EmitAndCountInstruction(
          MCInstBuilder(X86::MOV64rr).addReg(DestRegs[I]).addReg(SrcRegs[I]));

  // We emit a hard dependency on the __xray_TypedEvent symbol, which is the
  // name of the trampoline to be implemented by the XRay runtime.
  auto TSym = OutContext.getOrCreateSymbol("__xray_TypedEvent");
  MachineOperand TOp = MachineOperand::CreateMCSymbol(TSym);
  if (isPositionIndependent())
    TOp.setTargetFlags(X86II::MO_PLT);

  // Emit the call instruction.
  EmitAndCountInstruction(MCInstBuilder(X86::CALL64pcrel32)
                              .addOperand(MCIL.LowerSymbolOperand(TOp, TSym)));

  // Restore caller-saved and used registers.
  for (unsigned I = sizeof UsedMask; I-- > 0;)
    if (UsedMask[I])
      EmitAndCountInstruction(MCInstBuilder(X86::POP64r).addReg(DestRegs[I]));
    else
      emitX86Nops(*OutStreamer, 1, Subtarget);

  OutStreamer->AddComment("xray typed event end.");

  // Record the sled version.
  recordSled(CurSled, MI, SledKind::TYPED_EVENT, 2);
}

void X86AsmPrinter::LowerPATCHABLE_FUNCTION_ENTER(const MachineInstr &MI,
                                                  X86MCInstLower &MCIL) {

  NoAutoPaddingScope NoPadScope(*OutStreamer);

  const Function &F = MF->getFunction();
  if (F.hasFnAttribute("patchable-function-entry")) {
    unsigned Num;
    if (F.getFnAttribute("patchable-function-entry")
            .getValueAsString()
            .getAsInteger(10, Num))
      return;
    emitX86Nops(*OutStreamer, Num, Subtarget);
    return;
  }
  // We want to emit the following pattern:
  //
  //   .p2align 1, ...
  // .Lxray_sled_N:
  //   jmp .tmpN
  //   # 9 bytes worth of noops
  //
  // We need the 9 bytes because at runtime, we'd be patching over the full 11
  // bytes with the following pattern:
  //
  //   mov %r10, <function id, 32-bit>   // 6 bytes
  //   call <relative offset, 32-bits>   // 5 bytes
  //
  auto CurSled = OutContext.createTempSymbol("xray_sled_", true);
  OutStreamer->emitCodeAlignment(Align(2), &getSubtargetInfo());
  OutStreamer->emitLabel(CurSled);

  // Use a two-byte `jmp`. This version of JMP takes an 8-bit relative offset as
  // an operand (computed as an offset from the jmp instruction).
  // FIXME: Find another less hacky way do force the relative jump.
  OutStreamer->emitBytes("\xeb\x09");
  emitX86Nops(*OutStreamer, 9, Subtarget);
  recordSled(CurSled, MI, SledKind::FUNCTION_ENTER, 2);
}

void X86AsmPrinter::LowerPATCHABLE_RET(const MachineInstr &MI,
                                       X86MCInstLower &MCIL) {
  NoAutoPaddingScope NoPadScope(*OutStreamer);

  // Since PATCHABLE_RET takes the opcode of the return statement as an
  // argument, we use that to emit the correct form of the RET that we want.
  // i.e. when we see this:
  //
  //   PATCHABLE_RET X86::RET ...
  //
  // We should emit the RET followed by sleds.
  //
  //   .p2align 1, ...
  // .Lxray_sled_N:
  //   ret  # or equivalent instruction
  //   # 10 bytes worth of noops
  //
  // This just makes sure that the alignment for the next instruction is 2.
  auto CurSled = OutContext.createTempSymbol("xray_sled_", true);
  OutStreamer->emitCodeAlignment(Align(2), &getSubtargetInfo());
  OutStreamer->emitLabel(CurSled);
  unsigned OpCode = MI.getOperand(0).getImm();
  MCInst Ret;
  Ret.setOpcode(OpCode);
  for (auto &MO : drop_begin(MI.operands()))
    if (auto Op = MCIL.LowerMachineOperand(&MI, MO); Op.isValid())
      Ret.addOperand(Op);
  OutStreamer->emitInstruction(Ret, getSubtargetInfo());
  emitX86Nops(*OutStreamer, 10, Subtarget);
  recordSled(CurSled, MI, SledKind::FUNCTION_EXIT, 2);
}

void X86AsmPrinter::LowerPATCHABLE_TAIL_CALL(const MachineInstr &MI,
                                             X86MCInstLower &MCIL) {
  MCInst TC;
  TC.setOpcode(convertTailJumpOpcode(MI.getOperand(0).getImm()));
  // Drop the tail jump opcode.
  auto TCOperands = drop_begin(MI.operands());
  bool IsConditional = TC.getOpcode() == X86::JCC_1;
  MCSymbol *FallthroughLabel;
  if (IsConditional) {
    // Rewrite:
    //   je target
    //
    // To:
    //   jne .fallthrough
    //   .p2align 1, ...
    // .Lxray_sled_N:
    //   SLED_CODE
    //   jmp target
    // .fallthrough:
    FallthroughLabel = OutContext.createTempSymbol();
    EmitToStreamer(
        *OutStreamer,
        MCInstBuilder(X86::JCC_1)
            .addExpr(MCSymbolRefExpr::create(FallthroughLabel, OutContext))
            .addImm(X86::GetOppositeBranchCondition(
                static_cast<X86::CondCode>(MI.getOperand(2).getImm()))));
    TC.setOpcode(X86::JMP_1);
    // Drop the condition code.
    TCOperands = drop_end(TCOperands);
  }

  NoAutoPaddingScope NoPadScope(*OutStreamer);

  // Like PATCHABLE_RET, we have the actual instruction in the operands to this
  // instruction so we lower that particular instruction and its operands.
  // Unlike PATCHABLE_RET though, we put the sled before the JMP, much like how
  // we do it for PATCHABLE_FUNCTION_ENTER. The sled should be very similar to
  // the PATCHABLE_FUNCTION_ENTER case, followed by the lowering of the actual
  // tail call much like how we have it in PATCHABLE_RET.
  auto CurSled = OutContext.createTempSymbol("xray_sled_", true);
  OutStreamer->emitCodeAlignment(Align(2), &getSubtargetInfo());
  OutStreamer->emitLabel(CurSled);
  auto Target = OutContext.createTempSymbol();

  // Use a two-byte `jmp`. This version of JMP takes an 8-bit relative offset as
  // an operand (computed as an offset from the jmp instruction).
  // FIXME: Find another less hacky way do force the relative jump.
  OutStreamer->emitBytes("\xeb\x09");
  emitX86Nops(*OutStreamer, 9, Subtarget);
  OutStreamer->emitLabel(Target);
  recordSled(CurSled, MI, SledKind::TAIL_CALL, 2);

  // Before emitting the instruction, add a comment to indicate that this is
  // indeed a tail call.
  OutStreamer->AddComment("TAILCALL");
  for (auto &MO : TCOperands)
    if (auto Op = MCIL.LowerMachineOperand(&MI, MO); Op.isValid())
      TC.addOperand(Op);
  OutStreamer->emitInstruction(TC, getSubtargetInfo());

  if (IsConditional)
    OutStreamer->emitLabel(FallthroughLabel);
}

static unsigned getSrcIdx(const MachineInstr* MI, unsigned SrcIdx) {
  if (X86II::isKMasked(MI->getDesc().TSFlags)) {
    // Skip mask operand.
    ++SrcIdx;
    if (X86II::isKMergeMasked(MI->getDesc().TSFlags)) {
      // Skip passthru operand.
      ++SrcIdx;
    }
  }
  return SrcIdx;
}

static void printDstRegisterName(raw_ostream &CS, const MachineInstr *MI,
                                 unsigned SrcOpIdx) {
  const MachineOperand &DstOp = MI->getOperand(0);
  CS << X86ATTInstPrinter::getRegisterName(DstOp.getReg());

  // Handle AVX512 MASK/MASXZ write mask comments.
  // MASK: zmmX {%kY}
  // MASKZ: zmmX {%kY} {z}
  if (X86II::isKMasked(MI->getDesc().TSFlags)) {
    const MachineOperand &WriteMaskOp = MI->getOperand(SrcOpIdx - 1);
    StringRef Mask = X86ATTInstPrinter::getRegisterName(WriteMaskOp.getReg());
    CS << " {%" << Mask << "}";
    if (!X86II::isKMergeMasked(MI->getDesc().TSFlags)) {
      CS << " {z}";
    }
  }
}

static void printShuffleMask(raw_ostream &CS, StringRef Src1Name,
                             StringRef Src2Name, ArrayRef<int> Mask) {
  // One source operand, fix the mask to print all elements in one span.
  SmallVector<int, 8> ShuffleMask(Mask);
  if (Src1Name == Src2Name)
    for (int i = 0, e = ShuffleMask.size(); i != e; ++i)
      if (ShuffleMask[i] >= e)
        ShuffleMask[i] -= e;

  for (int i = 0, e = ShuffleMask.size(); i != e; ++i) {
    if (i != 0)
      CS << ",";
    if (ShuffleMask[i] == SM_SentinelZero) {
      CS << "zero";
      continue;
    }

    // Otherwise, it must come from src1 or src2.  Print the span of elements
    // that comes from this src.
    bool isSrc1 = ShuffleMask[i] < (int)e;
    CS << (isSrc1 ? Src1Name : Src2Name) << '[';

    bool IsFirst = true;
    while (i != e && ShuffleMask[i] != SM_SentinelZero &&
           (ShuffleMask[i] < (int)e) == isSrc1) {
      if (!IsFirst)
        CS << ',';
      else
        IsFirst = false;
      if (ShuffleMask[i] == SM_SentinelUndef)
        CS << "u";
      else
        CS << ShuffleMask[i] % (int)e;
      ++i;
    }
    CS << ']';
    --i; // For loop increments element #.
  }
}

static std::string getShuffleComment(const MachineInstr *MI, unsigned SrcOp1Idx,
                                     unsigned SrcOp2Idx, ArrayRef<int> Mask) {
  std::string Comment;

  const MachineOperand &SrcOp1 = MI->getOperand(SrcOp1Idx);
  const MachineOperand &SrcOp2 = MI->getOperand(SrcOp2Idx);
  StringRef Src1Name = SrcOp1.isReg()
                           ? X86ATTInstPrinter::getRegisterName(SrcOp1.getReg())
                           : "mem";
  StringRef Src2Name = SrcOp2.isReg()
                           ? X86ATTInstPrinter::getRegisterName(SrcOp2.getReg())
                           : "mem";

  raw_string_ostream CS(Comment);
  printDstRegisterName(CS, MI, SrcOp1Idx);
  CS << " = ";
  printShuffleMask(CS, Src1Name, Src2Name, Mask);

  return Comment;
}

static void printConstant(const APInt &Val, raw_ostream &CS,
                          bool PrintZero = false) {
  if (Val.getBitWidth() <= 64) {
    CS << (PrintZero ? 0ULL : Val.getZExtValue());
  } else {
    // print multi-word constant as (w0,w1)
    CS << "(";
    for (int i = 0, N = Val.getNumWords(); i < N; ++i) {
      if (i > 0)
        CS << ",";
      CS << (PrintZero ? 0ULL : Val.getRawData()[i]);
    }
    CS << ")";
  }
}

static void printConstant(const APFloat &Flt, raw_ostream &CS,
                          bool PrintZero = false) {
  SmallString<32> Str;
  // Force scientific notation to distinguish from integers.
  if (PrintZero)
    APFloat::getZero(Flt.getSemantics()).toString(Str, 0, 0);
  else
    Flt.toString(Str, 0, 0);
  CS << Str;
}

static void printConstant(const Constant *COp, unsigned BitWidth,
                          raw_ostream &CS, bool PrintZero = false) {
  if (isa<UndefValue>(COp)) {
    CS << "u";
  } else if (auto *CI = dyn_cast<ConstantInt>(COp)) {
    if (auto VTy = dyn_cast<FixedVectorType>(CI->getType())) {
      for (unsigned I = 0, E = VTy->getNumElements(); I != E; ++I) {
        if (I != 0)
          CS << ',';
        printConstant(CI->getValue(), CS, PrintZero);
      }
    } else
      printConstant(CI->getValue(), CS, PrintZero);
  } else if (auto *CF = dyn_cast<ConstantFP>(COp)) {
    if (auto VTy = dyn_cast<FixedVectorType>(CF->getType())) {
      for (unsigned I = 0, E = VTy->getNumElements(); I != E; ++I) {
        if (I != 0)
          CS << ',';
        printConstant(CF->getValueAPF(), CS, PrintZero);
      }
    } else
      printConstant(CF->getValueAPF(), CS, PrintZero);
  } else if (auto *CDS = dyn_cast<ConstantDataSequential>(COp)) {
    Type *EltTy = CDS->getElementType();
    bool IsInteger = EltTy->isIntegerTy();
    bool IsFP = EltTy->isHalfTy() || EltTy->isFloatTy() || EltTy->isDoubleTy();
    unsigned EltBits = EltTy->getPrimitiveSizeInBits();
    unsigned E = std::min(BitWidth / EltBits, (unsigned)CDS->getNumElements());
    if ((BitWidth % EltBits) == 0) {
      for (unsigned I = 0; I != E; ++I) {
        if (I != 0)
          CS << ",";
        if (IsInteger)
          printConstant(CDS->getElementAsAPInt(I), CS, PrintZero);
        else if (IsFP)
          printConstant(CDS->getElementAsAPFloat(I), CS, PrintZero);
        else
          CS << "?";
      }
    } else {
      CS << "?";
    }
  } else if (auto *CV = dyn_cast<ConstantVector>(COp)) {
    unsigned EltBits = CV->getType()->getScalarSizeInBits();
    unsigned E = std::min(BitWidth / EltBits, CV->getNumOperands());
    if ((BitWidth % EltBits) == 0) {
      for (unsigned I = 0; I != E; ++I) {
        if (I != 0)
          CS << ",";
        printConstant(CV->getOperand(I), EltBits, CS, PrintZero);
      }
    } else {
      CS << "?";
    }
  } else {
    CS << "?";
  }
}

static void printZeroUpperMove(const MachineInstr *MI, MCStreamer &OutStreamer,
                               int SclWidth, int VecWidth,
                               const char *ShuffleComment) {
  unsigned SrcIdx = getSrcIdx(MI, 1);

  std::string Comment;
  raw_string_ostream CS(Comment);
  printDstRegisterName(CS, MI, SrcIdx);
  CS << " = ";

  if (auto *C = X86::getConstantFromPool(*MI, SrcIdx)) {
    CS << "[";
    printConstant(C, SclWidth, CS);
    for (int I = 1, E = VecWidth / SclWidth; I < E; ++I) {
      CS << ",";
      printConstant(C, SclWidth, CS, true);
    }
    CS << "]";
    OutStreamer.AddComment(CS.str());
    return; // early-out
  }

  // We didn't find a constant load, fallback to a shuffle mask decode.
  CS << ShuffleComment;
  OutStreamer.AddComment(CS.str());
}

static void printBroadcast(const MachineInstr *MI, MCStreamer &OutStreamer,
                           int Repeats, int BitWidth) {
  unsigned SrcIdx = getSrcIdx(MI, 1);
  if (auto *C = X86::getConstantFromPool(*MI, SrcIdx)) {
    std::string Comment;
    raw_string_ostream CS(Comment);
    printDstRegisterName(CS, MI, SrcIdx);
    CS << " = [";
    for (int l = 0; l != Repeats; ++l) {
      if (l != 0)
        CS << ",";
      printConstant(C, BitWidth, CS);
    }
    CS << "]";
    OutStreamer.AddComment(CS.str());
  }
}

static bool printExtend(const MachineInstr *MI, MCStreamer &OutStreamer,
                        int SrcEltBits, int DstEltBits, bool IsSext) {
  unsigned SrcIdx = getSrcIdx(MI, 1);
  auto *C = X86::getConstantFromPool(*MI, SrcIdx);
  if (C && C->getType()->getScalarSizeInBits() == unsigned(SrcEltBits)) {
    if (auto *CDS = dyn_cast<ConstantDataSequential>(C)) {
      int NumElts = CDS->getNumElements();
      std::string Comment;
      raw_string_ostream CS(Comment);
      printDstRegisterName(CS, MI, SrcIdx);
      CS << " = [";
      for (int i = 0; i != NumElts; ++i) {
        if (i != 0)
          CS << ",";
        if (CDS->getElementType()->isIntegerTy()) {
          APInt Elt = CDS->getElementAsAPInt(i);
          Elt = IsSext ? Elt.sext(DstEltBits) : Elt.zext(DstEltBits);
          printConstant(Elt, CS);
        } else
          CS << "?";
      }
      CS << "]";
      OutStreamer.AddComment(CS.str());
      return true;
    }
  }

  return false;
}
static void printSignExtend(const MachineInstr *MI, MCStreamer &OutStreamer,
                            int SrcEltBits, int DstEltBits) {
  printExtend(MI, OutStreamer, SrcEltBits, DstEltBits, true);
}
static void printZeroExtend(const MachineInstr *MI, MCStreamer &OutStreamer,
                            int SrcEltBits, int DstEltBits) {
  if (printExtend(MI, OutStreamer, SrcEltBits, DstEltBits, false))
    return;

  // We didn't find a constant load, fallback to a shuffle mask decode.
  std::string Comment;
  raw_string_ostream CS(Comment);
  printDstRegisterName(CS, MI, getSrcIdx(MI, 1));
  CS << " = ";

  SmallVector<int> Mask;
  unsigned Width = X86::getVectorRegisterWidth(MI->getDesc().operands()[0]);
  assert((Width % DstEltBits) == 0 && (DstEltBits % SrcEltBits) == 0 &&
         "Illegal extension ratio");
  DecodeZeroExtendMask(SrcEltBits, DstEltBits, Width / DstEltBits, false, Mask);
  printShuffleMask(CS, "mem", "", Mask);

  OutStreamer.AddComment(CS.str());
}

void X86AsmPrinter::EmitSEHInstruction(const MachineInstr *MI) {
  assert(MF->hasWinCFI() && "SEH_ instruction in function without WinCFI?");
  assert((getSubtarget().isOSWindows() || getSubtarget().isUEFI()) &&
         "SEH_ instruction Windows and UEFI only");

  // Use the .cv_fpo directives if we're emitting CodeView on 32-bit x86.
  if (EmitFPOData) {
    X86TargetStreamer *XTS =
        static_cast<X86TargetStreamer *>(OutStreamer->getTargetStreamer());
    switch (MI->getOpcode()) {
    case X86::SEH_PushReg:
      XTS->emitFPOPushReg(MI->getOperand(0).getImm());
      break;
    case X86::SEH_StackAlloc:
      XTS->emitFPOStackAlloc(MI->getOperand(0).getImm());
      break;
    case X86::SEH_StackAlign:
      XTS->emitFPOStackAlign(MI->getOperand(0).getImm());
      break;
    case X86::SEH_SetFrame:
      assert(MI->getOperand(1).getImm() == 0 &&
             ".cv_fpo_setframe takes no offset");
      XTS->emitFPOSetFrame(MI->getOperand(0).getImm());
      break;
    case X86::SEH_EndPrologue:
      XTS->emitFPOEndPrologue();
      break;
    case X86::SEH_SaveReg:
    case X86::SEH_SaveXMM:
    case X86::SEH_PushFrame:
      llvm_unreachable("SEH_ directive incompatible with FPO");
      break;
    default:
      llvm_unreachable("expected SEH_ instruction");
    }
    return;
  }

  // Otherwise, use the .seh_ directives for all other Windows platforms.
  switch (MI->getOpcode()) {
  case X86::SEH_PushReg:
    OutStreamer->emitWinCFIPushReg(MI->getOperand(0).getImm());
    break;

  case X86::SEH_SaveReg:
    OutStreamer->emitWinCFISaveReg(MI->getOperand(0).getImm(),
                                   MI->getOperand(1).getImm());
    break;

  case X86::SEH_SaveXMM:
    OutStreamer->emitWinCFISaveXMM(MI->getOperand(0).getImm(),
                                   MI->getOperand(1).getImm());
    break;

  case X86::SEH_StackAlloc:
    OutStreamer->emitWinCFIAllocStack(MI->getOperand(0).getImm());
    break;

  case X86::SEH_SetFrame:
    OutStreamer->emitWinCFISetFrame(MI->getOperand(0).getImm(),
                                    MI->getOperand(1).getImm());
    break;

  case X86::SEH_PushFrame:
    OutStreamer->emitWinCFIPushFrame(MI->getOperand(0).getImm());
    break;

  case X86::SEH_EndPrologue:
    OutStreamer->emitWinCFIEndProlog();
    break;

  case X86::SEH_BeginEpilogue:
    OutStreamer->emitWinCFIBeginEpilogue();
    break;

  case X86::SEH_EndEpilogue:
    OutStreamer->emitWinCFIEndEpilogue();
    break;

  case X86::SEH_UnwindV2Start:
    OutStreamer->emitWinCFIUnwindV2Start();
    break;

  case X86::SEH_UnwindVersion:
    OutStreamer->emitWinCFIUnwindVersion(MI->getOperand(0).getImm());
    break;

  default:
    llvm_unreachable("expected SEH_ instruction");
  }
}

static void addConstantComments(const MachineInstr *MI,
                                MCStreamer &OutStreamer) {
  switch (MI->getOpcode()) {
  // Lower PSHUFB and VPERMILP normally but add a comment if we can find
  // a constant shuffle mask. We won't be able to do this at the MC layer
  // because the mask isn't an immediate.
  case X86::PSHUFBrm:
  case X86::VPSHUFBrm:
  case X86::VPSHUFBYrm:
  case X86::VPSHUFBZ128rm:
  case X86::VPSHUFBZ128rmk:
  case X86::VPSHUFBZ128rmkz:
  case X86::VPSHUFBZ256rm:
  case X86::VPSHUFBZ256rmk:
  case X86::VPSHUFBZ256rmkz:
  case X86::VPSHUFBZrm:
  case X86::VPSHUFBZrmk:
  case X86::VPSHUFBZrmkz: {
    unsigned SrcIdx = getSrcIdx(MI, 1);
    if (auto *C = X86::getConstantFromPool(*MI, SrcIdx + 1)) {
      unsigned Width = X86::getVectorRegisterWidth(MI->getDesc().operands()[0]);
      SmallVector<int, 64> Mask;
      DecodePSHUFBMask(C, Width, Mask);
      if (!Mask.empty())
        OutStreamer.AddComment(getShuffleComment(MI, SrcIdx, SrcIdx, Mask));
    }
    break;
  }

  case X86::VPERMILPSrm:
  case X86::VPERMILPSYrm:
  case X86::VPERMILPSZ128rm:
  case X86::VPERMILPSZ128rmk:
  case X86::VPERMILPSZ128rmkz:
  case X86::VPERMILPSZ256rm:
  case X86::VPERMILPSZ256rmk:
  case X86::VPERMILPSZ256rmkz:
  case X86::VPERMILPSZrm:
  case X86::VPERMILPSZrmk:
  case X86::VPERMILPSZrmkz: {
    unsigned SrcIdx = getSrcIdx(MI, 1);
    if (auto *C = X86::getConstantFromPool(*MI, SrcIdx + 1)) {
      unsigned Width = X86::getVectorRegisterWidth(MI->getDesc().operands()[0]);
      SmallVector<int, 16> Mask;
      DecodeVPERMILPMask(C, 32, Width, Mask);
      if (!Mask.empty())
        OutStreamer.AddComment(getShuffleComment(MI, SrcIdx, SrcIdx, Mask));
    }
    break;
  }
  case X86::VPERMILPDrm:
  case X86::VPERMILPDYrm:
  case X86::VPERMILPDZ128rm:
  case X86::VPERMILPDZ128rmk:
  case X86::VPERMILPDZ128rmkz:
  case X86::VPERMILPDZ256rm:
  case X86::VPERMILPDZ256rmk:
  case X86::VPERMILPDZ256rmkz:
  case X86::VPERMILPDZrm:
  case X86::VPERMILPDZrmk:
  case X86::VPERMILPDZrmkz: {
    unsigned SrcIdx = getSrcIdx(MI, 1);
    if (auto *C = X86::getConstantFromPool(*MI, SrcIdx + 1)) {
      unsigned Width = X86::getVectorRegisterWidth(MI->getDesc().operands()[0]);
      SmallVector<int, 16> Mask;
      DecodeVPERMILPMask(C, 64, Width, Mask);
      if (!Mask.empty())
        OutStreamer.AddComment(getShuffleComment(MI, SrcIdx, SrcIdx, Mask));
    }
    break;
  }

  case X86::VPERMIL2PDrm:
  case X86::VPERMIL2PSrm:
  case X86::VPERMIL2PDYrm:
  case X86::VPERMIL2PSYrm: {
    assert(MI->getNumOperands() >= (3 + X86::AddrNumOperands + 1) &&
           "Unexpected number of operands!");

    const MachineOperand &CtrlOp = MI->getOperand(MI->getNumOperands() - 1);
    if (!CtrlOp.isImm())
      break;

    unsigned ElSize;
    switch (MI->getOpcode()) {
    default: llvm_unreachable("Invalid opcode");
    case X86::VPERMIL2PSrm: case X86::VPERMIL2PSYrm: ElSize = 32; break;
    case X86::VPERMIL2PDrm: case X86::VPERMIL2PDYrm: ElSize = 64; break;
    }

    if (auto *C = X86::getConstantFromPool(*MI, 3)) {
      unsigned Width = X86::getVectorRegisterWidth(MI->getDesc().operands()[0]);
      SmallVector<int, 16> Mask;
      DecodeVPERMIL2PMask(C, (unsigned)CtrlOp.getImm(), ElSize, Width, Mask);
      if (!Mask.empty())
        OutStreamer.AddComment(getShuffleComment(MI, 1, 2, Mask));
    }
    break;
  }

  case X86::VPPERMrrm: {
    if (auto *C = X86::getConstantFromPool(*MI, 3)) {
      unsigned Width = X86::getVectorRegisterWidth(MI->getDesc().operands()[0]);
      SmallVector<int, 16> Mask;
      DecodeVPPERMMask(C, Width, Mask);
      if (!Mask.empty())
        OutStreamer.AddComment(getShuffleComment(MI, 1, 2, Mask));
    }
    break;
  }

  case X86::MMX_MOVQ64rm: {
    if (auto *C = X86::getConstantFromPool(*MI, 1)) {
      std::string Comment;
      raw_string_ostream CS(Comment);
      const MachineOperand &DstOp = MI->getOperand(0);
      CS << X86ATTInstPrinter::getRegisterName(DstOp.getReg()) << " = ";
      if (auto *CF = dyn_cast<ConstantFP>(C)) {
        CS << "0x" << toString(CF->getValueAPF().bitcastToAPInt(), 16, false);
        OutStreamer.AddComment(CS.str());
      }
    }
    break;
  }

#define INSTR_CASE(Prefix, Instr, Suffix, Postfix)                             \
  case X86::Prefix##Instr##Suffix##rm##Postfix:

#define CASE_ARITH_RM(Instr)                                                   \
  INSTR_CASE(, Instr, , )   /* SSE */                                          \
  INSTR_CASE(V, Instr, , )  /* AVX-128 */                                      \
  INSTR_CASE(V, Instr, Y, ) /* AVX-256 */                                      \
  INSTR_CASE(V, Instr, Z128, )                                                 \
  INSTR_CASE(V, Instr, Z128, k)                                                \
  INSTR_CASE(V, Instr, Z128, kz)                                               \
  INSTR_CASE(V, Instr, Z256, )                                                 \
  INSTR_CASE(V, Instr, Z256, k)                                                \
  INSTR_CASE(V, Instr, Z256, kz)                                               \
  INSTR_CASE(V, Instr, Z, )                                                    \
  INSTR_CASE(V, Instr, Z, k)                                                   \
  INSTR_CASE(V, Instr, Z, kz)

    // TODO: Add additional instructions when useful.
    CASE_ARITH_RM(PMADDUBSW) {
      unsigned SrcIdx = getSrcIdx(MI, 1);
      if (auto *C = X86::getConstantFromPool(*MI, SrcIdx + 1)) {
        std::string Comment;
        raw_string_ostream CS(Comment);
        unsigned VectorWidth =
            X86::getVectorRegisterWidth(MI->getDesc().operands()[0]);
        CS << "[";
        printConstant(C, VectorWidth, CS);
        CS << "]";
        OutStreamer.AddComment(CS.str());
      }
      break;
    }

    CASE_ARITH_RM(PMADDWD)
    CASE_ARITH_RM(PMULLD)
    CASE_ARITH_RM(PMULLW)
    CASE_ARITH_RM(PMULHW)
    CASE_ARITH_RM(PMULHUW)
    CASE_ARITH_RM(PMULHRSW) {
      unsigned SrcIdx = getSrcIdx(MI, 1);
      if (auto *C = X86::getConstantFromPool(*MI, SrcIdx + 1)) {
        std::string Comment;
        raw_string_ostream CS(Comment);
        unsigned VectorWidth =
            X86::getVectorRegisterWidth(MI->getDesc().operands()[0]);
        CS << "[";
        printConstant(C, VectorWidth, CS);
        CS << "]";
        OutStreamer.AddComment(CS.str());
      }
      break;
    }

#define MASK_AVX512_CASE(Instr)                                                \
  case Instr:                                                                  \
  case Instr##k:                                                               \
  case Instr##kz:

  case X86::MOVSDrm:
  case X86::VMOVSDrm:
  MASK_AVX512_CASE(X86::VMOVSDZrm)
  case X86::MOVSDrm_alt:
  case X86::VMOVSDrm_alt:
  case X86::VMOVSDZrm_alt:
  case X86::MOVQI2PQIrm:
  case X86::VMOVQI2PQIrm:
  case X86::VMOVQI2PQIZrm:
    printZeroUpperMove(MI, OutStreamer, 64, 128, "mem[0],zero");
    break;

  MASK_AVX512_CASE(X86::VMOVSHZrm)
  case X86::VMOVSHZrm_alt:
    printZeroUpperMove(MI, OutStreamer, 16, 128,
                       "mem[0],zero,zero,zero,zero,zero,zero,zero");
    break;

  case X86::MOVSSrm:
  case X86::VMOVSSrm:
  MASK_AVX512_CASE(X86::VMOVSSZrm)
  case X86::MOVSSrm_alt:
  case X86::VMOVSSrm_alt:
  case X86::VMOVSSZrm_alt:
  case X86::MOVDI2PDIrm:
  case X86::VMOVDI2PDIrm:
  case X86::VMOVDI2PDIZrm:
    printZeroUpperMove(MI, OutStreamer, 32, 128, "mem[0],zero,zero,zero");
    break;

#define MOV_CASE(Prefix, Suffix)                                               \
  case X86::Prefix##MOVAPD##Suffix##rm:                                        \
  case X86::Prefix##MOVAPS##Suffix##rm:                                        \
  case X86::Prefix##MOVUPD##Suffix##rm:                                        \
  case X86::Prefix##MOVUPS##Suffix##rm:                                        \
  case X86::Prefix##MOVDQA##Suffix##rm:                                        \
  case X86::Prefix##MOVDQU##Suffix##rm:

#define MOV_AVX512_CASE(Suffix, Postfix)                                       \
  case X86::VMOVDQA64##Suffix##rm##Postfix:                                    \
  case X86::VMOVDQA32##Suffix##rm##Postfix:                                    \
  case X86::VMOVDQU64##Suffix##rm##Postfix:                                    \
  case X86::VMOVDQU32##Suffix##rm##Postfix:                                    \
  case X86::VMOVDQU16##Suffix##rm##Postfix:                                    \
  case X86::VMOVDQU8##Suffix##rm##Postfix:                                     \
  case X86::VMOVAPS##Suffix##rm##Postfix:                                      \
  case X86::VMOVAPD##Suffix##rm##Postfix:                                      \
  case X86::VMOVUPS##Suffix##rm##Postfix:                                      \
  case X86::VMOVUPD##Suffix##rm##Postfix:

#define CASE_128_MOV_RM()                                                      \
  MOV_CASE(, )   /* SSE */                                                     \
  MOV_CASE(V, )  /* AVX-128 */                                                 \
  MOV_AVX512_CASE(Z128, )                                                      \
  MOV_AVX512_CASE(Z128, k)                                                     \
  MOV_AVX512_CASE(Z128, kz)

#define CASE_256_MOV_RM()                                                      \
  MOV_CASE(V, Y) /* AVX-256 */                                                 \
  MOV_AVX512_CASE(Z256, )                                                      \
  MOV_AVX512_CASE(Z256, k)                                                     \
  MOV_AVX512_CASE(Z256, kz)                                                    \

#define CASE_512_MOV_RM()                                                      \
  MOV_AVX512_CASE(Z, )                                                         \
  MOV_AVX512_CASE(Z, k)                                                        \
  MOV_AVX512_CASE(Z, kz)                                                       \

    // For loads from a constant pool to a vector register, print the constant
    // loaded.
    CASE_128_MOV_RM()
    printBroadcast(MI, OutStreamer, 1, 128);
    break;
    CASE_256_MOV_RM()
    printBroadcast(MI, OutStreamer, 1, 256);
    break;
    CASE_512_MOV_RM()
    printBroadcast(MI, OutStreamer, 1, 512);
    break;
  case X86::VBROADCASTF128rm:
  case X86::VBROADCASTI128rm:
  MASK_AVX512_CASE(X86::VBROADCASTF32X4Z256rm)
  MASK_AVX512_CASE(X86::VBROADCASTF64X2Z256rm)
  MASK_AVX512_CASE(X86::VBROADCASTI32X4Z256rm)
  MASK_AVX512_CASE(X86::VBROADCASTI64X2Z256rm)
    printBroadcast(MI, OutStreamer, 2, 128);
    break;
  MASK_AVX512_CASE(X86::VBROADCASTF32X4Zrm)
  MASK_AVX512_CASE(X86::VBROADCASTF64X2Zrm)
  MASK_AVX512_CASE(X86::VBROADCASTI32X4Zrm)
  MASK_AVX512_CASE(X86::VBROADCASTI64X2Zrm)
    printBroadcast(MI, OutStreamer, 4, 128);
    break;
  MASK_AVX512_CASE(X86::VBROADCASTF32X8Zrm)
  MASK_AVX512_CASE(X86::VBROADCASTF64X4Zrm)
  MASK_AVX512_CASE(X86::VBROADCASTI32X8Zrm)
  MASK_AVX512_CASE(X86::VBROADCASTI64X4Zrm)
    printBroadcast(MI, OutStreamer, 2, 256);
    break;

  // For broadcast loads from a constant pool to a vector register, repeatedly
  // print the constant loaded.
  case X86::MOVDDUPrm:
  case X86::VMOVDDUPrm:
  MASK_AVX512_CASE(X86::VMOVDDUPZ128rm)
  case X86::VPBROADCASTQrm:
  MASK_AVX512_CASE(X86::VPBROADCASTQZ128rm)
    printBroadcast(MI, OutStreamer, 2, 64);
    break;
  case X86::VBROADCASTSDYrm:
  MASK_AVX512_CASE(X86::VBROADCASTSDZ256rm)
  case X86::VPBROADCASTQYrm:
  MASK_AVX512_CASE(X86::VPBROADCASTQZ256rm)
    printBroadcast(MI, OutStreamer, 4, 64);
    break;
  MASK_AVX512_CASE(X86::VBROADCASTSDZrm)
  MASK_AVX512_CASE(X86::VPBROADCASTQZrm)
    printBroadcast(MI, OutStreamer, 8, 64);
    break;
  case X86::VBROADCASTSSrm:
  MASK_AVX512_CASE(X86::VBROADCASTSSZ128rm)
  case X86::VPBROADCASTDrm:
  MASK_AVX512_CASE(X86::VPBROADCASTDZ128rm)
    printBroadcast(MI, OutStreamer, 4, 32);
    break;
  case X86::VBROADCASTSSYrm:
    MASK_AVX512_CASE(X86::VBROADCASTSSZ256rm)
  case X86::VPBROADCASTDYrm:
  MASK_AVX512_CASE(X86::VPBROADCASTDZ256rm)
    printBroadcast(MI, OutStreamer, 8, 32);
    break;
  MASK_AVX512_CASE(X86::VBROADCASTSSZrm)
  MASK_AVX512_CASE(X86::VPBROADCASTDZrm)
    printBroadcast(MI, OutStreamer, 16, 32);
    break;
  case X86::VPBROADCASTWrm:
  MASK_AVX512_CASE(X86::VPBROADCASTWZ128rm)
    printBroadcast(MI, OutStreamer, 8, 16);
    break;
  case X86::VPBROADCASTWYrm:
  MASK_AVX512_CASE(X86::VPBROADCASTWZ256rm)
    printBroadcast(MI, OutStreamer, 16, 16);
    break;
  MASK_AVX512_CASE(X86::VPBROADCASTWZrm)
    printBroadcast(MI, OutStreamer, 32, 16);
    break;
  case X86::VPBROADCASTBrm:
  MASK_AVX512_CASE(X86::VPBROADCASTBZ128rm)
    printBroadcast(MI, OutStreamer, 16, 8);
    break;
  case X86::VPBROADCASTBYrm:
  MASK_AVX512_CASE(X86::VPBROADCASTBZ256rm)
    printBroadcast(MI, OutStreamer, 32, 8);
    break;
  MASK_AVX512_CASE(X86::VPBROADCASTBZrm)
    printBroadcast(MI, OutStreamer, 64, 8);
    break;

#define MOVX_CASE(Prefix, Ext, Type, Suffix, Postfix)                          \
  case X86::Prefix##PMOV##Ext##Type##Suffix##rm##Postfix:

#define CASE_MOVX_RM(Ext, Type)                                                \
  MOVX_CASE(, Ext, Type, , )                                                   \
  MOVX_CASE(V, Ext, Type, , )                                                  \
  MOVX_CASE(V, Ext, Type, Y, )                                                 \
  MOVX_CASE(V, Ext, Type, Z128, )                                              \
  MOVX_CASE(V, Ext, Type, Z128, k )                                            \
  MOVX_CASE(V, Ext, Type, Z128, kz )                                           \
  MOVX_CASE(V, Ext, Type, Z256, )                                              \
  MOVX_CASE(V, Ext, Type, Z256, k )                                            \
  MOVX_CASE(V, Ext, Type, Z256, kz )                                           \
  MOVX_CASE(V, Ext, Type, Z, )                                                 \
  MOVX_CASE(V, Ext, Type, Z, k )                                               \
  MOVX_CASE(V, Ext, Type, Z, kz )

    CASE_MOVX_RM(SX, BD)
    printSignExtend(MI, OutStreamer, 8, 32);
    break;
    CASE_MOVX_RM(SX, BQ)
    printSignExtend(MI, OutStreamer, 8, 64);
    break;
    CASE_MOVX_RM(SX, BW)
    printSignExtend(MI, OutStreamer, 8, 16);
    break;
    CASE_MOVX_RM(SX, DQ)
    printSignExtend(MI, OutStreamer, 32, 64);
    break;
    CASE_MOVX_RM(SX, WD)
    printSignExtend(MI, OutStreamer, 16, 32);
    break;
    CASE_MOVX_RM(SX, WQ)
    printSignExtend(MI, OutStreamer, 16, 64);
    break;

    CASE_MOVX_RM(ZX, BD)
    printZeroExtend(MI, OutStreamer, 8, 32);
    break;
    CASE_MOVX_RM(ZX, BQ)
    printZeroExtend(MI, OutStreamer, 8, 64);
    break;
    CASE_MOVX_RM(ZX, BW)
    printZeroExtend(MI, OutStreamer, 8, 16);
    break;
    CASE_MOVX_RM(ZX, DQ)
    printZeroExtend(MI, OutStreamer, 32, 64);
    break;
    CASE_MOVX_RM(ZX, WD)
    printZeroExtend(MI, OutStreamer, 16, 32);
    break;
    CASE_MOVX_RM(ZX, WQ)
    printZeroExtend(MI, OutStreamer, 16, 64);
    break;
  }
}

// Does the given operand refer to a DLLIMPORT function?
bool isImportedFunction(const MachineOperand &MO) {
  return MO.isGlobal() && (MO.getTargetFlags() == X86II::MO_DLLIMPORT);
}

// Is the given instruction a call to a CFGuard function?
bool isCallToCFGuardFunction(const MachineInstr *MI) {
  assert(MI->getOpcode() == X86::TAILJMPm64_REX ||
         MI->getOpcode() == X86::CALL64m);
  const MachineOperand &MO = MI->getOperand(3);
  return MO.isGlobal() && (MO.getTargetFlags() == X86II::MO_NO_FLAG) &&
         isCFGuardFunction(MO.getGlobal());
}

// Does the containing block for the given instruction contain any jump table
// info (indicating that the block is a dispatch for a jump table)?
bool hasJumpTableInfoInBlock(const llvm::MachineInstr *MI) {
  const MachineBasicBlock &MBB = *MI->getParent();
  for (auto I = MBB.instr_rbegin(), E = MBB.instr_rend(); I != E; ++I)
    if (I->isJumpTableDebugInfo())
      return true;

  return false;
}

void X86AsmPrinter::emitInstruction(const MachineInstr *MI) {
  // FIXME: Enable feature predicate checks once all the test pass.
  // X86_MC::verifyInstructionPredicates(MI->getOpcode(),
  //                                     Subtarget->getFeatureBits());

  X86MCInstLower MCInstLowering(*MF, *this);
  const X86RegisterInfo *RI =
      MF->getSubtarget<X86Subtarget>().getRegisterInfo();

  if (MI->getOpcode() == X86::OR64rm) {
    for (auto &Opd : MI->operands()) {
      if (Opd.isSymbol() && StringRef(Opd.getSymbolName()) ==
                                "swift_async_extendedFramePointerFlags") {
        ShouldEmitWeakSwiftAsyncExtendedFramePointerFlags = true;
      }
    }
  }

  // Add comments for values loaded from constant pool.
  if (OutStreamer->isVerboseAsm())
    addConstantComments(MI, *OutStreamer);

  // Add a comment about EVEX compression
  if (TM.Options.MCOptions.ShowMCEncoding) {
    if (MI->getAsmPrinterFlags() & X86::AC_EVEX_2_LEGACY)
      OutStreamer->AddComment("EVEX TO LEGACY Compression ", false);
    else if (MI->getAsmPrinterFlags() & X86::AC_EVEX_2_VEX)
      OutStreamer->AddComment("EVEX TO VEX Compression ", false);
    else if (MI->getAsmPrinterFlags() & X86::AC_EVEX_2_EVEX)
      OutStreamer->AddComment("EVEX TO EVEX Compression ", false);
  }

  // We use this to suppress NOP padding for Windows EH.
  bool IsTailJump = false;

  switch (MI->getOpcode()) {
  case TargetOpcode::DBG_VALUE:
    llvm_unreachable("Should be handled target independently");

  case X86::EH_RETURN:
  case X86::EH_RETURN64: {
    // Lower these as normal, but add some comments.
    Register Reg = MI->getOperand(0).getReg();
    OutStreamer->AddComment(StringRef("eh_return, addr: %") +
                            X86ATTInstPrinter::getRegisterName(Reg));
    break;
  }
  case X86::CLEANUPRET: {
    // Lower these as normal, but add some comments.
    OutStreamer->AddComment("CLEANUPRET");
    break;
  }

  case X86::CATCHRET: {
    // Lower these as normal, but add some comments.
    OutStreamer->AddComment("CATCHRET");
    break;
  }

  case X86::ENDBR32:
  case X86::ENDBR64: {
    // CurrentPatchableFunctionEntrySym can be CurrentFnBegin only for
    // -fpatchable-function-entry=N,0. The entry MBB is guaranteed to be
    // non-empty. If MI is the initial ENDBR, place the
    // __patchable_function_entries label after ENDBR.
    if (CurrentPatchableFunctionEntrySym &&
        CurrentPatchableFunctionEntrySym == CurrentFnBegin &&
        MI == &MF->front().front()) {
      MCInst Inst;
      MCInstLowering.Lower(MI, Inst);
      EmitAndCountInstruction(Inst);
      CurrentPatchableFunctionEntrySym = createTempSymbol("patch");
      OutStreamer->emitLabel(CurrentPatchableFunctionEntrySym);
      return;
    }
    break;
  }

  case X86::TAILJMPd64:
    if (IndCSPrefix && MI->hasRegisterImplicitUseOperand(X86::R11))
      EmitAndCountInstruction(MCInstBuilder(X86::CS_PREFIX));

    if (EnableImportCallOptimization && isImportedFunction(MI->getOperand(0))) {
      emitLabelAndRecordForImportCallOptimization(
          IMAGE_RETPOLINE_AMD64_IMPORT_BR);
    }

    // Lower this as normal, but add a comment.
    OutStreamer->AddComment("TAILCALL");
    IsTailJump = true;
    break;

  case X86::TAILJMPr:
  case X86::TAILJMPm:
  case X86::TAILJMPd:
  case X86::TAILJMPd_CC:
  case X86::TAILJMPr64:
  case X86::TAILJMPm64:
  case X86::TAILJMPd64_CC:
    if (EnableImportCallOptimization)
      report_fatal_error("Unexpected TAILJMP instruction was emitted when "
                         "import call optimization was enabled");

    // Lower these as normal, but add some comments.
    OutStreamer->AddComment("TAILCALL");
    IsTailJump = true;
    break;

  case X86::TAILJMPm64_REX:
    if (EnableImportCallOptimization && isCallToCFGuardFunction(MI)) {
      emitLabelAndRecordForImportCallOptimization(
          IMAGE_RETPOLINE_AMD64_CFG_BR_REX);
    }

    OutStreamer->AddComment("TAILCALL");
    IsTailJump = true;
    break;

  case X86::TAILJMPr64_REX: {
    if (EnableImportCallOptimization) {
      assert(MI->getOperand(0).getReg() == X86::RAX &&
             "Indirect tail calls with impcall enabled must go through RAX (as "
             "enforced by TCRETURNImpCallri64)");
      emitLabelAndRecordForImportCallOptimization(
          IMAGE_RETPOLINE_AMD64_INDIR_BR);
    }

    OutStreamer->AddComment("TAILCALL");
    IsTailJump = true;
    break;
  }

  case X86::JMP64r:
    if (EnableImportCallOptimization && hasJumpTableInfoInBlock(MI)) {
      uint16_t EncodedReg =
          this->getSubtarget().getRegisterInfo()->getEncodingValue(
              MI->getOperand(0).getReg().asMCReg());
      emitLabelAndRecordForImportCallOptimization(
          (ImportCallKind)(IMAGE_RETPOLINE_AMD64_SWITCHTABLE_FIRST +
                           EncodedReg));
    }
    break;

  case X86::JMP16r:
  case X86::JMP16m:
  case X86::JMP32r:
  case X86::JMP32m:
  case X86::JMP64m:
    if (EnableImportCallOptimization && hasJumpTableInfoInBlock(MI))
      report_fatal_error(
          "Unexpected JMP instruction was emitted for a jump-table when import "
          "call optimization was enabled");
    break;

  case X86::TLS_addr32:
  case X86::TLS_addr64:
  case X86::TLS_addrX32:
  case X86::TLS_base_addr32:
  case X86::TLS_base_addr64:
  case X86::TLS_base_addrX32:
  case X86::TLS_desc32:
  case X86::TLS_desc64:
    return LowerTlsAddr(MCInstLowering, *MI);

  case X86::MOVPC32r: {
    // This is a pseudo op for a two instruction sequence with a label, which
    // looks like:
    //     call "L1$pb"
    // "L1$pb":
    //     popl %esi

    // Emit the call.
    MCSymbol *PICBase = MF->getPICBaseSymbol();
    // FIXME: We would like an efficient form for this, so we don't have to do a
    // lot of extra uniquing.
    EmitAndCountInstruction(
        MCInstBuilder(X86::CALLpcrel32)
            .addExpr(MCSymbolRefExpr::create(PICBase, OutContext)));

    const X86FrameLowering *FrameLowering =
        MF->getSubtarget<X86Subtarget>().getFrameLowering();
    bool hasFP = FrameLowering->hasFP(*MF);

    // TODO: This is needed only if we require precise CFA.
    bool HasActiveDwarfFrame = OutStreamer->getNumFrameInfos() &&
                               !OutStreamer->getDwarfFrameInfos().back().End;

    int stackGrowth = -RI->getSlotSize();

    if (HasActiveDwarfFrame && !hasFP) {
      OutStreamer->emitCFIAdjustCfaOffset(-stackGrowth);
      MF->getInfo<X86MachineFunctionInfo>()->setHasCFIAdjustCfa(true);
    }

    // Emit the label.
    OutStreamer->emitLabel(PICBase);

    // popl $reg
    EmitAndCountInstruction(
        MCInstBuilder(X86::POP32r).addReg(MI->getOperand(0).getReg()));

    if (HasActiveDwarfFrame && !hasFP) {
      OutStreamer->emitCFIAdjustCfaOffset(stackGrowth);
    }
    return;
  }

  case X86::ADD32ri: {
    // Lower the MO_GOT_ABSOLUTE_ADDRESS form of ADD32ri.
    if (MI->getOperand(2).getTargetFlags() != X86II::MO_GOT_ABSOLUTE_ADDRESS)
      break;

    // Okay, we have something like:
    //  EAX = ADD32ri EAX, MO_GOT_ABSOLUTE_ADDRESS(@MYGLOBAL)

    // For this, we want to print something like:
    //   MYGLOBAL + (. - PICBASE)
    // However, we can't generate a ".", so just emit a new label here and refer
    // to it.
    MCSymbol *DotSym = OutContext.createTempSymbol();
    OutStreamer->emitLabel(DotSym);

    // Now that we have emitted the label, lower the complex operand expression.
    MCSymbol *OpSym = MCInstLowering.GetSymbolFromOperand(MI->getOperand(2));

    const MCExpr *DotExpr = MCSymbolRefExpr::create(DotSym, OutContext);
    const MCExpr *PICBase =
        MCSymbolRefExpr::create(MF->getPICBaseSymbol(), OutContext);
    DotExpr = MCBinaryExpr::createSub(DotExpr, PICBase, OutContext);

    DotExpr = MCBinaryExpr::createAdd(
        MCSymbolRefExpr::create(OpSym, OutContext), DotExpr, OutContext);

    EmitAndCountInstruction(MCInstBuilder(X86::ADD32ri)
                                .addReg(MI->getOperand(0).getReg())
                                .addReg(MI->getOperand(1).getReg())
                                .addExpr(DotExpr));
    return;
  }
  case TargetOpcode::STATEPOINT:
    return LowerSTATEPOINT(*MI, MCInstLowering);

  case TargetOpcode::FAULTING_OP:
    return LowerFAULTING_OP(*MI, MCInstLowering);

  case TargetOpcode::FENTRY_CALL:
    return LowerFENTRY_CALL(*MI, MCInstLowering);

  case TargetOpcode::PATCHABLE_OP:
    return LowerPATCHABLE_OP(*MI, MCInstLowering);

  case TargetOpcode::STACKMAP:
    return LowerSTACKMAP(*MI);

  case TargetOpcode::PATCHPOINT:
    return LowerPATCHPOINT(*MI, MCInstLowering);

  case TargetOpcode::PATCHABLE_FUNCTION_ENTER:
    return LowerPATCHABLE_FUNCTION_ENTER(*MI, MCInstLowering);

  case TargetOpcode::PATCHABLE_RET:
    return LowerPATCHABLE_RET(*MI, MCInstLowering);

  case TargetOpcode::PATCHABLE_TAIL_CALL:
    return LowerPATCHABLE_TAIL_CALL(*MI, MCInstLowering);

  case TargetOpcode::PATCHABLE_EVENT_CALL:
    return LowerPATCHABLE_EVENT_CALL(*MI, MCInstLowering);

  case TargetOpcode::PATCHABLE_TYPED_EVENT_CALL:
    return LowerPATCHABLE_TYPED_EVENT_CALL(*MI, MCInstLowering);

  case X86::MORESTACK_RET:
    EmitAndCountInstruction(MCInstBuilder(getRetOpcode(*Subtarget)));
    return;

  case X86::KCFI_CHECK:
    return LowerKCFI_CHECK(*MI);

  case X86::ASAN_CHECK_MEMACCESS:
    return LowerASAN_CHECK_MEMACCESS(*MI);

  case X86::MORESTACK_RET_RESTORE_R10:
    // Return, then restore R10.
    EmitAndCountInstruction(MCInstBuilder(getRetOpcode(*Subtarget)));
    EmitAndCountInstruction(
        MCInstBuilder(X86::MOV64rr).addReg(X86::R10).addReg(X86::RAX));
    return;

  case X86::SEH_PushReg:
  case X86::SEH_SaveReg:
  case X86::SEH_SaveXMM:
  case X86::SEH_StackAlloc:
  case X86::SEH_StackAlign:
  case X86::SEH_SetFrame:
  case X86::SEH_PushFrame:
  case X86::SEH_EndPrologue:
  case X86::SEH_EndEpilogue:
  case X86::SEH_UnwindV2Start:
  case X86::SEH_UnwindVersion:
    EmitSEHInstruction(MI);
    return;

  case X86::SEH_BeginEpilogue: {
    assert(MF->hasWinCFI() && "SEH_ instruction in function without WinCFI?");
    EmitSEHInstruction(MI);
    return;
  }
  case X86::UBSAN_UD1:
    EmitAndCountInstruction(MCInstBuilder(X86::UD1Lm)
                                .addReg(X86::EAX)
                                .addReg(X86::EAX)
                                .addImm(1)
                                .addReg(X86::NoRegister)
                                .addImm(MI->getOperand(0).getImm())
                                .addReg(X86::NoRegister));
    return;
  case X86::CALL64pcrel32:
    if (IndCSPrefix && MI->hasRegisterImplicitUseOperand(X86::R11))
      EmitAndCountInstruction(MCInstBuilder(X86::CS_PREFIX));

    if (EnableImportCallOptimization && isImportedFunction(MI->getOperand(0))) {
      emitLabelAndRecordForImportCallOptimization(
          IMAGE_RETPOLINE_AMD64_IMPORT_CALL);

      MCInst TmpInst;
      MCInstLowering.Lower(MI, TmpInst);

      // For Import Call Optimization to work, we need a the call instruction
      // with a rex prefix, and a 5-byte nop after the call instruction.
      EmitAndCountInstruction(MCInstBuilder(X86::REX64_PREFIX));
      emitCallInstruction(TmpInst);
      emitNop(*OutStreamer, 5, Subtarget);
      maybeEmitNopAfterCallForWindowsEH(MI);
      return;
    }

    break;

  case X86::CALL64r:
    if (EnableImportCallOptimization) {
      assert(MI->getOperand(0).getReg() == X86::RAX &&
             "Indirect calls with impcall enabled must go through RAX (as "
             "enforced by CALL64r_ImpCall)");

      emitLabelAndRecordForImportCallOptimization(
          IMAGE_RETPOLINE_AMD64_INDIR_CALL);
      MCInst TmpInst;
      MCInstLowering.Lower(MI, TmpInst);
      emitCallInstruction(TmpInst);

      // For Import Call Optimization to work, we need a 3-byte nop after the
      // call instruction.
      emitNop(*OutStreamer, 3, Subtarget);
      maybeEmitNopAfterCallForWindowsEH(MI);
      return;
    }
    break;

  case X86::CALL64m:
    if (EnableImportCallOptimization && isCallToCFGuardFunction(MI)) {
      emitLabelAndRecordForImportCallOptimization(
          IMAGE_RETPOLINE_AMD64_CFG_CALL);
    }
    break;

  case X86::JCC_1:
    // Two instruction prefixes (2EH for branch not-taken and 3EH for branch
    // taken) are used as branch hints. Here we add branch taken prefix for
    // jump instruction with higher probability than threshold.
    if (getSubtarget().hasBranchHint() && EnableBranchHint) {
      const MachineBranchProbabilityInfo *MBPI =
          &getAnalysis<MachineBranchProbabilityInfoWrapperPass>().getMBPI();
      MachineBasicBlock *DestBB = MI->getOperand(0).getMBB();
      BranchProbability EdgeProb =
          MBPI->getEdgeProbability(MI->getParent(), DestBB);
      BranchProbability Threshold(BranchHintProbabilityThreshold, 100);
      if (EdgeProb > Threshold)
        EmitAndCountInstruction(MCInstBuilder(X86::DS_PREFIX));
    }
    break;
  }

  MCInst TmpInst;
  MCInstLowering.Lower(MI, TmpInst);

  if (MI->isCall()) {
    emitCallInstruction(TmpInst);
    // Since tail calls transfer control without leaving a stack frame, there is
    // never a need for NOP padding tail calls.
    if (!IsTailJump)
      maybeEmitNopAfterCallForWindowsEH(MI);
    return;
  }

  EmitAndCountInstruction(TmpInst);
}

void X86AsmPrinter::emitCallInstruction(const llvm::MCInst &MCI) {
  // Stackmap shadows cannot include branch targets, so we can count the bytes
  // in a call towards the shadow, but must ensure that the no thread returns
  // in to the stackmap shadow.  The only way to achieve this is if the call
  // is at the end of the shadow.

  // Count then size of the call towards the shadow
  SMShadowTracker.count(MCI, getSubtargetInfo(), CodeEmitter.get());
  // Then flush the shadow so that we fill with nops before the call, not
  // after it.
  SMShadowTracker.emitShadowPadding(*OutStreamer, getSubtargetInfo());
  // Then emit the call
  OutStreamer->emitInstruction(MCI, getSubtargetInfo());
}

// Determines whether a NOP is required after a CALL, so that Windows EH
// IP2State tables have the correct information.
//
// On most Windows platforms (AMD64, ARM64, ARM32, IA64, but *not* x86-32),
// exception handling works by looking up instruction pointers in lookup
// tables. These lookup tables are stored in .xdata sections in executables.
// One element of the lookup tables are the "IP2State" tables (Instruction
// Pointer to State).
//
// If a function has any instructions that require cleanup during exception
// unwinding, then it will have an IP2State table. Each entry in the IP2State
// table describes a range of bytes in the function's instruction stream, and
// associates an "EH state number" with that range of instructions. A value of
// -1 means "the null state", which does not require any code to execute.
// A value other than -1 is an index into the State table.
//
// The entries in the IP2State table contain byte offsets within the instruction
// stream of the function. The Windows ABI requires that these offsets are
// aligned to instruction boundaries; they are not permitted to point to a byte
// that is not the first byte of an instruction.
//
// Unfortunately, CALL instructions present a problem during unwinding. CALL
// instructions push the address of the instruction after the CALL instruction,
// so that execution can resume after the CALL. If the CALL is the last
// instruction within an IP2State region, then the return address (on the stack)
// points to the *next* IP2State region. This means that the unwinder will
// use the wrong cleanup funclet during unwinding.
//
// To fix this problem, the Windows AMD64 ABI requires that CALL instructions
// are never placed at the end of an IP2State region. Stated equivalently, the
// end of a CALL instruction cannot be aligned to an IP2State boundary.  If a
// CALL instruction would occur at the end of an IP2State region, then the
// compiler must insert a NOP instruction after the CALL. The NOP instruction
// is placed in the same EH region as the CALL instruction, so that the return
// address points to the NOP and the unwinder will locate the correct region.
//
// NOP padding is only necessary on Windows AMD64 targets. On ARM64 and ARM32,
// instructions have a fixed size so the unwinder knows how to "back up" by
// one instruction.
//
// Interaction with Import Call Optimization (ICO):
//
// Import Call Optimization (ICO) is a compiler + OS feature on Windows which
// improves the performance and security of DLL imports. ICO relies on using a
// specific CALL idiom that can be replaced by the OS DLL loader. This removes
// a load and indirect CALL and replaces it with a single direct CALL.
//
// To achieve this, ICO also inserts NOPs after the CALL instruction. If the
// end of the CALL is aligned with an EH state transition, we *also* insert
// a single-byte NOP.  **Both forms of NOPs must be preserved.**  They cannot
// be combined into a single larger NOP; nor can the second NOP be removed.
//
// This is necessary because, if ICO is active and the call site is modified
// by the loader, the loader will end up overwriting the NOPs that were inserted
// for ICO. That means that those NOPs cannot be used for the correct
// termination of the exception handling region (the IP2State transition),
// so we still need an additional NOP instruction.  The NOPs cannot be combined
// into a longer NOP (which is ordinarily desirable) because then ICO would
// split one instruction, producing a malformed instruction after the ICO call.
void X86AsmPrinter::maybeEmitNopAfterCallForWindowsEH(const MachineInstr *MI) {
  // We only need to insert NOPs after CALLs when targeting Windows on AMD64.
  // (Don't let the name fool you: Itanium refers to table-based exception
  // handling, not the Itanium architecture.)
  if (MAI->getExceptionHandlingType() != ExceptionHandling::WinEH ||
      MAI->getWinEHEncodingType() != WinEH::EncodingType::Itanium) {
    return;
  }

  bool HasEHPersonality = MF->getWinEHFuncInfo() != nullptr;

  // Set up MBB iterator, initially positioned on the same MBB as MI.
  MachineFunction::const_iterator MFI(MI->getParent());
  MachineFunction::const_iterator MFE(MF->end());

  // Set up instruction iterator, positioned immediately *after* MI.
  MachineBasicBlock::const_iterator MBBI(MI);
  MachineBasicBlock::const_iterator MBBE = MI->getParent()->end();
  ++MBBI; // Step over MI

  // This loop iterates MBBs
  for (;;) {
    // This loop iterates instructions
    for (; MBBI != MBBE; ++MBBI) {
      // Check the instruction that follows this CALL.
      const MachineInstr &NextMI = *MBBI;

      // If there is an EH_LABEL after this CALL, then there is an EH state
      // transition after this CALL. This is exactly the situation which
      // requires NOP padding.
      if (NextMI.isEHLabel()) {
        if (HasEHPersonality) {
          EmitAndCountInstruction(MCInstBuilder(X86::NOOP));
          return;
        }
        // We actually want to continue, in case there is an SEH_BeginEpilogue
        // instruction after the EH_LABEL. In some situations, IR is produced
        // that contains EH_LABEL pseudo-instructions, even when we are not
        // generating IP2State tables. We still need to insert a NOP before
        // SEH_BeginEpilogue in that case.
        continue;
      }

      // Somewhat similarly, if the CALL is the last instruction before the
      // SEH prologue, then we also need a NOP. This is necessary because the
      // Windows stack unwinder will not invoke a function's exception handler
      // if the instruction pointer is in the function prologue or epilogue.
      //
      // We always emit a NOP before SEH_BeginEpilogue, even if there is no
      // personality function (unwind info) for this frame. This is the same
      // behavior as MSVC.
      if (NextMI.getOpcode() == X86::SEH_BeginEpilogue) {
        EmitAndCountInstruction(MCInstBuilder(X86::NOOP));
        return;
      }

      if (!NextMI.isPseudo() && !NextMI.isMetaInstruction()) {
        // We found a real instruction. During the CALL, the return IP will
        // point to this instruction. Since this instruction has the same EH
        // state as the call itself (because there is no intervening EH_LABEL),
        // the IP2State table will be accurate; there is no need to insert a
        // NOP.
        return;
      }

      // The next instruction is a pseudo-op. Ignore it and keep searching.
      // Because these instructions do not generate any machine code, they
      // cannot prevent the IP2State table from pointing at the wrong
      // instruction during a CALL.
    }

    // We've reached the end of this MBB. Find the next MBB in program order.
    // MBB order should be finalized by this point, so falling across MBBs is
    // expected.
    ++MFI;
    if (MFI == MFE) {
      // No more blocks; we've reached the end of the function. This should
      // only happen with no-return functions, but double-check to be sure.
      if (HasEHPersonality) {
        // If the CALL has no successors, then it is a noreturn function.
        // Insert an INT3 instead of a NOP. This accomplishes the same purpose,
        // but is more clear to read. Also, analysis tools will understand
        // that they should not continue disassembling after the CALL (unless
        // there are other branches to that label).
        if (MI->getParent()->succ_empty())
          EmitAndCountInstruction(MCInstBuilder(X86::INT3));
        else
          EmitAndCountInstruction(MCInstBuilder(X86::NOOP));
      }
      return;
    }

    // Set up iterator to scan the next basic block.
    const MachineBasicBlock *NextMBB = &*MFI;
    MBBI = NextMBB->instr_begin();
    MBBE = NextMBB->instr_end();
  }
}

void X86AsmPrinter::emitLabelAndRecordForImportCallOptimization(
    ImportCallKind Kind) {
  assert(EnableImportCallOptimization);

  MCSymbol *CallSiteSymbol = MMI->getContext().createNamedTempSymbol("impcall");
  OutStreamer->emitLabel(CallSiteSymbol);

  SectionToImportedFunctionCalls[OutStreamer->getCurrentSectionOnly()]
      .push_back({CallSiteSymbol, Kind});
}
