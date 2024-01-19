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
#include "MCTargetDesc/X86ShuffleDecode.h"
#include "MCTargetDesc/X86TargetStreamer.h"
#include "X86AsmPrinter.h"
#include "X86MachineFunctionInfo.h"
#include "X86RegisterInfo.h"
#include "X86ShuffleDecodeConstantPool.h"
#include "X86Subtarget.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/StackMaps.h"
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
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizer.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizerCommon.h"
#include <string>

using namespace llvm;

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

  std::optional<MCOperand> LowerMachineOperand(const MachineInstr *MI,
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

void X86AsmPrinter::StackMapShadowTracker::count(MCInst &Inst,
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
    : Ctx(mf.getContext()), MF(mf), TM(mf.getTarget()), MAI(*TM.getMCAsmInfo()),
      AsmPrinter(asmprinter) {}

MachineModuleInfoMachO &X86MCInstLower::getMachOMMI() const {
  return MF.getMMI().getObjFileInfo<MachineModuleInfoMachO>();
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
        MF.getMMI().getObjFileInfo<MachineModuleInfoCOFF>();
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
  MCSymbolRefExpr::VariantKind RefKind = MCSymbolRefExpr::VK_None;

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
    RefKind = MCSymbolRefExpr::VK_TLVP;
    break;
  case X86II::MO_TLVP_PIC_BASE:
    Expr = MCSymbolRefExpr::create(Sym, MCSymbolRefExpr::VK_TLVP, Ctx);
    // Subtract the pic base.
    Expr = MCBinaryExpr::createSub(
        Expr, MCSymbolRefExpr::create(MF.getPICBaseSymbol(), Ctx), Ctx);
    break;
  case X86II::MO_SECREL:
    RefKind = MCSymbolRefExpr::VK_SECREL;
    break;
  case X86II::MO_TLSGD:
    RefKind = MCSymbolRefExpr::VK_TLSGD;
    break;
  case X86II::MO_TLSLD:
    RefKind = MCSymbolRefExpr::VK_TLSLD;
    break;
  case X86II::MO_TLSLDM:
    RefKind = MCSymbolRefExpr::VK_TLSLDM;
    break;
  case X86II::MO_GOTTPOFF:
    RefKind = MCSymbolRefExpr::VK_GOTTPOFF;
    break;
  case X86II::MO_INDNTPOFF:
    RefKind = MCSymbolRefExpr::VK_INDNTPOFF;
    break;
  case X86II::MO_TPOFF:
    RefKind = MCSymbolRefExpr::VK_TPOFF;
    break;
  case X86II::MO_DTPOFF:
    RefKind = MCSymbolRefExpr::VK_DTPOFF;
    break;
  case X86II::MO_NTPOFF:
    RefKind = MCSymbolRefExpr::VK_NTPOFF;
    break;
  case X86II::MO_GOTNTPOFF:
    RefKind = MCSymbolRefExpr::VK_GOTNTPOFF;
    break;
  case X86II::MO_GOTPCREL:
    RefKind = MCSymbolRefExpr::VK_GOTPCREL;
    break;
  case X86II::MO_GOTPCREL_NORELAX:
    RefKind = MCSymbolRefExpr::VK_GOTPCREL_NORELAX;
    break;
  case X86II::MO_GOT:
    RefKind = MCSymbolRefExpr::VK_GOT;
    break;
  case X86II::MO_GOTOFF:
    RefKind = MCSymbolRefExpr::VK_GOTOFF;
    break;
  case X86II::MO_PLT:
    RefKind = MCSymbolRefExpr::VK_PLT;
    break;
  case X86II::MO_ABS8:
    RefKind = MCSymbolRefExpr::VK_X86_ABS8;
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
    Expr = MCSymbolRefExpr::create(Sym, RefKind, Ctx);

  if (!MO.isJTI() && !MO.isMBB() && MO.getOffset())
    Expr = MCBinaryExpr::createAdd(
        Expr, MCConstantExpr::create(MO.getOffset(), Ctx), Ctx);
  return MCOperand::createExpr(Expr);
}

static unsigned getRetOpcode(const X86Subtarget &Subtarget) {
  return Subtarget.is64Bit() ? X86::RET64 : X86::RET32;
}

std::optional<MCOperand>
X86MCInstLower::LowerMachineOperand(const MachineInstr *MI,
                                    const MachineOperand &MO) const {
  switch (MO.getType()) {
  default:
    MI->print(errs());
    llvm_unreachable("unknown operand type");
  case MachineOperand::MO_Register:
    // Ignore all implicit register operands.
    if (MO.isImplicit())
      return std::nullopt;
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
    return std::nullopt;
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
    if (auto MaybeMCOp = LowerMachineOperand(MI, MO))
      OutMI.addOperand(*MaybeMCOp);

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
    unsigned DestReg = OutMI.getOperand(0).getReg();
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
    const MachineOperand *FlagDef = MI->findRegisterDefOperand(X86::EFLAGS);
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
  bool Is64Bits = MI.getOpcode() != X86::TLS_addr32 &&
                  MI.getOpcode() != X86::TLS_base_addr32;
  bool Is64BitsLP64 = MI.getOpcode() == X86::TLS_addr64 ||
                      MI.getOpcode() == X86::TLS_base_addr64;
  MCContext &Ctx = OutStreamer->getContext();

  MCSymbolRefExpr::VariantKind SRVK;
  switch (MI.getOpcode()) {
  case X86::TLS_addr32:
  case X86::TLS_addr64:
  case X86::TLS_addrX32:
    SRVK = MCSymbolRefExpr::VK_TLSGD;
    break;
  case X86::TLS_base_addr32:
    SRVK = MCSymbolRefExpr::VK_TLSLDM;
    break;
  case X86::TLS_base_addr64:
  case X86::TLS_base_addrX32:
    SRVK = MCSymbolRefExpr::VK_TLSLD;
    break;
  default:
    llvm_unreachable("unexpected opcode");
  }

  const MCSymbolRefExpr *Sym = MCSymbolRefExpr::create(
      MCInstLowering.GetSymbolFromOperand(MI.getOperand(3)), SRVK, Ctx);

  // As of binutils 2.32, ld has a bogus TLS relaxation error when the GD/LD
  // code sequence using R_X86_64_GOTPCREL (instead of R_X86_64_GOTPCRELX) is
  // attempted to be relaxed to IE/LE (binutils PR24784). Work around the bug by
  // only using GOT when GOTPCRELX is enabled.
  // TODO Delete the workaround when GOTPCRELX becomes commonplace.
  bool UseGot = MMI->getModule()->getRtLibUseGOT() &&
                Ctx.getAsmInfo()->canRelaxRelocations();

  if (Is64Bits) {
    bool NeedsPadding = SRVK == MCSymbolRefExpr::VK_TLSGD;
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
      const MCExpr *Expr = MCSymbolRefExpr::create(
          TlsGetAddr, MCSymbolRefExpr::VK_GOTPCREL, Ctx);
      EmitAndCountInstruction(MCInstBuilder(X86::CALL64m)
                                  .addReg(X86::RIP)
                                  .addImm(1)
                                  .addReg(0)
                                  .addExpr(Expr)
                                  .addReg(0));
    } else {
      EmitAndCountInstruction(
          MCInstBuilder(X86::CALL64pcrel32)
              .addExpr(MCSymbolRefExpr::create(TlsGetAddr,
                                               MCSymbolRefExpr::VK_PLT, Ctx)));
    }
  } else {
    if (SRVK == MCSymbolRefExpr::VK_TLSGD && !UseGot) {
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
      const MCExpr *Expr =
          MCSymbolRefExpr::create(TlsGetAddr, MCSymbolRefExpr::VK_GOT, Ctx);
      EmitAndCountInstruction(MCInstBuilder(X86::CALL32m)
                                  .addReg(X86::EBX)
                                  .addImm(1)
                                  .addReg(0)
                                  .addExpr(Expr)
                                  .addReg(0));
    } else {
      EmitAndCountInstruction(
          MCInstBuilder(X86::CALLpcrel32)
              .addExpr(MCSymbolRefExpr::create(TlsGetAddr,
                                               MCSymbolRefExpr::VK_PLT, Ctx)));
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
    if (auto MaybeOperand = MCIL.LowerMachineOperand(&FaultingMI, MO))
      MI.addOperand(*MaybeOperand);

  OutStreamer->AddComment("on-fault: " + HandlerLabel->getName());
  OutStreamer->emitInstruction(MI, getSubtargetInfo());
}

void X86AsmPrinter::LowerFENTRY_CALL(const MachineInstr &MI,
                                     X86MCInstLower &MCIL) {
  bool Is64Bits = Subtarget->is64Bit();
  MCContext &Ctx = OutStreamer->getContext();
  MCSymbol *fentry = Ctx.getOrCreateSymbol("__fentry__");
  const MCSymbolRefExpr *Op =
      MCSymbolRefExpr::create(fentry, MCSymbolRefExpr::VK_None, Ctx);

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
  getAddressSanitizerParams(Triple(TM.getTargetTriple()), 64,
                            AccessInfo.CompileKernel, &ShadowBase,
                            &MappingScale, &OrShadowOffset);

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
  // PATCHABLE_OP minsize, opcode, operands

  NoAutoPaddingScope NoPadScope(*OutStreamer);

  unsigned MinSize = MI.getOperand(0).getImm();
  unsigned Opcode = MI.getOperand(1).getImm();
  // Opcode PATCHABLE_OP is a special case: there is no instruction to wrap,
  // simply emit a nop of size MinSize.
  bool EmptyInst = (Opcode == TargetOpcode::PATCHABLE_OP);

  MCInst MCI;
  MCI.setOpcode(Opcode);
  for (auto &MO : drop_begin(MI.operands(), 2))
    if (auto MaybeOperand = MCIL.LowerMachineOperand(&MI, MO))
      MCI.addOperand(*MaybeOperand);

  SmallString<256> Code;
  if (!EmptyInst) {
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
    } else if (MinSize == 2 && Opcode == X86::PUSH64r) {
      // This is an optimization that lets us get away without emitting a nop in
      // many cases.
      //
      // NB! In some cases the encoding for PUSH64r (e.g. PUSH64r %r9) takes two
      // bytes too, so the check on MinSize is important.
      MCI.setOpcode(X86::PUSH64rmr);
    } else {
      unsigned NopSize = emitNop(*OutStreamer, MinSize, Subtarget);
      assert(NopSize == MinSize && "Could not implement MinSize!");
      (void)NopSize;
    }
  }
  if (!EmptyInst)
    OutStreamer->emitInstruction(MCI, getSubtargetInfo());
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
    if (auto Op = MCIL.LowerMachineOperand(&MI, MI.getOperand(I))) {
      assert(Op->isReg() && "Only support arguments in registers");
      SrcRegs[I] = getX86SubSuperRegister(Op->getReg(), 64);
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
    if (auto Op = MCIL.LowerMachineOperand(&MI, MI.getOperand(I))) {
      // TODO: Is register only support adequate?
      assert(Op->isReg() && "Only supports arguments in registers");
      SrcRegs[I] = getX86SubSuperRegister(Op->getReg(), 64);
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
    if (auto MaybeOperand = MCIL.LowerMachineOperand(&MI, MO))
      Ret.addOperand(*MaybeOperand);
  OutStreamer->emitInstruction(Ret, getSubtargetInfo());
  emitX86Nops(*OutStreamer, 10, Subtarget);
  recordSled(CurSled, MI, SledKind::FUNCTION_EXIT, 2);
}

void X86AsmPrinter::LowerPATCHABLE_TAIL_CALL(const MachineInstr &MI,
                                             X86MCInstLower &MCIL) {
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

  unsigned OpCode = MI.getOperand(0).getImm();
  OpCode = convertTailJumpOpcode(OpCode);
  MCInst TC;
  TC.setOpcode(OpCode);

  // Before emitting the instruction, add a comment to indicate that this is
  // indeed a tail call.
  OutStreamer->AddComment("TAILCALL");
  for (auto &MO : drop_begin(MI.operands()))
    if (auto MaybeOperand = MCIL.LowerMachineOperand(&MI, MO))
      TC.addOperand(*MaybeOperand);
  OutStreamer->emitInstruction(TC, getSubtargetInfo());
}

// Returns instruction preceding MBBI in MachineFunction.
// If MBBI is the first instruction of the first basic block, returns null.
static MachineBasicBlock::const_iterator
PrevCrossBBInst(MachineBasicBlock::const_iterator MBBI) {
  const MachineBasicBlock *MBB = MBBI->getParent();
  while (MBBI == MBB->begin()) {
    if (MBB == &MBB->getParent()->front())
      return MachineBasicBlock::const_iterator();
    MBB = MBB->getPrevNode();
    MBBI = MBB->end();
  }
  --MBBI;
  return MBBI;
}

static std::string getShuffleComment(const MachineInstr *MI, unsigned SrcOp1Idx,
                                     unsigned SrcOp2Idx, ArrayRef<int> Mask) {
  std::string Comment;

  // Compute the name for a register. This is really goofy because we have
  // multiple instruction printers that could (in theory) use different
  // names. Fortunately most people use the ATT style (outside of Windows)
  // and they actually agree on register naming here. Ultimately, this is
  // a comment, and so its OK if it isn't perfect.
  auto GetRegisterName = [](MCRegister Reg) -> StringRef {
    return X86ATTInstPrinter::getRegisterName(Reg);
  };

  const MachineOperand &DstOp = MI->getOperand(0);
  const MachineOperand &SrcOp1 = MI->getOperand(SrcOp1Idx);
  const MachineOperand &SrcOp2 = MI->getOperand(SrcOp2Idx);

  StringRef DstName = DstOp.isReg() ? GetRegisterName(DstOp.getReg()) : "mem";
  StringRef Src1Name =
      SrcOp1.isReg() ? GetRegisterName(SrcOp1.getReg()) : "mem";
  StringRef Src2Name =
      SrcOp2.isReg() ? GetRegisterName(SrcOp2.getReg()) : "mem";

  // One source operand, fix the mask to print all elements in one span.
  SmallVector<int, 8> ShuffleMask(Mask);
  if (Src1Name == Src2Name)
    for (int i = 0, e = ShuffleMask.size(); i != e; ++i)
      if (ShuffleMask[i] >= e)
        ShuffleMask[i] -= e;

  raw_string_ostream CS(Comment);
  CS << DstName;

  // Handle AVX512 MASK/MASXZ write mask comments.
  // MASK: zmmX {%kY}
  // MASKZ: zmmX {%kY} {z}
  if (SrcOp1Idx > 1) {
    assert((SrcOp1Idx == 2 || SrcOp1Idx == 3) && "Unexpected writemask");

    const MachineOperand &WriteMaskOp = MI->getOperand(SrcOp1Idx - 1);
    if (WriteMaskOp.isReg()) {
      CS << " {%" << GetRegisterName(WriteMaskOp.getReg()) << "}";

      if (SrcOp1Idx == 2) {
        CS << " {z}";
      }
    }
  }

  CS << " = ";

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
  CS.flush();

  return Comment;
}

static void printConstant(const APInt &Val, raw_ostream &CS) {
  if (Val.getBitWidth() <= 64) {
    CS << Val.getZExtValue();
  } else {
    // print multi-word constant as (w0,w1)
    CS << "(";
    for (int i = 0, N = Val.getNumWords(); i < N; ++i) {
      if (i > 0)
        CS << ",";
      CS << Val.getRawData()[i];
    }
    CS << ")";
  }
}

static void printConstant(const APFloat &Flt, raw_ostream &CS) {
  SmallString<32> Str;
  // Force scientific notation to distinguish from integers.
  Flt.toString(Str, 0, 0);
  CS << Str;
}

static void printConstant(const Constant *COp, unsigned BitWidth,
                          raw_ostream &CS) {
  if (isa<UndefValue>(COp)) {
    CS << "u";
  } else if (auto *CI = dyn_cast<ConstantInt>(COp)) {
    printConstant(CI->getValue(), CS);
  } else if (auto *CF = dyn_cast<ConstantFP>(COp)) {
    printConstant(CF->getValueAPF(), CS);
  } else if (auto *CDS = dyn_cast<ConstantDataSequential>(COp)) {
    Type *EltTy = CDS->getElementType();
    bool IsInteger = EltTy->isIntegerTy();
    bool IsFP = EltTy->isHalfTy() || EltTy->isFloatTy() || EltTy->isDoubleTy();
    unsigned EltBits = EltTy->getPrimitiveSizeInBits();
    unsigned E = std::min(BitWidth / EltBits, CDS->getNumElements());
    assert((BitWidth % EltBits) == 0 && "Broadcast element size mismatch");
    for (unsigned I = 0; I != E; ++I) {
      if (I != 0)
        CS << ",";
      if (IsInteger)
        printConstant(CDS->getElementAsAPInt(I), CS);
      else if (IsFP)
        printConstant(CDS->getElementAsAPFloat(I), CS);
      else
        CS << "?";
    }
  } else {
    CS << "?";
  }
}

void X86AsmPrinter::EmitSEHInstruction(const MachineInstr *MI) {
  assert(MF->hasWinCFI() && "SEH_ instruction in function without WinCFI?");
  assert((getSubtarget().isOSWindows() || TM.getTargetTriple().isUEFI()) &&
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

  default:
    llvm_unreachable("expected SEH_ instruction");
  }
}

static unsigned getRegisterWidth(const MCOperandInfo &Info) {
  if (Info.RegClass == X86::VR128RegClassID ||
      Info.RegClass == X86::VR128XRegClassID)
    return 128;
  if (Info.RegClass == X86::VR256RegClassID ||
      Info.RegClass == X86::VR256XRegClassID)
    return 256;
  if (Info.RegClass == X86::VR512RegClassID)
    return 512;
  llvm_unreachable("Unknown register class!");
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
    unsigned SrcIdx = 1;
    if (X86II::isKMasked(MI->getDesc().TSFlags)) {
      // Skip mask operand.
      ++SrcIdx;
      if (X86II::isKMergeMasked(MI->getDesc().TSFlags)) {
        // Skip passthru operand.
        ++SrcIdx;
      }
    }
    unsigned MaskIdx = SrcIdx + 1 + X86::AddrDisp;

    assert(MI->getNumOperands() >= (SrcIdx + 1 + X86::AddrNumOperands) &&
           "Unexpected number of operands!");

    const MachineOperand &MaskOp = MI->getOperand(MaskIdx);
    if (auto *C = X86::getConstantFromPool(*MI, MaskOp)) {
      unsigned Width = getRegisterWidth(MI->getDesc().operands()[0]);
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
  case X86::VPERMILPSZrmkz:
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
    unsigned ElSize;
    switch (MI->getOpcode()) {
    default: llvm_unreachable("Invalid opcode");
    case X86::VPERMILPSrm:
    case X86::VPERMILPSYrm:
    case X86::VPERMILPSZ128rm:
    case X86::VPERMILPSZ256rm:
    case X86::VPERMILPSZrm:
    case X86::VPERMILPSZ128rmkz:
    case X86::VPERMILPSZ256rmkz:
    case X86::VPERMILPSZrmkz:
    case X86::VPERMILPSZ128rmk:
    case X86::VPERMILPSZ256rmk:
    case X86::VPERMILPSZrmk:
      ElSize = 32;
      break;
    case X86::VPERMILPDrm:
    case X86::VPERMILPDYrm:
    case X86::VPERMILPDZ128rm:
    case X86::VPERMILPDZ256rm:
    case X86::VPERMILPDZrm:
    case X86::VPERMILPDZ128rmkz:
    case X86::VPERMILPDZ256rmkz:
    case X86::VPERMILPDZrmkz:
    case X86::VPERMILPDZ128rmk:
    case X86::VPERMILPDZ256rmk:
    case X86::VPERMILPDZrmk:
      ElSize = 64;
      break;
    }

    unsigned SrcIdx = 1;
    if (X86II::isKMasked(MI->getDesc().TSFlags)) {
      // Skip mask operand.
      ++SrcIdx;
      if (X86II::isKMergeMasked(MI->getDesc().TSFlags)) {
        // Skip passthru operand.
        ++SrcIdx;
      }
    }
    unsigned MaskIdx = SrcIdx + 1 + X86::AddrDisp;

    assert(MI->getNumOperands() >= (SrcIdx + 1 + X86::AddrNumOperands) &&
           "Unexpected number of operands!");

    const MachineOperand &MaskOp = MI->getOperand(MaskIdx);
    if (auto *C = X86::getConstantFromPool(*MI, MaskOp)) {
      unsigned Width = getRegisterWidth(MI->getDesc().operands()[0]);
      SmallVector<int, 16> Mask;
      DecodeVPERMILPMask(C, ElSize, Width, Mask);
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

    const MachineOperand &MaskOp = MI->getOperand(3 + X86::AddrDisp);
    if (auto *C = X86::getConstantFromPool(*MI, MaskOp)) {
      unsigned Width = getRegisterWidth(MI->getDesc().operands()[0]);
      SmallVector<int, 16> Mask;
      DecodeVPERMIL2PMask(C, (unsigned)CtrlOp.getImm(), ElSize, Width, Mask);
      if (!Mask.empty())
        OutStreamer.AddComment(getShuffleComment(MI, 1, 2, Mask));
    }
    break;
  }

  case X86::VPPERMrrm: {
    assert(MI->getNumOperands() >= (3 + X86::AddrNumOperands) &&
           "Unexpected number of operands!");

    const MachineOperand &MaskOp = MI->getOperand(3 + X86::AddrDisp);
    if (auto *C = X86::getConstantFromPool(*MI, MaskOp)) {
      unsigned Width = getRegisterWidth(MI->getDesc().operands()[0]);
      SmallVector<int, 16> Mask;
      DecodeVPPERMMask(C, Width, Mask);
      if (!Mask.empty())
        OutStreamer.AddComment(getShuffleComment(MI, 1, 2, Mask));
    }
    break;
  }

  case X86::MMX_MOVQ64rm: {
    assert(MI->getNumOperands() == (1 + X86::AddrNumOperands) &&
           "Unexpected number of operands!");
    if (auto *C =
            X86::getConstantFromPool(*MI, MI->getOperand(1 + X86::AddrDisp))) {
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

  case X86::MOVSDrm:
  case X86::MOVSSrm:
  case X86::VMOVSDrm:
  case X86::VMOVSSrm:
  case X86::VMOVSDZrm:
  case X86::VMOVSSZrm:
  case X86::MOVSDrm_alt:
  case X86::MOVSSrm_alt:
  case X86::VMOVSDrm_alt:
  case X86::VMOVSSrm_alt:
  case X86::VMOVSDZrm_alt:
  case X86::VMOVSSZrm_alt:
  case X86::MOVDI2PDIrm:
  case X86::MOVQI2PQIrm:
  case X86::VMOVDI2PDIrm:
  case X86::VMOVQI2PQIrm:
  case X86::VMOVDI2PDIZrm:
  case X86::VMOVQI2PQIZrm: {
    assert(MI->getNumOperands() >= (1 + X86::AddrNumOperands) &&
           "Unexpected number of operands!");
    int SclWidth = 32;
    int VecWidth = 128;

    switch (MI->getOpcode()) {
    default:
      llvm_unreachable("Invalid opcode");
    case X86::MOVSDrm:
    case X86::VMOVSDrm:
    case X86::VMOVSDZrm:
    case X86::MOVSDrm_alt:
    case X86::VMOVSDrm_alt:
    case X86::VMOVSDZrm_alt:
    case X86::MOVQI2PQIrm:
    case X86::VMOVQI2PQIrm:
    case X86::VMOVQI2PQIZrm:
      SclWidth = 64;
      VecWidth = 128;
      break;
    case X86::MOVSSrm:
    case X86::VMOVSSrm:
    case X86::VMOVSSZrm:
    case X86::MOVSSrm_alt:
    case X86::VMOVSSrm_alt:
    case X86::VMOVSSZrm_alt:
    case X86::MOVDI2PDIrm:
    case X86::VMOVDI2PDIrm:
    case X86::VMOVDI2PDIZrm:
      SclWidth = 32;
      VecWidth = 128;
      break;
    }
    std::string Comment;
    raw_string_ostream CS(Comment);
    const MachineOperand &DstOp = MI->getOperand(0);
    CS << X86ATTInstPrinter::getRegisterName(DstOp.getReg()) << " = ";

    if (auto *C =
            X86::getConstantFromPool(*MI, MI->getOperand(1 + X86::AddrDisp))) {
      if ((unsigned)SclWidth == C->getType()->getScalarSizeInBits()) {
        if (auto *CI = dyn_cast<ConstantInt>(C)) {
          CS << "[";
          printConstant(CI->getValue(), CS);
          for (int I = 1, E = VecWidth / SclWidth; I < E; ++I) {
            CS << ",0";
          }
          CS << "]";
          OutStreamer.AddComment(CS.str());
          break; // early-out
        }
        if (auto *CF = dyn_cast<ConstantFP>(C)) {
          CS << "[";
          printConstant(CF->getValue(), CS);
          APFloat ZeroFP = APFloat::getZero(CF->getValue().getSemantics());
          for (int I = 1, E = VecWidth / SclWidth; I < E; ++I) {
            CS << ",";
            printConstant(ZeroFP, CS);
          }
          CS << "]";
          OutStreamer.AddComment(CS.str());
          break; // early-out
        }
      }
    }

    // We didn't find a constant load, fallback to a shuffle mask decode.
    CS << (SclWidth == 32 ? "mem[0],zero,zero,zero" : "mem[0],zero");
    OutStreamer.AddComment(CS.str());
    break;
  }

#define MOV_CASE(Prefix, Suffix)                                               \
  case X86::Prefix##MOVAPD##Suffix##rm:                                        \
  case X86::Prefix##MOVAPS##Suffix##rm:                                        \
  case X86::Prefix##MOVUPD##Suffix##rm:                                        \
  case X86::Prefix##MOVUPS##Suffix##rm:                                        \
  case X86::Prefix##MOVDQA##Suffix##rm:                                        \
  case X86::Prefix##MOVDQU##Suffix##rm:

#define MOV_AVX512_CASE(Suffix)                                                \
  case X86::VMOVDQA64##Suffix##rm:                                             \
  case X86::VMOVDQA32##Suffix##rm:                                             \
  case X86::VMOVDQU64##Suffix##rm:                                             \
  case X86::VMOVDQU32##Suffix##rm:                                             \
  case X86::VMOVDQU16##Suffix##rm:                                             \
  case X86::VMOVDQU8##Suffix##rm:                                              \
  case X86::VMOVAPS##Suffix##rm:                                               \
  case X86::VMOVAPD##Suffix##rm:                                               \
  case X86::VMOVUPS##Suffix##rm:                                               \
  case X86::VMOVUPD##Suffix##rm:

#define CASE_128_MOV_RM()                                                      \
  MOV_CASE(, )   /* SSE */                                                     \
  MOV_CASE(V, )  /* AVX-128 */                                                 \
  MOV_AVX512_CASE(Z128)

#define CASE_256_MOV_RM()                                                      \
  MOV_CASE(V, Y) /* AVX-256 */                                                 \
  MOV_AVX512_CASE(Z256)

#define CASE_512_MOV_RM()                                                      \
  MOV_AVX512_CASE(Z)

#define CASE_ALL_MOV_RM()                                                      \
  MOV_CASE(, )   /* SSE */                                                     \
  MOV_CASE(V, )  /* AVX-128 */                                                 \
  MOV_CASE(V, Y) /* AVX-256 */                                                 \
  MOV_AVX512_CASE(Z)                                                           \
  MOV_AVX512_CASE(Z256)                                                        \
  MOV_AVX512_CASE(Z128)

    // For loads from a constant pool to a vector register, print the constant
    // loaded.
    CASE_ALL_MOV_RM()
  case X86::VBROADCASTF128rm:
  case X86::VBROADCASTI128rm:
  case X86::VBROADCASTF32X4Z256rm:
  case X86::VBROADCASTF32X4rm:
  case X86::VBROADCASTF32X8rm:
  case X86::VBROADCASTF64X2Z128rm:
  case X86::VBROADCASTF64X2rm:
  case X86::VBROADCASTF64X4rm:
  case X86::VBROADCASTI32X4Z256rm:
  case X86::VBROADCASTI32X4rm:
  case X86::VBROADCASTI32X8rm:
  case X86::VBROADCASTI64X2Z128rm:
  case X86::VBROADCASTI64X2rm:
  case X86::VBROADCASTI64X4rm:
    assert(MI->getNumOperands() >= (1 + X86::AddrNumOperands) &&
           "Unexpected number of operands!");
    if (auto *C =
            X86::getConstantFromPool(*MI, MI->getOperand(1 + X86::AddrDisp))) {
      int NumLanes = 1;
      int BitWidth = 128;
      int CstEltSize = C->getType()->getScalarSizeInBits();

      // Get destination BitWidth + override NumLanes for the broadcasts.
      switch (MI->getOpcode()) {
      CASE_128_MOV_RM()                NumLanes = 1; BitWidth = 128; break;
      CASE_256_MOV_RM()                NumLanes = 1; BitWidth = 256; break;
      CASE_512_MOV_RM()                NumLanes = 1; BitWidth = 512; break;
      case X86::VBROADCASTF128rm:      NumLanes = 2; BitWidth = 128; break;
      case X86::VBROADCASTI128rm:      NumLanes = 2; BitWidth = 128; break;
      case X86::VBROADCASTF32X4Z256rm: NumLanes = 2; BitWidth = 128; break;
      case X86::VBROADCASTF32X4rm:     NumLanes = 4; BitWidth = 128; break;
      case X86::VBROADCASTF32X8rm:     NumLanes = 2; BitWidth = 256; break;
      case X86::VBROADCASTF64X2Z128rm: NumLanes = 2; BitWidth = 128; break;
      case X86::VBROADCASTF64X2rm:     NumLanes = 4; BitWidth = 128; break;
      case X86::VBROADCASTF64X4rm:     NumLanes = 2; BitWidth = 256; break;
      case X86::VBROADCASTI32X4Z256rm: NumLanes = 2; BitWidth = 128; break;
      case X86::VBROADCASTI32X4rm:     NumLanes = 4; BitWidth = 128; break;
      case X86::VBROADCASTI32X8rm:     NumLanes = 2; BitWidth = 256; break;
      case X86::VBROADCASTI64X2Z128rm: NumLanes = 2; BitWidth = 128; break;
      case X86::VBROADCASTI64X2rm:     NumLanes = 4; BitWidth = 128; break;
      case X86::VBROADCASTI64X4rm:     NumLanes = 2; BitWidth = 256; break;
      }

      std::string Comment;
      raw_string_ostream CS(Comment);
      const MachineOperand &DstOp = MI->getOperand(0);
      CS << X86ATTInstPrinter::getRegisterName(DstOp.getReg()) << " = ";
      if (auto *CDS = dyn_cast<ConstantDataSequential>(C)) {
        int NumElements = CDS->getNumElements();
        if ((BitWidth % CstEltSize) == 0)
          NumElements = std::min<int>(NumElements, BitWidth / CstEltSize);
        CS << "[";
        for (int l = 0; l != NumLanes; ++l) {
          for (int i = 0; i < NumElements; ++i) {
            if (i != 0 || l != 0)
              CS << ",";
            if (CDS->getElementType()->isIntegerTy())
              printConstant(CDS->getElementAsAPInt(i), CS);
            else if (CDS->getElementType()->isHalfTy() ||
                     CDS->getElementType()->isFloatTy() ||
                     CDS->getElementType()->isDoubleTy())
              printConstant(CDS->getElementAsAPFloat(i), CS);
            else
              CS << "?";
          }
        }
        CS << "]";
        OutStreamer.AddComment(CS.str());
      } else if (auto *CV = dyn_cast<ConstantVector>(C)) {
        int NumOperands = CV->getNumOperands();
        if ((BitWidth % CstEltSize) == 0)
          NumOperands = std::min<int>(NumOperands, BitWidth / CstEltSize);
        CS << "<";
        for (int l = 0; l != NumLanes; ++l) {
          for (int i = 0; i < NumOperands; ++i) {
            if (i != 0 || l != 0)
              CS << ",";
            printConstant(CV->getOperand(i),
                          CV->getType()->getPrimitiveSizeInBits(), CS);
          }
        }
        CS << ">";
        OutStreamer.AddComment(CS.str());
      }
    }
    break;

  case X86::MOVDDUPrm:
  case X86::VMOVDDUPrm:
  case X86::VMOVDDUPZ128rm:
  case X86::VBROADCASTSSrm:
  case X86::VBROADCASTSSYrm:
  case X86::VBROADCASTSSZ128rm:
  case X86::VBROADCASTSSZ256rm:
  case X86::VBROADCASTSSZrm:
  case X86::VBROADCASTSDYrm:
  case X86::VBROADCASTSDZ256rm:
  case X86::VBROADCASTSDZrm:
  case X86::VPBROADCASTBrm:
  case X86::VPBROADCASTBYrm:
  case X86::VPBROADCASTBZ128rm:
  case X86::VPBROADCASTBZ256rm:
  case X86::VPBROADCASTBZrm:
  case X86::VPBROADCASTDrm:
  case X86::VPBROADCASTDYrm:
  case X86::VPBROADCASTDZ128rm:
  case X86::VPBROADCASTDZ256rm:
  case X86::VPBROADCASTDZrm:
  case X86::VPBROADCASTQrm:
  case X86::VPBROADCASTQYrm:
  case X86::VPBROADCASTQZ128rm:
  case X86::VPBROADCASTQZ256rm:
  case X86::VPBROADCASTQZrm:
  case X86::VPBROADCASTWrm:
  case X86::VPBROADCASTWYrm:
  case X86::VPBROADCASTWZ128rm:
  case X86::VPBROADCASTWZ256rm:
  case X86::VPBROADCASTWZrm:
    assert(MI->getNumOperands() >= (1 + X86::AddrNumOperands) &&
           "Unexpected number of operands!");
    if (auto *C =
            X86::getConstantFromPool(*MI, MI->getOperand(1 + X86::AddrDisp))) {
      int NumElts, EltBits;
      switch (MI->getOpcode()) {
      default: llvm_unreachable("Invalid opcode");
      case X86::MOVDDUPrm:          NumElts = 2;  EltBits = 64; break;
      case X86::VMOVDDUPrm:         NumElts = 2;  EltBits = 64; break;
      case X86::VMOVDDUPZ128rm:     NumElts = 2;  EltBits = 64; break;
      case X86::VBROADCASTSSrm:     NumElts = 4;  EltBits = 32; break;
      case X86::VBROADCASTSSYrm:    NumElts = 8;  EltBits = 32; break;
      case X86::VBROADCASTSSZ128rm: NumElts = 4;  EltBits = 32; break;
      case X86::VBROADCASTSSZ256rm: NumElts = 8;  EltBits = 32; break;
      case X86::VBROADCASTSSZrm:    NumElts = 16; EltBits = 32; break;
      case X86::VBROADCASTSDYrm:    NumElts = 4;  EltBits = 64; break;
      case X86::VBROADCASTSDZ256rm: NumElts = 4;  EltBits = 64; break;
      case X86::VBROADCASTSDZrm:    NumElts = 8;  EltBits = 64; break;
      case X86::VPBROADCASTBrm:     NumElts = 16; EltBits = 8; break;
      case X86::VPBROADCASTBYrm:    NumElts = 32; EltBits = 8; break;
      case X86::VPBROADCASTBZ128rm: NumElts = 16; EltBits = 8; break;
      case X86::VPBROADCASTBZ256rm: NumElts = 32; EltBits = 8; break;
      case X86::VPBROADCASTBZrm:    NumElts = 64; EltBits = 8; break;
      case X86::VPBROADCASTDrm:     NumElts = 4;  EltBits = 32; break;
      case X86::VPBROADCASTDYrm:    NumElts = 8;  EltBits = 32; break;
      case X86::VPBROADCASTDZ128rm: NumElts = 4;  EltBits = 32; break;
      case X86::VPBROADCASTDZ256rm: NumElts = 8;  EltBits = 32; break;
      case X86::VPBROADCASTDZrm:    NumElts = 16; EltBits = 32; break;
      case X86::VPBROADCASTQrm:     NumElts = 2;  EltBits = 64; break;
      case X86::VPBROADCASTQYrm:    NumElts = 4;  EltBits = 64; break;
      case X86::VPBROADCASTQZ128rm: NumElts = 2;  EltBits = 64; break;
      case X86::VPBROADCASTQZ256rm: NumElts = 4;  EltBits = 64; break;
      case X86::VPBROADCASTQZrm:    NumElts = 8;  EltBits = 64; break;
      case X86::VPBROADCASTWrm:     NumElts = 8;  EltBits = 16; break;
      case X86::VPBROADCASTWYrm:    NumElts = 16; EltBits = 16; break;
      case X86::VPBROADCASTWZ128rm: NumElts = 8;  EltBits = 16; break;
      case X86::VPBROADCASTWZ256rm: NumElts = 16; EltBits = 16; break;
      case X86::VPBROADCASTWZrm:    NumElts = 32; EltBits = 16; break;
      }

      std::string Comment;
      raw_string_ostream CS(Comment);
      const MachineOperand &DstOp = MI->getOperand(0);
      CS << X86ATTInstPrinter::getRegisterName(DstOp.getReg()) << " = ";
      CS << "[";
      for (int i = 0; i != NumElts; ++i) {
        if (i != 0)
          CS << ",";
        printConstant(C, EltBits, CS);
      }
      CS << "]";
      OutStreamer.AddComment(CS.str());
    }
  }
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
    [[fallthrough]];
  case X86::TAILJMPr:
  case X86::TAILJMPm:
  case X86::TAILJMPd:
  case X86::TAILJMPd_CC:
  case X86::TAILJMPr64:
  case X86::TAILJMPm64:
  case X86::TAILJMPd64_CC:
  case X86::TAILJMPr64_REX:
  case X86::TAILJMPm64_REX:
    // Lower these as normal, but add some comments.
    OutStreamer->AddComment("TAILCALL");
    break;

  case X86::TLS_addr32:
  case X86::TLS_addr64:
  case X86::TLS_addrX32:
  case X86::TLS_base_addr32:
  case X86::TLS_base_addr64:
  case X86::TLS_base_addrX32:
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
    EmitSEHInstruction(MI);
    return;

  case X86::SEH_Epilogue: {
    assert(MF->hasWinCFI() && "SEH_ instruction in function without WinCFI?");
    MachineBasicBlock::const_iterator MBBI(MI);
    // Check if preceded by a call and emit nop if so.
    for (MBBI = PrevCrossBBInst(MBBI);
         MBBI != MachineBasicBlock::const_iterator();
         MBBI = PrevCrossBBInst(MBBI)) {
      // Pseudo instructions that aren't a call are assumed to not emit any
      // code. If they do, we worst case generate unnecessary noops after a
      // call.
      if (MBBI->isCall() || !MBBI->isPseudo()) {
        if (MBBI->isCall())
          EmitAndCountInstruction(MCInstBuilder(X86::NOOP));
        break;
      }
    }
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
    break;
  }

  MCInst TmpInst;
  MCInstLowering.Lower(MI, TmpInst);

  // Stackmap shadows cannot include branch targets, so we can count the bytes
  // in a call towards the shadow, but must ensure that the no thread returns
  // in to the stackmap shadow.  The only way to achieve this is if the call
  // is at the end of the shadow.
  if (MI->isCall()) {
    // Count then size of the call towards the shadow
    SMShadowTracker.count(TmpInst, getSubtargetInfo(), CodeEmitter.get());
    // Then flush the shadow so that we fill with nops before the call, not
    // after it.
    SMShadowTracker.emitShadowPadding(*OutStreamer, getSubtargetInfo());
    // Then emit the call
    OutStreamer->emitInstruction(TmpInst, getSubtargetInfo());
    return;
  }

  EmitAndCountInstruction(TmpInst);
}
