//===-- X86MCInstLower.cpp - Convert X86 MachineInstr to an MCInst --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains code to lower X86 MachineInstrs to their corresponding
// MCInst records.
//
//===----------------------------------------------------------------------===//

#include "X86AsmPrinter.h"
#include "InstPrinter/X86ATTInstPrinter.h"
#include "X86COFFMachineModuleInfo.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/CodeGen/StackMaps.h"
#include "llvm/IR/Type.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Target/Mangler.h"
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

  void Lower(const MachineInstr *MI, MCInst &OutMI) const;

  MCSymbol *GetSymbolFromOperand(const MachineOperand &MO) const;
  MCOperand LowerSymbolOperand(const MachineOperand &MO, MCSymbol *Sym) const;

private:
  MachineModuleInfoMachO &getMachOMMI() const;
  Mangler *getMang() const {
    return AsmPrinter.Mang;
  }
};

} // end anonymous namespace

X86MCInstLower::X86MCInstLower(const MachineFunction &mf,
                               X86AsmPrinter &asmprinter)
: Ctx(mf.getContext()), MF(mf), TM(mf.getTarget()),
  MAI(*TM.getMCAsmInfo()), AsmPrinter(asmprinter) {}

MachineModuleInfoMachO &X86MCInstLower::getMachOMMI() const {
  return MF.getMMI().getObjFileInfo<MachineModuleInfoMachO>();
}


/// GetSymbolFromOperand - Lower an MO_GlobalAddress or MO_ExternalSymbol
/// operand to an MCSymbol.
MCSymbol *X86MCInstLower::
GetSymbolFromOperand(const MachineOperand &MO) const {
  assert((MO.isGlobal() || MO.isSymbol() || MO.isMBB()) && "Isn't a symbol reference");

  SmallString<128> Name;

  if (MO.isGlobal()) {
    const GlobalValue *GV = MO.getGlobal();
    bool isImplicitlyPrivate = false;
    if (MO.getTargetFlags() == X86II::MO_DARWIN_STUB ||
        MO.getTargetFlags() == X86II::MO_DARWIN_NONLAZY ||
        MO.getTargetFlags() == X86II::MO_DARWIN_NONLAZY_PIC_BASE ||
        MO.getTargetFlags() == X86II::MO_DARWIN_HIDDEN_NONLAZY_PIC_BASE)
      isImplicitlyPrivate = true;

    getMang()->getNameWithPrefix(Name, GV, isImplicitlyPrivate);
  } else if (MO.isSymbol()) {
    Name += MAI.getGlobalPrefix();
    Name += MO.getSymbolName();
  } else if (MO.isMBB()) {
    Name += MO.getMBB()->getSymbol()->getName();
  }

  // If the target flags on the operand changes the name of the symbol, do that
  // before we return the symbol.
  switch (MO.getTargetFlags()) {
  default: break;
  case X86II::MO_DLLIMPORT: {
    // Handle dllimport linkage.
    const char *Prefix = "__imp_";
    Name.insert(Name.begin(), Prefix, Prefix+strlen(Prefix));
    break;
  }
  case X86II::MO_DARWIN_NONLAZY:
  case X86II::MO_DARWIN_NONLAZY_PIC_BASE: {
    Name += "$non_lazy_ptr";
    MCSymbol *Sym = Ctx.GetOrCreateSymbol(Name.str());

    MachineModuleInfoImpl::StubValueTy &StubSym =
      getMachOMMI().getGVStubEntry(Sym);
    if (StubSym.getPointer() == 0) {
      assert(MO.isGlobal() && "Extern symbol not handled yet");
      StubSym =
        MachineModuleInfoImpl::
        StubValueTy(AsmPrinter.getSymbol(MO.getGlobal()),
                    !MO.getGlobal()->hasInternalLinkage());
    }
    return Sym;
  }
  case X86II::MO_DARWIN_HIDDEN_NONLAZY_PIC_BASE: {
    Name += "$non_lazy_ptr";
    MCSymbol *Sym = Ctx.GetOrCreateSymbol(Name.str());
    MachineModuleInfoImpl::StubValueTy &StubSym =
      getMachOMMI().getHiddenGVStubEntry(Sym);
    if (StubSym.getPointer() == 0) {
      assert(MO.isGlobal() && "Extern symbol not handled yet");
      StubSym =
        MachineModuleInfoImpl::
        StubValueTy(AsmPrinter.getSymbol(MO.getGlobal()),
                    !MO.getGlobal()->hasInternalLinkage());
    }
    return Sym;
  }
  case X86II::MO_DARWIN_STUB: {
    Name += "$stub";
    MCSymbol *Sym = Ctx.GetOrCreateSymbol(Name.str());
    MachineModuleInfoImpl::StubValueTy &StubSym =
      getMachOMMI().getFnStubEntry(Sym);
    if (StubSym.getPointer())
      return Sym;

    if (MO.isGlobal()) {
      StubSym =
        MachineModuleInfoImpl::
        StubValueTy(AsmPrinter.getSymbol(MO.getGlobal()),
                    !MO.getGlobal()->hasInternalLinkage());
    } else {
      Name.erase(Name.end()-5, Name.end());
      StubSym =
        MachineModuleInfoImpl::
        StubValueTy(Ctx.GetOrCreateSymbol(Name.str()), false);
    }
    return Sym;
  }
  }

  return Ctx.GetOrCreateSymbol(Name.str());
}

MCOperand X86MCInstLower::LowerSymbolOperand(const MachineOperand &MO,
                                             MCSymbol *Sym) const {
  // FIXME: We would like an efficient form for this, so we don't have to do a
  // lot of extra uniquing.
  const MCExpr *Expr = 0;
  MCSymbolRefExpr::VariantKind RefKind = MCSymbolRefExpr::VK_None;

  switch (MO.getTargetFlags()) {
  default: llvm_unreachable("Unknown target flag on GV operand");
  case X86II::MO_NO_FLAG:    // No flag.
  // These affect the name of the symbol, not any suffix.
  case X86II::MO_DARWIN_NONLAZY:
  case X86II::MO_DLLIMPORT:
  case X86II::MO_DARWIN_STUB:
    break;

  case X86II::MO_TLVP:      RefKind = MCSymbolRefExpr::VK_TLVP; break;
  case X86II::MO_TLVP_PIC_BASE:
    Expr = MCSymbolRefExpr::Create(Sym, MCSymbolRefExpr::VK_TLVP, Ctx);
    // Subtract the pic base.
    Expr = MCBinaryExpr::CreateSub(Expr,
                                  MCSymbolRefExpr::Create(MF.getPICBaseSymbol(),
                                                           Ctx),
                                   Ctx);
    break;
  case X86II::MO_SECREL:    RefKind = MCSymbolRefExpr::VK_SECREL; break;
  case X86II::MO_TLSGD:     RefKind = MCSymbolRefExpr::VK_TLSGD; break;
  case X86II::MO_TLSLD:     RefKind = MCSymbolRefExpr::VK_TLSLD; break;
  case X86II::MO_TLSLDM:    RefKind = MCSymbolRefExpr::VK_TLSLDM; break;
  case X86II::MO_GOTTPOFF:  RefKind = MCSymbolRefExpr::VK_GOTTPOFF; break;
  case X86II::MO_INDNTPOFF: RefKind = MCSymbolRefExpr::VK_INDNTPOFF; break;
  case X86II::MO_TPOFF:     RefKind = MCSymbolRefExpr::VK_TPOFF; break;
  case X86II::MO_DTPOFF:    RefKind = MCSymbolRefExpr::VK_DTPOFF; break;
  case X86II::MO_NTPOFF:    RefKind = MCSymbolRefExpr::VK_NTPOFF; break;
  case X86II::MO_GOTNTPOFF: RefKind = MCSymbolRefExpr::VK_GOTNTPOFF; break;
  case X86II::MO_GOTPCREL:  RefKind = MCSymbolRefExpr::VK_GOTPCREL; break;
  case X86II::MO_GOT:       RefKind = MCSymbolRefExpr::VK_GOT; break;
  case X86II::MO_GOTOFF:    RefKind = MCSymbolRefExpr::VK_GOTOFF; break;
  case X86II::MO_PLT:       RefKind = MCSymbolRefExpr::VK_PLT; break;
  case X86II::MO_PIC_BASE_OFFSET:
  case X86II::MO_DARWIN_NONLAZY_PIC_BASE:
  case X86II::MO_DARWIN_HIDDEN_NONLAZY_PIC_BASE:
    Expr = MCSymbolRefExpr::Create(Sym, Ctx);
    // Subtract the pic base.
    Expr = MCBinaryExpr::CreateSub(Expr,
                            MCSymbolRefExpr::Create(MF.getPICBaseSymbol(), Ctx),
                                   Ctx);
    if (MO.isJTI() && MAI.hasSetDirective()) {
      // If .set directive is supported, use it to reduce the number of
      // relocations the assembler will generate for differences between
      // local labels. This is only safe when the symbols are in the same
      // section so we are restricting it to jumptable references.
      MCSymbol *Label = Ctx.CreateTempSymbol();
      AsmPrinter.OutStreamer.EmitAssignment(Label, Expr);
      Expr = MCSymbolRefExpr::Create(Label, Ctx);
    }
    break;
  }

  if (Expr == 0)
    Expr = MCSymbolRefExpr::Create(Sym, RefKind, Ctx);

  if (!MO.isJTI() && !MO.isMBB() && MO.getOffset())
    Expr = MCBinaryExpr::CreateAdd(Expr,
                                   MCConstantExpr::Create(MO.getOffset(), Ctx),
                                   Ctx);
  return MCOperand::CreateExpr(Expr);
}


/// LowerUnaryToTwoAddr - R = setb   -> R = sbb R, R
static void LowerUnaryToTwoAddr(MCInst &OutMI, unsigned NewOpc) {
  OutMI.setOpcode(NewOpc);
  OutMI.addOperand(OutMI.getOperand(0));
  OutMI.addOperand(OutMI.getOperand(0));
}

/// \brief Simplify FOO $imm, %{al,ax,eax,rax} to FOO $imm, for instruction with
/// a short fixed-register form.
static void SimplifyShortImmForm(MCInst &Inst, unsigned Opcode) {
  unsigned ImmOp = Inst.getNumOperands() - 1;
  assert(Inst.getOperand(0).isReg() &&
         (Inst.getOperand(ImmOp).isImm() || Inst.getOperand(ImmOp).isExpr()) &&
         ((Inst.getNumOperands() == 3 && Inst.getOperand(1).isReg() &&
           Inst.getOperand(0).getReg() == Inst.getOperand(1).getReg()) ||
          Inst.getNumOperands() == 2) && "Unexpected instruction!");

  // Check whether the destination register can be fixed.
  unsigned Reg = Inst.getOperand(0).getReg();
  if (Reg != X86::AL && Reg != X86::AX && Reg != X86::EAX && Reg != X86::RAX)
    return;

  // If so, rewrite the instruction.
  MCOperand Saved = Inst.getOperand(ImmOp);
  Inst = MCInst();
  Inst.setOpcode(Opcode);
  Inst.addOperand(Saved);
}

/// \brief If a movsx instruction has a shorter encoding for the used register
/// simplify the instruction to use it instead.
static void SimplifyMOVSX(MCInst &Inst) {
  unsigned NewOpcode = 0;
  unsigned Op0 = Inst.getOperand(0).getReg(), Op1 = Inst.getOperand(1).getReg();
  switch (Inst.getOpcode()) {
  default:
    llvm_unreachable("Unexpected instruction!");
  case X86::MOVSX16rr8:  // movsbw %al, %ax   --> cbtw
    if (Op0 == X86::AX && Op1 == X86::AL)
      NewOpcode = X86::CBW;
    break;
  case X86::MOVSX32rr16: // movswl %ax, %eax  --> cwtl
    if (Op0 == X86::EAX && Op1 == X86::AX)
      NewOpcode = X86::CWDE;
    break;
  case X86::MOVSX64rr32: // movslq %eax, %rax --> cltq
    if (Op0 == X86::RAX && Op1 == X86::EAX)
      NewOpcode = X86::CDQE;
    break;
  }

  if (NewOpcode != 0) {
    Inst = MCInst();
    Inst.setOpcode(NewOpcode);
  }
}

/// \brief Simplify things like MOV32rm to MOV32o32a.
static void SimplifyShortMoveForm(X86AsmPrinter &Printer, MCInst &Inst,
                                  unsigned Opcode) {
  // Don't make these simplifications in 64-bit mode; other assemblers don't
  // perform them because they make the code larger.
  if (Printer.getSubtarget().is64Bit())
    return;

  bool IsStore = Inst.getOperand(0).isReg() && Inst.getOperand(1).isReg();
  unsigned AddrBase = IsStore;
  unsigned RegOp = IsStore ? 0 : 5;
  unsigned AddrOp = AddrBase + 3;
  assert(Inst.getNumOperands() == 6 && Inst.getOperand(RegOp).isReg() &&
         Inst.getOperand(AddrBase + 0).isReg() && // base
         Inst.getOperand(AddrBase + 1).isImm() && // scale
         Inst.getOperand(AddrBase + 2).isReg() && // index register
         (Inst.getOperand(AddrOp).isExpr() ||     // address
          Inst.getOperand(AddrOp).isImm())&&
         Inst.getOperand(AddrBase + 4).isReg() && // segment
         "Unexpected instruction!");

  // Check whether the destination register can be fixed.
  unsigned Reg = Inst.getOperand(RegOp).getReg();
  if (Reg != X86::AL && Reg != X86::AX && Reg != X86::EAX && Reg != X86::RAX)
    return;

  // Check whether this is an absolute address.
  // FIXME: We know TLVP symbol refs aren't, but there should be a better way
  // to do this here.
  bool Absolute = true;
  if (Inst.getOperand(AddrOp).isExpr()) {
    const MCExpr *MCE = Inst.getOperand(AddrOp).getExpr();
    if (const MCSymbolRefExpr *SRE = dyn_cast<MCSymbolRefExpr>(MCE))
      if (SRE->getKind() == MCSymbolRefExpr::VK_TLVP)
        Absolute = false;
  }

  if (Absolute &&
      (Inst.getOperand(AddrBase + 0).getReg() != 0 ||
       Inst.getOperand(AddrBase + 2).getReg() != 0 ||
       Inst.getOperand(AddrBase + 4).getReg() != 0 ||
       Inst.getOperand(AddrBase + 1).getImm() != 1))
    return;

  // If so, rewrite the instruction.
  MCOperand Saved = Inst.getOperand(AddrOp);
  Inst = MCInst();
  Inst.setOpcode(Opcode);
  Inst.addOperand(Saved);
}

void X86MCInstLower::Lower(const MachineInstr *MI, MCInst &OutMI) const {
  OutMI.setOpcode(MI->getOpcode());

  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);

    MCOperand MCOp;
    switch (MO.getType()) {
    default:
      MI->dump();
      llvm_unreachable("unknown operand type");
    case MachineOperand::MO_Register:
      // Ignore all implicit register operands.
      if (MO.isImplicit()) continue;
      MCOp = MCOperand::CreateReg(MO.getReg());
      break;
    case MachineOperand::MO_Immediate:
      MCOp = MCOperand::CreateImm(MO.getImm());
      break;
    case MachineOperand::MO_MachineBasicBlock:
    case MachineOperand::MO_GlobalAddress:
    case MachineOperand::MO_ExternalSymbol:
      MCOp = LowerSymbolOperand(MO, GetSymbolFromOperand(MO));
      break;
    case MachineOperand::MO_JumpTableIndex:
      MCOp = LowerSymbolOperand(MO, AsmPrinter.GetJTISymbol(MO.getIndex()));
      break;
    case MachineOperand::MO_ConstantPoolIndex:
      MCOp = LowerSymbolOperand(MO, AsmPrinter.GetCPISymbol(MO.getIndex()));
      break;
    case MachineOperand::MO_BlockAddress:
      MCOp = LowerSymbolOperand(MO,
                     AsmPrinter.GetBlockAddressSymbol(MO.getBlockAddress()));
      break;
    case MachineOperand::MO_RegisterMask:
      // Ignore call clobbers.
      continue;
    }

    OutMI.addOperand(MCOp);
  }

  // Handle a few special cases to eliminate operand modifiers.
ReSimplify:
  switch (OutMI.getOpcode()) {
  case X86::LEA64_32r:
  case X86::LEA64r:
  case X86::LEA16r:
  case X86::LEA32r:
    // LEA should have a segment register, but it must be empty.
    assert(OutMI.getNumOperands() == 1+X86::AddrNumOperands &&
           "Unexpected # of LEA operands");
    assert(OutMI.getOperand(1+X86::AddrSegmentReg).getReg() == 0 &&
           "LEA has segment specified!");
    break;
  case X86::MOV32r0:      LowerUnaryToTwoAddr(OutMI, X86::XOR32rr); break;

  case X86::MOV32ri64:
    OutMI.setOpcode(X86::MOV32ri);
    break;

  // Commute operands to get a smaller encoding by using VEX.R instead of VEX.B
  // if one of the registers is extended, but other isn't.
  case X86::VMOVAPDrr:
  case X86::VMOVAPDYrr:
  case X86::VMOVAPSrr:
  case X86::VMOVAPSYrr:
  case X86::VMOVDQArr:
  case X86::VMOVDQAYrr:
  case X86::VMOVDQUrr:
  case X86::VMOVDQUYrr:
  case X86::VMOVUPDrr:
  case X86::VMOVUPDYrr:
  case X86::VMOVUPSrr:
  case X86::VMOVUPSYrr: {
    if (!X86II::isX86_64ExtendedReg(OutMI.getOperand(0).getReg()) &&
        X86II::isX86_64ExtendedReg(OutMI.getOperand(1).getReg())) {
      unsigned NewOpc;
      switch (OutMI.getOpcode()) {
      default: llvm_unreachable("Invalid opcode");
      case X86::VMOVAPDrr:  NewOpc = X86::VMOVAPDrr_REV;  break;
      case X86::VMOVAPDYrr: NewOpc = X86::VMOVAPDYrr_REV; break;
      case X86::VMOVAPSrr:  NewOpc = X86::VMOVAPSrr_REV;  break;
      case X86::VMOVAPSYrr: NewOpc = X86::VMOVAPSYrr_REV; break;
      case X86::VMOVDQArr:  NewOpc = X86::VMOVDQArr_REV;  break;
      case X86::VMOVDQAYrr: NewOpc = X86::VMOVDQAYrr_REV; break;
      case X86::VMOVDQUrr:  NewOpc = X86::VMOVDQUrr_REV;  break;
      case X86::VMOVDQUYrr: NewOpc = X86::VMOVDQUYrr_REV; break;
      case X86::VMOVUPDrr:  NewOpc = X86::VMOVUPDrr_REV;  break;
      case X86::VMOVUPDYrr: NewOpc = X86::VMOVUPDYrr_REV; break;
      case X86::VMOVUPSrr:  NewOpc = X86::VMOVUPSrr_REV;  break;
      case X86::VMOVUPSYrr: NewOpc = X86::VMOVUPSYrr_REV; break;
      }
      OutMI.setOpcode(NewOpc);
    }
    break;
  }
  case X86::VMOVSDrr:
  case X86::VMOVSSrr: {
    if (!X86II::isX86_64ExtendedReg(OutMI.getOperand(0).getReg()) &&
        X86II::isX86_64ExtendedReg(OutMI.getOperand(2).getReg())) {
      unsigned NewOpc;
      switch (OutMI.getOpcode()) {
      default: llvm_unreachable("Invalid opcode");
      case X86::VMOVSDrr:   NewOpc = X86::VMOVSDrr_REV;   break;
      case X86::VMOVSSrr:   NewOpc = X86::VMOVSSrr_REV;   break;
      }
      OutMI.setOpcode(NewOpc);
    }
    break;
  }

  // TAILJMPr64, CALL64r, CALL64pcrel32 - These instructions have register
  // inputs modeled as normal uses instead of implicit uses.  As such, truncate
  // off all but the first operand (the callee).  FIXME: Change isel.
  case X86::TAILJMPr64:
  case X86::CALL64r:
  case X86::CALL64pcrel32: {
    unsigned Opcode = OutMI.getOpcode();
    MCOperand Saved = OutMI.getOperand(0);
    OutMI = MCInst();
    OutMI.setOpcode(Opcode);
    OutMI.addOperand(Saved);
    break;
  }

  case X86::EH_RETURN:
  case X86::EH_RETURN64: {
    OutMI = MCInst();
    OutMI.setOpcode(X86::RET);
    break;
  }

  // TAILJMPd, TAILJMPd64 - Lower to the correct jump instructions.
  case X86::TAILJMPr:
  case X86::TAILJMPd:
  case X86::TAILJMPd64: {
    unsigned Opcode;
    switch (OutMI.getOpcode()) {
    default: llvm_unreachable("Invalid opcode");
    case X86::TAILJMPr: Opcode = X86::JMP32r; break;
    case X86::TAILJMPd:
    case X86::TAILJMPd64: Opcode = X86::JMP_1; break;
    }

    MCOperand Saved = OutMI.getOperand(0);
    OutMI = MCInst();
    OutMI.setOpcode(Opcode);
    OutMI.addOperand(Saved);
    break;
  }

  // These are pseudo-ops for OR to help with the OR->ADD transformation.  We do
  // this with an ugly goto in case the resultant OR uses EAX and needs the
  // short form.
  case X86::ADD16rr_DB:   OutMI.setOpcode(X86::OR16rr); goto ReSimplify;
  case X86::ADD32rr_DB:   OutMI.setOpcode(X86::OR32rr); goto ReSimplify;
  case X86::ADD64rr_DB:   OutMI.setOpcode(X86::OR64rr); goto ReSimplify;
  case X86::ADD16ri_DB:   OutMI.setOpcode(X86::OR16ri); goto ReSimplify;
  case X86::ADD32ri_DB:   OutMI.setOpcode(X86::OR32ri); goto ReSimplify;
  case X86::ADD64ri32_DB: OutMI.setOpcode(X86::OR64ri32); goto ReSimplify;
  case X86::ADD16ri8_DB:  OutMI.setOpcode(X86::OR16ri8); goto ReSimplify;
  case X86::ADD32ri8_DB:  OutMI.setOpcode(X86::OR32ri8); goto ReSimplify;
  case X86::ADD64ri8_DB:  OutMI.setOpcode(X86::OR64ri8); goto ReSimplify;

  // The assembler backend wants to see branches in their small form and relax
  // them to their large form.  The JIT can only handle the large form because
  // it does not do relaxation.  For now, translate the large form to the
  // small one here.
  case X86::JMP_4: OutMI.setOpcode(X86::JMP_1); break;
  case X86::JO_4:  OutMI.setOpcode(X86::JO_1); break;
  case X86::JNO_4: OutMI.setOpcode(X86::JNO_1); break;
  case X86::JB_4:  OutMI.setOpcode(X86::JB_1); break;
  case X86::JAE_4: OutMI.setOpcode(X86::JAE_1); break;
  case X86::JE_4:  OutMI.setOpcode(X86::JE_1); break;
  case X86::JNE_4: OutMI.setOpcode(X86::JNE_1); break;
  case X86::JBE_4: OutMI.setOpcode(X86::JBE_1); break;
  case X86::JA_4:  OutMI.setOpcode(X86::JA_1); break;
  case X86::JS_4:  OutMI.setOpcode(X86::JS_1); break;
  case X86::JNS_4: OutMI.setOpcode(X86::JNS_1); break;
  case X86::JP_4:  OutMI.setOpcode(X86::JP_1); break;
  case X86::JNP_4: OutMI.setOpcode(X86::JNP_1); break;
  case X86::JL_4:  OutMI.setOpcode(X86::JL_1); break;
  case X86::JGE_4: OutMI.setOpcode(X86::JGE_1); break;
  case X86::JLE_4: OutMI.setOpcode(X86::JLE_1); break;
  case X86::JG_4:  OutMI.setOpcode(X86::JG_1); break;

  // Atomic load and store require a separate pseudo-inst because Acquire
  // implies mayStore and Release implies mayLoad; fix these to regular MOV
  // instructions here
  case X86::ACQUIRE_MOV8rm:  OutMI.setOpcode(X86::MOV8rm); goto ReSimplify;
  case X86::ACQUIRE_MOV16rm: OutMI.setOpcode(X86::MOV16rm); goto ReSimplify;
  case X86::ACQUIRE_MOV32rm: OutMI.setOpcode(X86::MOV32rm); goto ReSimplify;
  case X86::ACQUIRE_MOV64rm: OutMI.setOpcode(X86::MOV64rm); goto ReSimplify;
  case X86::RELEASE_MOV8mr:  OutMI.setOpcode(X86::MOV8mr); goto ReSimplify;
  case X86::RELEASE_MOV16mr: OutMI.setOpcode(X86::MOV16mr); goto ReSimplify;
  case X86::RELEASE_MOV32mr: OutMI.setOpcode(X86::MOV32mr); goto ReSimplify;
  case X86::RELEASE_MOV64mr: OutMI.setOpcode(X86::MOV64mr); goto ReSimplify;

  // We don't currently select the correct instruction form for instructions
  // which have a short %eax, etc. form. Handle this by custom lowering, for
  // now.
  //
  // Note, we are currently not handling the following instructions:
  // MOV64ao8, MOV64o8a
  // XCHG16ar, XCHG32ar, XCHG64ar
  case X86::MOV8mr_NOREX:
  case X86::MOV8mr:     SimplifyShortMoveForm(AsmPrinter, OutMI, X86::MOV8ao8); break;
  case X86::MOV8rm_NOREX:
  case X86::MOV8rm:     SimplifyShortMoveForm(AsmPrinter, OutMI, X86::MOV8o8a); break;
  case X86::MOV16mr:    SimplifyShortMoveForm(AsmPrinter, OutMI, X86::MOV16ao16); break;
  case X86::MOV16rm:    SimplifyShortMoveForm(AsmPrinter, OutMI, X86::MOV16o16a); break;
  case X86::MOV32mr:    SimplifyShortMoveForm(AsmPrinter, OutMI, X86::MOV32ao32); break;
  case X86::MOV32rm:    SimplifyShortMoveForm(AsmPrinter, OutMI, X86::MOV32o32a); break;

  case X86::ADC8ri:     SimplifyShortImmForm(OutMI, X86::ADC8i8);    break;
  case X86::ADC16ri:    SimplifyShortImmForm(OutMI, X86::ADC16i16);  break;
  case X86::ADC32ri:    SimplifyShortImmForm(OutMI, X86::ADC32i32);  break;
  case X86::ADC64ri32:  SimplifyShortImmForm(OutMI, X86::ADC64i32);  break;
  case X86::ADD8ri:     SimplifyShortImmForm(OutMI, X86::ADD8i8);    break;
  case X86::ADD16ri:    SimplifyShortImmForm(OutMI, X86::ADD16i16);  break;
  case X86::ADD32ri:    SimplifyShortImmForm(OutMI, X86::ADD32i32);  break;
  case X86::ADD64ri32:  SimplifyShortImmForm(OutMI, X86::ADD64i32);  break;
  case X86::AND8ri:     SimplifyShortImmForm(OutMI, X86::AND8i8);    break;
  case X86::AND16ri:    SimplifyShortImmForm(OutMI, X86::AND16i16);  break;
  case X86::AND32ri:    SimplifyShortImmForm(OutMI, X86::AND32i32);  break;
  case X86::AND64ri32:  SimplifyShortImmForm(OutMI, X86::AND64i32);  break;
  case X86::CMP8ri:     SimplifyShortImmForm(OutMI, X86::CMP8i8);    break;
  case X86::CMP16ri:    SimplifyShortImmForm(OutMI, X86::CMP16i16);  break;
  case X86::CMP32ri:    SimplifyShortImmForm(OutMI, X86::CMP32i32);  break;
  case X86::CMP64ri32:  SimplifyShortImmForm(OutMI, X86::CMP64i32);  break;
  case X86::OR8ri:      SimplifyShortImmForm(OutMI, X86::OR8i8);     break;
  case X86::OR16ri:     SimplifyShortImmForm(OutMI, X86::OR16i16);   break;
  case X86::OR32ri:     SimplifyShortImmForm(OutMI, X86::OR32i32);   break;
  case X86::OR64ri32:   SimplifyShortImmForm(OutMI, X86::OR64i32);   break;
  case X86::SBB8ri:     SimplifyShortImmForm(OutMI, X86::SBB8i8);    break;
  case X86::SBB16ri:    SimplifyShortImmForm(OutMI, X86::SBB16i16);  break;
  case X86::SBB32ri:    SimplifyShortImmForm(OutMI, X86::SBB32i32);  break;
  case X86::SBB64ri32:  SimplifyShortImmForm(OutMI, X86::SBB64i32);  break;
  case X86::SUB8ri:     SimplifyShortImmForm(OutMI, X86::SUB8i8);    break;
  case X86::SUB16ri:    SimplifyShortImmForm(OutMI, X86::SUB16i16);  break;
  case X86::SUB32ri:    SimplifyShortImmForm(OutMI, X86::SUB32i32);  break;
  case X86::SUB64ri32:  SimplifyShortImmForm(OutMI, X86::SUB64i32);  break;
  case X86::TEST8ri:    SimplifyShortImmForm(OutMI, X86::TEST8i8);   break;
  case X86::TEST16ri:   SimplifyShortImmForm(OutMI, X86::TEST16i16); break;
  case X86::TEST32ri:   SimplifyShortImmForm(OutMI, X86::TEST32i32); break;
  case X86::TEST64ri32: SimplifyShortImmForm(OutMI, X86::TEST64i32); break;
  case X86::XOR8ri:     SimplifyShortImmForm(OutMI, X86::XOR8i8);    break;
  case X86::XOR16ri:    SimplifyShortImmForm(OutMI, X86::XOR16i16);  break;
  case X86::XOR32ri:    SimplifyShortImmForm(OutMI, X86::XOR32i32);  break;
  case X86::XOR64ri32:  SimplifyShortImmForm(OutMI, X86::XOR64i32);  break;

  // Try to shrink some forms of movsx.
  case X86::MOVSX16rr8:
  case X86::MOVSX32rr16:
  case X86::MOVSX64rr32:
    SimplifyMOVSX(OutMI);
    break;
  }
}

static void LowerTlsAddr(MCStreamer &OutStreamer,
                         X86MCInstLower &MCInstLowering,
                         const MachineInstr &MI) {

  bool is64Bits = MI.getOpcode() == X86::TLS_addr64 ||
                  MI.getOpcode() == X86::TLS_base_addr64;

  bool needsPadding = MI.getOpcode() == X86::TLS_addr64;

  MCContext &context = OutStreamer.getContext();

  if (needsPadding)
    OutStreamer.EmitInstruction(MCInstBuilder(X86::DATA16_PREFIX));

  MCSymbolRefExpr::VariantKind SRVK;
  switch (MI.getOpcode()) {
    case X86::TLS_addr32:
    case X86::TLS_addr64:
      SRVK = MCSymbolRefExpr::VK_TLSGD;
      break;
    case X86::TLS_base_addr32:
      SRVK = MCSymbolRefExpr::VK_TLSLDM;
      break;
    case X86::TLS_base_addr64:
      SRVK = MCSymbolRefExpr::VK_TLSLD;
      break;
    default:
      llvm_unreachable("unexpected opcode");
  }

  MCSymbol *sym = MCInstLowering.GetSymbolFromOperand(MI.getOperand(3));
  const MCSymbolRefExpr *symRef = MCSymbolRefExpr::Create(sym, SRVK, context);

  MCInst LEA;
  if (is64Bits) {
    LEA.setOpcode(X86::LEA64r);
    LEA.addOperand(MCOperand::CreateReg(X86::RDI)); // dest
    LEA.addOperand(MCOperand::CreateReg(X86::RIP)); // base
    LEA.addOperand(MCOperand::CreateImm(1));        // scale
    LEA.addOperand(MCOperand::CreateReg(0));        // index
    LEA.addOperand(MCOperand::CreateExpr(symRef));  // disp
    LEA.addOperand(MCOperand::CreateReg(0));        // seg
  } else if (SRVK == MCSymbolRefExpr::VK_TLSLDM) {
    LEA.setOpcode(X86::LEA32r);
    LEA.addOperand(MCOperand::CreateReg(X86::EAX)); // dest
    LEA.addOperand(MCOperand::CreateReg(X86::EBX)); // base
    LEA.addOperand(MCOperand::CreateImm(1));        // scale
    LEA.addOperand(MCOperand::CreateReg(0));        // index
    LEA.addOperand(MCOperand::CreateExpr(symRef));  // disp
    LEA.addOperand(MCOperand::CreateReg(0));        // seg
  } else {
    LEA.setOpcode(X86::LEA32r);
    LEA.addOperand(MCOperand::CreateReg(X86::EAX)); // dest
    LEA.addOperand(MCOperand::CreateReg(0));        // base
    LEA.addOperand(MCOperand::CreateImm(1));        // scale
    LEA.addOperand(MCOperand::CreateReg(X86::EBX)); // index
    LEA.addOperand(MCOperand::CreateExpr(symRef));  // disp
    LEA.addOperand(MCOperand::CreateReg(0));        // seg
  }
  OutStreamer.EmitInstruction(LEA);

  if (needsPadding) {
    OutStreamer.EmitInstruction(MCInstBuilder(X86::DATA16_PREFIX));
    OutStreamer.EmitInstruction(MCInstBuilder(X86::DATA16_PREFIX));
    OutStreamer.EmitInstruction(MCInstBuilder(X86::REX64_PREFIX));
  }

  StringRef name = is64Bits ? "__tls_get_addr" : "___tls_get_addr";
  MCSymbol *tlsGetAddr = context.GetOrCreateSymbol(name);
  const MCSymbolRefExpr *tlsRef =
    MCSymbolRefExpr::Create(tlsGetAddr,
                            MCSymbolRefExpr::VK_PLT,
                            context);

  OutStreamer.EmitInstruction(MCInstBuilder(is64Bits ? X86::CALL64pcrel32
                                                     : X86::CALLpcrel32)
    .addExpr(tlsRef));
}

static std::pair<StackMaps::Location, MachineInstr::const_mop_iterator>
parseMemoryOperand(StackMaps::Location::LocationType LocTy, unsigned Size,
                   MachineInstr::const_mop_iterator MOI,
                   MachineInstr::const_mop_iterator MOE) {

  typedef StackMaps::Location Location;

  assert(std::distance(MOI, MOE) >= 5 && "Too few operands to encode mem op.");

  const MachineOperand &Base = *MOI;
  const MachineOperand &Scale = *(++MOI);
  const MachineOperand &Index = *(++MOI);
  const MachineOperand &Disp = *(++MOI);
  const MachineOperand &ZeroReg = *(++MOI);

  // Sanity check for supported operand format.
  assert(Base.isReg() &&
         Scale.isImm() && Scale.getImm() == 1 &&
         Index.isReg() && Index.getReg() == 0 &&
         Disp.isImm() && ZeroReg.isReg() && (ZeroReg.getReg() == 0) &&
         "Unsupported x86 memory operand sequence.");
  (void)Scale;
  (void)Index;
  (void)ZeroReg;

  return std::make_pair(
    Location(LocTy, Size, Base.getReg(), Disp.getImm()), ++MOI);
}

std::pair<StackMaps::Location, MachineInstr::const_mop_iterator>
X86AsmPrinter::stackmapOperandParser(MachineInstr::const_mop_iterator MOI,
                                     MachineInstr::const_mop_iterator MOE,
                                     const TargetMachine &TM) {

  typedef StackMaps::Location Location;

  const MachineOperand &MOP = *MOI;
  assert(!MOP.isRegMask() && (!MOP.isReg() || !MOP.isImplicit()) &&
         "Register mask and implicit operands should not be processed.");

  if (MOP.isImm()) {
    // Verify anyregcc
    // [<def>], <id>, <numBytes>, <target>, <numArgs>, <cc>, ...

    switch (MOP.getImm()) {
    default: llvm_unreachable("Unrecognized operand type.");
    case StackMaps::DirectMemRefOp: {
      unsigned Size = TM.getDataLayout()->getPointerSizeInBits();
      assert((Size % 8) == 0 && "Need pointer size in bytes.");
      Size /= 8;
      return parseMemoryOperand(StackMaps::Location::Direct, Size,
                                llvm::next(MOI), MOE);
    }
    case StackMaps::IndirectMemRefOp: {
      ++MOI;
      int64_t Size = MOI->getImm();
      assert(Size > 0 && "Need a valid size for indirect memory locations.");
      return parseMemoryOperand(StackMaps::Location::Indirect, Size,
                                llvm::next(MOI), MOE);
    }
    case StackMaps::ConstantOp: {
      ++MOI;
      assert(MOI->isImm() && "Expected constant operand.");
      int64_t Imm = MOI->getImm();
      return std::make_pair(
        Location(Location::Constant, sizeof(int64_t), 0, Imm), ++MOI);
    }
    }
  }

  // Otherwise this is a reg operand. The physical register number will
  // ultimately be encoded as a DWARF regno. The stack map also records the size
  // of a spill slot that can hold the register content. (The runtime can
  // track the actual size of the data type if it needs to.)
  assert(MOP.isReg() && "Expected register operand here.");
  assert(TargetRegisterInfo::isPhysicalRegister(MOP.getReg()) &&
         "Virtreg operands should have been rewritten before now.");
  const TargetRegisterClass *RC =
    TM.getRegisterInfo()->getMinimalPhysRegClass(MOP.getReg());
  assert(!MOP.getSubReg() && "Physical subreg still around.");
  return std::make_pair(
    Location(Location::Register, RC->getSize(), MOP.getReg(), 0), ++MOI);
}

// Lower a stackmap of the form:
// <id>, <shadowBytes>, ...
static void LowerSTACKMAP(MCStreamer &OutStreamer,
                          StackMaps &SM,
                          const MachineInstr &MI)
{
  unsigned NumNOPBytes = MI.getOperand(1).getImm();
  SM.recordStackMap(MI);
  // Emit padding.
  // FIXME: These nops ensure that the stackmap's shadow is covered by
  // instructions from the same basic block, but the nops should not be
  // necessary if instructions from the same block follow the stackmap.
  for (unsigned i = 0; i < NumNOPBytes; ++i)
    OutStreamer.EmitInstruction(MCInstBuilder(X86::NOOP));
}

// Lower a patchpoint of the form:
// [<def>], <id>, <numBytes>, <target>, <numArgs>, <cc>, ...
static void LowerPATCHPOINT(MCStreamer &OutStreamer,
                            StackMaps &SM,
                            const MachineInstr &MI) {
  SM.recordPatchPoint(MI);

  PatchPointOpers opers(&MI);
  unsigned ScratchIdx = opers.getNextScratchIdx();
  unsigned EncodedBytes = 0;
  int64_t CallTarget = opers.getMetaOper(PatchPointOpers::TargetPos).getImm();
  if (CallTarget) {
    // Emit MOV to materialize the target address and the CALL to target.
    // This is encoded with 12-13 bytes, depending on which register is used.
    // We conservatively assume that it is 12 bytes and emit in worst case one
    // extra NOP byte.
    EncodedBytes = 12;
    OutStreamer.EmitInstruction(MCInstBuilder(X86::MOV64ri)
                                .addReg(MI.getOperand(ScratchIdx).getReg())
                                .addImm(CallTarget));
    OutStreamer.EmitInstruction(MCInstBuilder(X86::CALL64r)
                                .addReg(MI.getOperand(ScratchIdx).getReg()));
  }
  // Emit padding.
  unsigned NumBytes = opers.getMetaOper(PatchPointOpers::NBytesPos).getImm();
  assert(NumBytes >= EncodedBytes &&
         "Patchpoint can't request size less than the length of a call.");

  for (unsigned i = EncodedBytes; i < NumBytes; ++i)
    OutStreamer.EmitInstruction(MCInstBuilder(X86::NOOP));
}

void X86AsmPrinter::EmitInstruction(const MachineInstr *MI) {
  X86MCInstLower MCInstLowering(*MF, *this);
  switch (MI->getOpcode()) {
  case TargetOpcode::DBG_VALUE:
    llvm_unreachable("Should be handled target independently");

  // Emit nothing here but a comment if we can.
  case X86::Int_MemBarrier:
    if (OutStreamer.hasRawTextSupport())
      OutStreamer.EmitRawText(StringRef("\t#MEMBARRIER"));
    return;


  case X86::EH_RETURN:
  case X86::EH_RETURN64: {
    // Lower these as normal, but add some comments.
    unsigned Reg = MI->getOperand(0).getReg();
    OutStreamer.AddComment(StringRef("eh_return, addr: %") +
                           X86ATTInstPrinter::getRegisterName(Reg));
    break;
  }
  case X86::TAILJMPr:
  case X86::TAILJMPd:
  case X86::TAILJMPd64:
    // Lower these as normal, but add some comments.
    OutStreamer.AddComment("TAILCALL");
    break;

  case X86::TLS_addr32:
  case X86::TLS_addr64:
  case X86::TLS_base_addr32:
  case X86::TLS_base_addr64:
    return LowerTlsAddr(OutStreamer, MCInstLowering, *MI);

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
    OutStreamer.EmitInstruction(MCInstBuilder(X86::CALLpcrel32)
      .addExpr(MCSymbolRefExpr::Create(PICBase, OutContext)));

    // Emit the label.
    OutStreamer.EmitLabel(PICBase);

    // popl $reg
    OutStreamer.EmitInstruction(MCInstBuilder(X86::POP32r)
      .addReg(MI->getOperand(0).getReg()));
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
    MCSymbol *DotSym = OutContext.CreateTempSymbol();
    OutStreamer.EmitLabel(DotSym);

    // Now that we have emitted the label, lower the complex operand expression.
    MCSymbol *OpSym = MCInstLowering.GetSymbolFromOperand(MI->getOperand(2));

    const MCExpr *DotExpr = MCSymbolRefExpr::Create(DotSym, OutContext);
    const MCExpr *PICBase =
      MCSymbolRefExpr::Create(MF->getPICBaseSymbol(), OutContext);
    DotExpr = MCBinaryExpr::CreateSub(DotExpr, PICBase, OutContext);

    DotExpr = MCBinaryExpr::CreateAdd(MCSymbolRefExpr::Create(OpSym,OutContext),
                                      DotExpr, OutContext);

    OutStreamer.EmitInstruction(MCInstBuilder(X86::ADD32ri)
      .addReg(MI->getOperand(0).getReg())
      .addReg(MI->getOperand(1).getReg())
      .addExpr(DotExpr));
    return;
  }

  case TargetOpcode::STACKMAP:
    return LowerSTACKMAP(OutStreamer, SM, *MI);

  case TargetOpcode::PATCHPOINT:
    return LowerPATCHPOINT(OutStreamer, SM, *MI);

  case X86::MORESTACK_RET:
    OutStreamer.EmitInstruction(MCInstBuilder(X86::RET));
    return;

  case X86::MORESTACK_RET_RESTORE_R10:
    // Return, then restore R10.
    OutStreamer.EmitInstruction(MCInstBuilder(X86::RET));
    OutStreamer.EmitInstruction(MCInstBuilder(X86::MOV64rr)
      .addReg(X86::R10)
      .addReg(X86::RAX));
    return;
  }

  MCInst TmpInst;
  MCInstLowering.Lower(MI, TmpInst);
  OutStreamer.EmitInstruction(TmpInst);
}
