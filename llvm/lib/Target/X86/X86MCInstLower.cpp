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
#include "X86COFFMachineModuleInfo.h"
#include "InstPrinter/X86ATTInstPrinter.h"
#include "llvm/Type.h"
#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Target/Mangler.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/ADT/SmallString.h"
using namespace llvm;

namespace {

/// X86MCInstLower - This class is used to lower an MachineInstr into an MCInst.
class X86MCInstLower {
  MCContext &Ctx;
  Mangler *Mang;
  const MachineFunction &MF;
  const TargetMachine &TM;
  const MCAsmInfo &MAI;
  X86AsmPrinter &AsmPrinter;
public:
  X86MCInstLower(Mangler *mang, const MachineFunction &MF,
                 X86AsmPrinter &asmprinter);

  void Lower(const MachineInstr *MI, MCInst &OutMI) const;

  MCSymbol *GetSymbolFromOperand(const MachineOperand &MO) const;
  MCOperand LowerSymbolOperand(const MachineOperand &MO, MCSymbol *Sym) const;

private:
  MachineModuleInfoMachO &getMachOMMI() const;
};

} // end anonymous namespace

X86MCInstLower::X86MCInstLower(Mangler *mang, const MachineFunction &mf,
                               X86AsmPrinter &asmprinter)
: Ctx(mf.getContext()), Mang(mang), MF(mf), TM(mf.getTarget()),
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

    Mang->getNameWithPrefix(Name, GV, isImplicitlyPrivate);
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
        StubValueTy(Mang->getSymbol(MO.getGlobal()),
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
        StubValueTy(Mang->getSymbol(MO.getGlobal()),
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
        StubValueTy(Mang->getSymbol(MO.getGlobal()),
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



static void lower_subreg32(MCInst *MI, unsigned OpNo) {
  // Convert registers in the addr mode according to subreg32.
  unsigned Reg = MI->getOperand(OpNo).getReg();
  if (Reg != 0)
    MI->getOperand(OpNo).setReg(getX86SubSuperRegister(Reg, MVT::i32));
}

static void lower_lea64_32mem(MCInst *MI, unsigned OpNo) {
  // Convert registers in the addr mode according to subreg64.
  for (unsigned i = 0; i != 4; ++i) {
    if (!MI->getOperand(OpNo+i).isReg()) continue;

    unsigned Reg = MI->getOperand(OpNo+i).getReg();
    if (Reg == 0) continue;

    MI->getOperand(OpNo+i).setReg(getX86SubSuperRegister(Reg, MVT::i64));
  }
}

/// LowerSubReg32_Op0 - Things like MOVZX16rr8 -> MOVZX32rr8.
static void LowerSubReg32_Op0(MCInst &OutMI, unsigned NewOpc) {
  OutMI.setOpcode(NewOpc);
  lower_subreg32(&OutMI, 0);
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
  case X86::LEA64_32r: // Handle 'subreg rewriting' for the lea64_32mem operand.
    lower_lea64_32mem(&OutMI, 1);
    // FALL THROUGH.
  case X86::LEA64r:
  case X86::LEA16r:
  case X86::LEA32r:
    // LEA should have a segment register, but it must be empty.
    assert(OutMI.getNumOperands() == 1+X86::AddrNumOperands &&
           "Unexpected # of LEA operands");
    assert(OutMI.getOperand(1+X86::AddrSegmentReg).getReg() == 0 &&
           "LEA has segment specified!");
    break;
  case X86::MOVZX64rr32:  LowerSubReg32_Op0(OutMI, X86::MOV32rr); break;
  case X86::MOVZX64rm32:  LowerSubReg32_Op0(OutMI, X86::MOV32rm); break;
  case X86::MOV64ri64i32: LowerSubReg32_Op0(OutMI, X86::MOV32ri); break;
  case X86::MOVZX64rr8:   LowerSubReg32_Op0(OutMI, X86::MOVZX32rr8); break;
  case X86::MOVZX64rm8:   LowerSubReg32_Op0(OutMI, X86::MOVZX32rm8); break;
  case X86::MOVZX64rr16:  LowerSubReg32_Op0(OutMI, X86::MOVZX32rr16); break;
  case X86::MOVZX64rm16:  LowerSubReg32_Op0(OutMI, X86::MOVZX32rm16); break;
  case X86::MOV8r0:       LowerUnaryToTwoAddr(OutMI, X86::XOR8rr); break;
  case X86::MOV32r0:      LowerUnaryToTwoAddr(OutMI, X86::XOR32rr); break;

  case X86::MOV16r0:
    LowerSubReg32_Op0(OutMI, X86::MOV32r0);   // MOV16r0 -> MOV32r0
    LowerUnaryToTwoAddr(OutMI, X86::XOR32rr); // MOV32r0 -> XOR32rr
    break;
  case X86::MOV64r0:
    LowerSubReg32_Op0(OutMI, X86::MOV32r0);   // MOV64r0 -> MOV32r0
    LowerUnaryToTwoAddr(OutMI, X86::XOR32rr); // MOV32r0 -> XOR32rr
    break;

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

  case X86::MORESTACK_RET:
    OutMI.setOpcode(X86::RET);
    break;

  case X86::MORESTACK_RET_RESTORE_R10: {
    MCInst retInst;

    OutMI.setOpcode(X86::MOV64rr);
    OutMI.addOperand(MCOperand::CreateReg(X86::R10));
    OutMI.addOperand(MCOperand::CreateReg(X86::RAX));

    retInst.setOpcode(X86::RET);
    AsmPrinter.OutStreamer.EmitInstruction(retInst);
    break;
  }
  }
}

static void LowerTlsAddr(MCStreamer &OutStreamer,
                         X86MCInstLower &MCInstLowering,
                         const MachineInstr &MI) {

  bool is64Bits = MI.getOpcode() == X86::TLS_addr64 ||
                  MI.getOpcode() == X86::TLS_base_addr64;

  bool needsPadding = MI.getOpcode() == X86::TLS_addr64;

  MCContext &context = OutStreamer.getContext();

  if (needsPadding) {
    MCInst prefix;
    prefix.setOpcode(X86::DATA16_PREFIX);
    OutStreamer.EmitInstruction(prefix);
  }

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
    MCInst prefix;
    prefix.setOpcode(X86::DATA16_PREFIX);
    OutStreamer.EmitInstruction(prefix);
    prefix.setOpcode(X86::DATA16_PREFIX);
    OutStreamer.EmitInstruction(prefix);
    prefix.setOpcode(X86::REX64_PREFIX);
    OutStreamer.EmitInstruction(prefix);
  }

  MCInst call;
  if (is64Bits)
    call.setOpcode(X86::CALL64pcrel32);
  else
    call.setOpcode(X86::CALLpcrel32);
  StringRef name = is64Bits ? "__tls_get_addr" : "___tls_get_addr";
  MCSymbol *tlsGetAddr = context.GetOrCreateSymbol(name);
  const MCSymbolRefExpr *tlsRef =
    MCSymbolRefExpr::Create(tlsGetAddr,
                            MCSymbolRefExpr::VK_PLT,
                            context);

  call.addOperand(MCOperand::CreateExpr(tlsRef));
  OutStreamer.EmitInstruction(call);
}

void X86AsmPrinter::EmitInstruction(const MachineInstr *MI) {
  X86MCInstLower MCInstLowering(Mang, *MF, *this);
  switch (MI->getOpcode()) {
  case TargetOpcode::DBG_VALUE:
    if (isVerbose() && OutStreamer.hasRawTextSupport()) {
      std::string TmpStr;
      raw_string_ostream OS(TmpStr);
      PrintDebugValueComment(MI, OS);
      OutStreamer.EmitRawText(StringRef(OS.str()));
    }
    return;

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
    MCInst TmpInst;
    // This is a pseudo op for a two instruction sequence with a label, which
    // looks like:
    //     call "L1$pb"
    // "L1$pb":
    //     popl %esi

    // Emit the call.
    MCSymbol *PICBase = MF->getPICBaseSymbol();
    TmpInst.setOpcode(X86::CALLpcrel32);
    // FIXME: We would like an efficient form for this, so we don't have to do a
    // lot of extra uniquing.
    TmpInst.addOperand(MCOperand::CreateExpr(MCSymbolRefExpr::Create(PICBase,
                                                                 OutContext)));
    OutStreamer.EmitInstruction(TmpInst);

    // Emit the label.
    OutStreamer.EmitLabel(PICBase);

    // popl $reg
    TmpInst.setOpcode(X86::POP32r);
    TmpInst.getOperand(0) = MCOperand::CreateReg(MI->getOperand(0).getReg());
    OutStreamer.EmitInstruction(TmpInst);
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

    MCInst TmpInst;
    TmpInst.setOpcode(X86::ADD32ri);
    TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
    TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(1).getReg()));
    TmpInst.addOperand(MCOperand::CreateExpr(DotExpr));
    OutStreamer.EmitInstruction(TmpInst);
    return;
  }
  }

  MCInst TmpInst;
  MCInstLowering.Lower(MI, TmpInst);
  OutStreamer.EmitInstruction(TmpInst);
}
