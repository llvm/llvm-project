//===-- X86/X86CodeEmitter.cpp - Convert X86 code to machine code ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the pass that transforms the X86 machine instructions into
// relocatable machine code.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "x86-emitter"
#include "X86InstrInfo.h"
#include "X86JITInfo.h"
#include "X86Subtarget.h"
#include "X86TargetMachine.h"
#include "X86Relocations.h"
#include "X86.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Function.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetOptions.h"
using namespace llvm;

STATISTIC(NumEmitted, "Number of machine instructions emitted");

namespace {
  class VISIBILITY_HIDDEN Emitter : public MachineFunctionPass {
    const X86InstrInfo  *II;
    const TargetData    *TD;
    X86TargetMachine    &TM;
    MachineCodeEmitter  &MCE;
    intptr_t PICBaseOffset;
    bool Is64BitMode;
    bool IsPIC;
  public:
    static char ID;
    explicit Emitter(X86TargetMachine &tm, MachineCodeEmitter &mce)
      : MachineFunctionPass(&ID), II(0), TD(0), TM(tm), 
      MCE(mce), PICBaseOffset(0), Is64BitMode(false),
      IsPIC(TM.getRelocationModel() == Reloc::PIC_) {}
    Emitter(X86TargetMachine &tm, MachineCodeEmitter &mce,
            const X86InstrInfo &ii, const TargetData &td, bool is64)
      : MachineFunctionPass(&ID), II(&ii), TD(&td), TM(tm), 
      MCE(mce), PICBaseOffset(0), Is64BitMode(is64),
      IsPIC(TM.getRelocationModel() == Reloc::PIC_) {}

    bool runOnMachineFunction(MachineFunction &MF);

    virtual const char *getPassName() const {
      return "X86 Machine Code Emitter";
    }

    void emitInstruction(const MachineInstr &MI,
                         const TargetInstrDesc *Desc);
    
    void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<MachineModuleInfo>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

  private:
    void emitPCRelativeBlockAddress(MachineBasicBlock *MBB);
    void emitGlobalAddress(GlobalValue *GV, unsigned Reloc,
                           intptr_t Disp = 0, intptr_t PCAdj = 0,
                           bool NeedStub = false, bool Indirect = false);
    void emitExternalSymbolAddress(const char *ES, unsigned Reloc);
    void emitConstPoolAddress(unsigned CPI, unsigned Reloc, intptr_t Disp = 0,
                              intptr_t PCAdj = 0);
    void emitJumpTableAddress(unsigned JTI, unsigned Reloc,
                              intptr_t PCAdj = 0);

    void emitDisplacementField(const MachineOperand *RelocOp, int DispVal,
                               intptr_t PCAdj = 0);

    void emitRegModRMByte(unsigned ModRMReg, unsigned RegOpcodeField);
    void emitRegModRMByte(unsigned RegOpcodeField);
    void emitSIBByte(unsigned SS, unsigned Index, unsigned Base);
    void emitConstant(uint64_t Val, unsigned Size);

    void emitMemModRMByte(const MachineInstr &MI,
                          unsigned Op, unsigned RegOpcodeField,
                          intptr_t PCAdj = 0);

    unsigned getX86RegNum(unsigned RegNo) const;

    bool gvNeedsNonLazyPtr(const GlobalValue *GV);
  };
  char Emitter::ID = 0;
}

/// createX86CodeEmitterPass - Return a pass that emits the collected X86 code
/// to the specified MCE object.
FunctionPass *llvm::createX86CodeEmitterPass(X86TargetMachine &TM,
                                             MachineCodeEmitter &MCE) {
  return new Emitter(TM, MCE);
}

bool Emitter::runOnMachineFunction(MachineFunction &MF) {
 
  MCE.setModuleInfo(&getAnalysis<MachineModuleInfo>());
  
  II = TM.getInstrInfo();
  TD = TM.getTargetData();
  Is64BitMode = TM.getSubtarget<X86Subtarget>().is64Bit();
  IsPIC = TM.getRelocationModel() == Reloc::PIC_;
  
  do {
    DOUT << "JITTing function '" << MF.getFunction()->getName() << "'\n";
    MCE.startFunction(MF);
    for (MachineFunction::iterator MBB = MF.begin(), E = MF.end(); 
         MBB != E; ++MBB) {
      MCE.StartMachineBasicBlock(MBB);
      for (MachineBasicBlock::const_iterator I = MBB->begin(), E = MBB->end();
           I != E; ++I) {
        const TargetInstrDesc &Desc = I->getDesc();
        emitInstruction(*I, &Desc);
        // MOVPC32r is basically a call plus a pop instruction.
        if (Desc.getOpcode() == X86::MOVPC32r)
          emitInstruction(*I, &II->get(X86::POP32r));
        NumEmitted++;  // Keep track of the # of mi's emitted
      }
    }
  } while (MCE.finishFunction(MF));

  return false;
}

/// emitPCRelativeBlockAddress - This method keeps track of the information
/// necessary to resolve the address of this block later and emits a dummy
/// value.
///
void Emitter::emitPCRelativeBlockAddress(MachineBasicBlock *MBB) {
  // Remember where this reference was and where it is to so we can
  // deal with it later.
  MCE.addRelocation(MachineRelocation::getBB(MCE.getCurrentPCOffset(),
                                             X86::reloc_pcrel_word, MBB));
  MCE.emitWordLE(0);
}

/// emitGlobalAddress - Emit the specified address to the code stream assuming
/// this is part of a "take the address of a global" instruction.
///
void Emitter::emitGlobalAddress(GlobalValue *GV, unsigned Reloc,
                                intptr_t Disp /* = 0 */,
                                intptr_t PCAdj /* = 0 */,
                                bool NeedStub /* = false */,
                                bool Indirect /* = false */) {
  intptr_t RelocCST = 0;
  if (Reloc == X86::reloc_picrel_word)
    RelocCST = PICBaseOffset;
  else if (Reloc == X86::reloc_pcrel_word)
    RelocCST = PCAdj;
  MachineRelocation MR = Indirect
    ? MachineRelocation::getIndirectSymbol(MCE.getCurrentPCOffset(), Reloc,
                                           GV, RelocCST, NeedStub)
    : MachineRelocation::getGV(MCE.getCurrentPCOffset(), Reloc,
                               GV, RelocCST, NeedStub);
  MCE.addRelocation(MR);
  // The relocated value will be added to the displacement
  if (Reloc == X86::reloc_absolute_dword)
    MCE.emitDWordLE(Disp);
  else
    MCE.emitWordLE((int32_t)Disp);
}

/// emitExternalSymbolAddress - Arrange for the address of an external symbol to
/// be emitted to the current location in the function, and allow it to be PC
/// relative.
void Emitter::emitExternalSymbolAddress(const char *ES, unsigned Reloc) {
  intptr_t RelocCST = (Reloc == X86::reloc_picrel_word) ? PICBaseOffset : 0;
  MCE.addRelocation(MachineRelocation::getExtSym(MCE.getCurrentPCOffset(),
                                                 Reloc, ES, RelocCST));
  if (Reloc == X86::reloc_absolute_dword)
    MCE.emitDWordLE(0);
  else
    MCE.emitWordLE(0);
}

/// emitConstPoolAddress - Arrange for the address of an constant pool
/// to be emitted to the current location in the function, and allow it to be PC
/// relative.
void Emitter::emitConstPoolAddress(unsigned CPI, unsigned Reloc,
                                   intptr_t Disp /* = 0 */,
                                   intptr_t PCAdj /* = 0 */) {
  intptr_t RelocCST = 0;
  if (Reloc == X86::reloc_picrel_word)
    RelocCST = PICBaseOffset;
  else if (Reloc == X86::reloc_pcrel_word)
    RelocCST = PCAdj;
  MCE.addRelocation(MachineRelocation::getConstPool(MCE.getCurrentPCOffset(),
                                                    Reloc, CPI, RelocCST));
  // The relocated value will be added to the displacement
  if (Reloc == X86::reloc_absolute_dword)
    MCE.emitDWordLE(Disp);
  else
    MCE.emitWordLE((int32_t)Disp);
}

/// emitJumpTableAddress - Arrange for the address of a jump table to
/// be emitted to the current location in the function, and allow it to be PC
/// relative.
void Emitter::emitJumpTableAddress(unsigned JTI, unsigned Reloc,
                                   intptr_t PCAdj /* = 0 */) {
  intptr_t RelocCST = 0;
  if (Reloc == X86::reloc_picrel_word)
    RelocCST = PICBaseOffset;
  else if (Reloc == X86::reloc_pcrel_word)
    RelocCST = PCAdj;
  MCE.addRelocation(MachineRelocation::getJumpTable(MCE.getCurrentPCOffset(),
                                                    Reloc, JTI, RelocCST));
  // The relocated value will be added to the displacement
  if (Reloc == X86::reloc_absolute_dword)
    MCE.emitDWordLE(0);
  else
    MCE.emitWordLE(0);
}

unsigned Emitter::getX86RegNum(unsigned RegNo) const {
  return II->getRegisterInfo().getX86RegNum(RegNo);
}

inline static unsigned char ModRMByte(unsigned Mod, unsigned RegOpcode,
                                      unsigned RM) {
  assert(Mod < 4 && RegOpcode < 8 && RM < 8 && "ModRM Fields out of range!");
  return RM | (RegOpcode << 3) | (Mod << 6);
}

void Emitter::emitRegModRMByte(unsigned ModRMReg, unsigned RegOpcodeFld){
  MCE.emitByte(ModRMByte(3, RegOpcodeFld, getX86RegNum(ModRMReg)));
}

void Emitter::emitRegModRMByte(unsigned RegOpcodeFld) {
  MCE.emitByte(ModRMByte(3, RegOpcodeFld, 0));
}

void Emitter::emitSIBByte(unsigned SS, unsigned Index, unsigned Base) {
  // SIB byte is in the same format as the ModRMByte...
  MCE.emitByte(ModRMByte(SS, Index, Base));
}

void Emitter::emitConstant(uint64_t Val, unsigned Size) {
  // Output the constant in little endian byte order...
  for (unsigned i = 0; i != Size; ++i) {
    MCE.emitByte(Val & 255);
    Val >>= 8;
  }
}

/// isDisp8 - Return true if this signed displacement fits in a 8-bit 
/// sign-extended field. 
static bool isDisp8(int Value) {
  return Value == (signed char)Value;
}

bool Emitter::gvNeedsNonLazyPtr(const GlobalValue *GV) {
  // For Darwin, simulate the linktime GOT by using the same non-lazy-pointer
  // mechanism as 32-bit mode.
  return (!Is64BitMode || TM.getSubtarget<X86Subtarget>().isTargetDarwin()) &&
    TM.getSubtarget<X86Subtarget>().GVRequiresExtraLoad(GV, TM, false);
}

void Emitter::emitDisplacementField(const MachineOperand *RelocOp,
                                    int DispVal, intptr_t PCAdj) {
  // If this is a simple integer displacement that doesn't require a relocation,
  // emit it now.
  if (!RelocOp) {
    emitConstant(DispVal, 4);
    return;
  }
  
  // Otherwise, this is something that requires a relocation.  Emit it as such
  // now.
  if (RelocOp->isGlobal()) {
    // In 64-bit static small code model, we could potentially emit absolute.
    // But it's probably not beneficial.
    //  89 05 00 00 00 00     mov    %eax,0(%rip)  # PC-relative
    //  89 04 25 00 00 00 00  mov    %eax,0x0      # Absolute
    unsigned rt = Is64BitMode ? X86::reloc_pcrel_word
      : (IsPIC ? X86::reloc_picrel_word : X86::reloc_absolute_word);
    bool NeedStub = isa<Function>(RelocOp->getGlobal());
    bool Indirect = gvNeedsNonLazyPtr(RelocOp->getGlobal());
    emitGlobalAddress(RelocOp->getGlobal(), rt, RelocOp->getOffset(),
                      PCAdj, NeedStub, Indirect);
  } else if (RelocOp->isCPI()) {
    unsigned rt = Is64BitMode ? X86::reloc_pcrel_word : X86::reloc_picrel_word;
    emitConstPoolAddress(RelocOp->getIndex(), rt,
                         RelocOp->getOffset(), PCAdj);
  } else if (RelocOp->isJTI()) {
    unsigned rt = Is64BitMode ? X86::reloc_pcrel_word : X86::reloc_picrel_word;
    emitJumpTableAddress(RelocOp->getIndex(), rt, PCAdj);
  } else {
    assert(0 && "Unknown value to relocate!");
  }
}

void Emitter::emitMemModRMByte(const MachineInstr &MI,
                               unsigned Op, unsigned RegOpcodeField,
                               intptr_t PCAdj) {
  const MachineOperand &Op3 = MI.getOperand(Op+3);
  int DispVal = 0;
  const MachineOperand *DispForReloc = 0;
  
  // Figure out what sort of displacement we have to handle here.
  if (Op3.isGlobal()) {
    DispForReloc = &Op3;
  } else if (Op3.isCPI()) {
    if (Is64BitMode || IsPIC) {
      DispForReloc = &Op3;
    } else {
      DispVal += MCE.getConstantPoolEntryAddress(Op3.getIndex());
      DispVal += Op3.getOffset();
    }
  } else if (Op3.isJTI()) {
    if (Is64BitMode || IsPIC) {
      DispForReloc = &Op3;
    } else {
      DispVal += MCE.getJumpTableEntryAddress(Op3.getIndex());
    }
  } else {
    DispVal = Op3.getImm();
  }

  const MachineOperand &Base     = MI.getOperand(Op);
  const MachineOperand &Scale    = MI.getOperand(Op+1);
  const MachineOperand &IndexReg = MI.getOperand(Op+2);

  unsigned BaseReg = Base.getReg();

  // Is a SIB byte needed?
  if ((!Is64BitMode || DispForReloc) && IndexReg.getReg() == 0 &&
      (BaseReg == 0 || getX86RegNum(BaseReg) != N86::ESP)) {
    if (BaseReg == 0) {  // Just a displacement?
      // Emit special case [disp32] encoding
      MCE.emitByte(ModRMByte(0, RegOpcodeField, 5));
      
      emitDisplacementField(DispForReloc, DispVal, PCAdj);
    } else {
      unsigned BaseRegNo = getX86RegNum(BaseReg);
      if (!DispForReloc && DispVal == 0 && BaseRegNo != N86::EBP) {
        // Emit simple indirect register encoding... [EAX] f.e.
        MCE.emitByte(ModRMByte(0, RegOpcodeField, BaseRegNo));
      } else if (!DispForReloc && isDisp8(DispVal)) {
        // Emit the disp8 encoding... [REG+disp8]
        MCE.emitByte(ModRMByte(1, RegOpcodeField, BaseRegNo));
        emitConstant(DispVal, 1);
      } else {
        // Emit the most general non-SIB encoding: [REG+disp32]
        MCE.emitByte(ModRMByte(2, RegOpcodeField, BaseRegNo));
        emitDisplacementField(DispForReloc, DispVal, PCAdj);
      }
    }

  } else {  // We need a SIB byte, so start by outputting the ModR/M byte first
    assert(IndexReg.getReg() != X86::ESP &&
           IndexReg.getReg() != X86::RSP && "Cannot use ESP as index reg!");

    bool ForceDisp32 = false;
    bool ForceDisp8  = false;
    if (BaseReg == 0) {
      // If there is no base register, we emit the special case SIB byte with
      // MOD=0, BASE=5, to JUST get the index, scale, and displacement.
      MCE.emitByte(ModRMByte(0, RegOpcodeField, 4));
      ForceDisp32 = true;
    } else if (DispForReloc) {
      // Emit the normal disp32 encoding.
      MCE.emitByte(ModRMByte(2, RegOpcodeField, 4));
      ForceDisp32 = true;
    } else if (DispVal == 0 && getX86RegNum(BaseReg) != N86::EBP) {
      // Emit no displacement ModR/M byte
      MCE.emitByte(ModRMByte(0, RegOpcodeField, 4));
    } else if (isDisp8(DispVal)) {
      // Emit the disp8 encoding...
      MCE.emitByte(ModRMByte(1, RegOpcodeField, 4));
      ForceDisp8 = true;           // Make sure to force 8 bit disp if Base=EBP
    } else {
      // Emit the normal disp32 encoding...
      MCE.emitByte(ModRMByte(2, RegOpcodeField, 4));
    }

    // Calculate what the SS field value should be...
    static const unsigned SSTable[] = { ~0, 0, 1, ~0, 2, ~0, ~0, ~0, 3 };
    unsigned SS = SSTable[Scale.getImm()];

    if (BaseReg == 0) {
      // Handle the SIB byte for the case where there is no base.  The
      // displacement has already been output.
      unsigned IndexRegNo;
      if (IndexReg.getReg())
        IndexRegNo = getX86RegNum(IndexReg.getReg());
      else
        IndexRegNo = 4;   // For example [ESP+1*<noreg>+4]
      emitSIBByte(SS, IndexRegNo, 5);
    } else {
      unsigned BaseRegNo = getX86RegNum(BaseReg);
      unsigned IndexRegNo;
      if (IndexReg.getReg())
        IndexRegNo = getX86RegNum(IndexReg.getReg());
      else
        IndexRegNo = 4;   // For example [ESP+1*<noreg>+4]
      emitSIBByte(SS, IndexRegNo, BaseRegNo);
    }

    // Do we need to output a displacement?
    if (ForceDisp8) {
      emitConstant(DispVal, 1);
    } else if (DispVal != 0 || ForceDisp32) {
      emitDisplacementField(DispForReloc, DispVal, PCAdj);
    }
  }
}

void Emitter::emitInstruction(const MachineInstr &MI,
                              const TargetInstrDesc *Desc) {
  DOUT << MI;

  unsigned Opcode = Desc->Opcode;

  // Emit the lock opcode prefix as needed.
  if (Desc->TSFlags & X86II::LOCK) MCE.emitByte(0xF0);

  // Emit segment override opcode prefix as needed.
  switch (Desc->TSFlags & X86II::SegOvrMask) {
  case X86II::FS:
    MCE.emitByte(0x64);
    break;
  case X86II::GS:
    MCE.emitByte(0x65);
    break;
  default: assert(0 && "Invalid segment!");
  case 0: break;  // No segment override!
  }

  // Emit the repeat opcode prefix as needed.
  if ((Desc->TSFlags & X86II::Op0Mask) == X86II::REP) MCE.emitByte(0xF3);

  // Emit the operand size opcode prefix as needed.
  if (Desc->TSFlags & X86II::OpSize) MCE.emitByte(0x66);

  // Emit the address size opcode prefix as needed.
  if (Desc->TSFlags & X86II::AdSize) MCE.emitByte(0x67);

  bool Need0FPrefix = false;
  switch (Desc->TSFlags & X86II::Op0Mask) {
  case X86II::TB:  // Two-byte opcode prefix
  case X86II::T8:  // 0F 38
  case X86II::TA:  // 0F 3A
    Need0FPrefix = true;
    break;
  case X86II::REP: break; // already handled.
  case X86II::XS:   // F3 0F
    MCE.emitByte(0xF3);
    Need0FPrefix = true;
    break;
  case X86II::XD:   // F2 0F
    MCE.emitByte(0xF2);
    Need0FPrefix = true;
    break;
  case X86II::D8: case X86II::D9: case X86II::DA: case X86II::DB:
  case X86II::DC: case X86II::DD: case X86II::DE: case X86II::DF:
    MCE.emitByte(0xD8+
                 (((Desc->TSFlags & X86II::Op0Mask)-X86II::D8)
                                   >> X86II::Op0Shift));
    break; // Two-byte opcode prefix
  default: assert(0 && "Invalid prefix!");
  case 0: break;  // No prefix!
  }

  if (Is64BitMode) {
    // REX prefix
    unsigned REX = X86InstrInfo::determineREX(MI);
    if (REX)
      MCE.emitByte(0x40 | REX);
  }

  // 0x0F escape code must be emitted just before the opcode.
  if (Need0FPrefix)
    MCE.emitByte(0x0F);

  switch (Desc->TSFlags & X86II::Op0Mask) {
  case X86II::T8:  // 0F 38
    MCE.emitByte(0x38);
    break;
  case X86II::TA:    // 0F 3A
    MCE.emitByte(0x3A);
    break;
  }

  // If this is a two-address instruction, skip one of the register operands.
  unsigned NumOps = Desc->getNumOperands();
  unsigned CurOp = 0;
  if (NumOps > 1 && Desc->getOperandConstraint(1, TOI::TIED_TO) != -1)
    ++CurOp;
  else if (NumOps > 2 && Desc->getOperandConstraint(NumOps-1, TOI::TIED_TO)== 0)
    // Skip the last source operand that is tied_to the dest reg. e.g. LXADD32
    --NumOps;

  unsigned char BaseOpcode = II->getBaseOpcodeFor(Desc);
  switch (Desc->TSFlags & X86II::FormMask) {
  default: assert(0 && "Unknown FormMask value in X86 MachineCodeEmitter!");
  case X86II::Pseudo:
    // Remember the current PC offset, this is the PIC relocation
    // base address.
    switch (Opcode) {
    default: 
      assert(0 && "psuedo instructions should be removed before code emission");
      break;
    case TargetInstrInfo::INLINEASM: {
      // We allow inline assembler nodes with empty bodies - they can
      // implicitly define registers, which is ok for JIT.
      if (MI.getOperand(0).getSymbolName()[0]) {
        assert(0 && "JIT does not support inline asm!\n");
        abort();
      }
      break;
    }
    case TargetInstrInfo::DBG_LABEL:
    case TargetInstrInfo::EH_LABEL:
      MCE.emitLabel(MI.getOperand(0).getImm());
      break;
    case TargetInstrInfo::IMPLICIT_DEF:
    case TargetInstrInfo::DECLARE:
    case X86::DWARF_LOC:
    case X86::FP_REG_KILL:
      break;
    case X86::TLS_tp: {
      MCE.emitByte(BaseOpcode);
      unsigned RegOpcodeField = getX86RegNum(MI.getOperand(0).getReg());
      MCE.emitByte(ModRMByte(0, RegOpcodeField, 5));
      emitConstant(0, 4);
      break;
    }
    case X86::TLS_gs_ri: {
      MCE.emitByte(BaseOpcode);
      unsigned RegOpcodeField = getX86RegNum(MI.getOperand(0).getReg());
      MCE.emitByte(ModRMByte(0, RegOpcodeField, 5));
      GlobalValue* GV = MI.getOperand(1).getGlobal();
      unsigned rt = Is64BitMode ? X86::reloc_pcrel_word
        : (IsPIC ? X86::reloc_picrel_word : X86::reloc_absolute_word);
      emitGlobalAddress(GV, rt);
      break;
    }
    case X86::MOVPC32r: {
      // This emits the "call" portion of this pseudo instruction.
      MCE.emitByte(BaseOpcode);
      emitConstant(0, X86InstrInfo::sizeOfImm(Desc));
      // Remember PIC base.
      PICBaseOffset = (intptr_t) MCE.getCurrentPCOffset();
      X86JITInfo *JTI = TM.getJITInfo();
      JTI->setPICBase(MCE.getCurrentPCValue());
      break;
    }
    }
    CurOp = NumOps;
    break;
  case X86II::RawFrm:
    MCE.emitByte(BaseOpcode);

    if (CurOp != NumOps) {
      const MachineOperand &MO = MI.getOperand(CurOp++);

      DOUT << "RawFrm CurOp " << CurOp << "\n";
      DOUT << "isMBB " << MO.isMBB() << "\n";
      DOUT << "isGlobal " << MO.isGlobal() << "\n";
      DOUT << "isSymbol " << MO.isSymbol() << "\n";
      DOUT << "isImm " << MO.isImm() << "\n";

      if (MO.isMBB()) {
        emitPCRelativeBlockAddress(MO.getMBB());
      } else if (MO.isGlobal()) {
        // Assume undefined functions may be outside the Small codespace.
        bool NeedStub = 
          (Is64BitMode && 
              (TM.getCodeModel() == CodeModel::Large ||
               TM.getSubtarget<X86Subtarget>().isTargetDarwin())) ||
          Opcode == X86::TAILJMPd;
        emitGlobalAddress(MO.getGlobal(), X86::reloc_pcrel_word,
                          MO.getOffset(), 0, NeedStub);
      } else if (MO.isSymbol()) {
        emitExternalSymbolAddress(MO.getSymbolName(), X86::reloc_pcrel_word);
      } else if (MO.isImm()) {
        emitConstant(MO.getImm(), X86InstrInfo::sizeOfImm(Desc));
      } else {
        assert(0 && "Unknown RawFrm operand!");
      }
    }
    break;

  case X86II::AddRegFrm:
    MCE.emitByte(BaseOpcode + getX86RegNum(MI.getOperand(CurOp++).getReg()));
    
    if (CurOp != NumOps) {
      const MachineOperand &MO1 = MI.getOperand(CurOp++);
      unsigned Size = X86InstrInfo::sizeOfImm(Desc);
      if (MO1.isImm())
        emitConstant(MO1.getImm(), Size);
      else {
        unsigned rt = Is64BitMode ? X86::reloc_pcrel_word
          : (IsPIC ? X86::reloc_picrel_word : X86::reloc_absolute_word);
        // This should not occur on Darwin for relocatable objects.
        if (Opcode == X86::MOV64ri)
          rt = X86::reloc_absolute_dword;  // FIXME: add X86II flag?
        if (MO1.isGlobal()) {
          bool NeedStub = isa<Function>(MO1.getGlobal());
          bool Indirect = gvNeedsNonLazyPtr(MO1.getGlobal());
          emitGlobalAddress(MO1.getGlobal(), rt, MO1.getOffset(), 0,
                            NeedStub, Indirect);
        } else if (MO1.isSymbol())
          emitExternalSymbolAddress(MO1.getSymbolName(), rt);
        else if (MO1.isCPI())
          emitConstPoolAddress(MO1.getIndex(), rt);
        else if (MO1.isJTI())
          emitJumpTableAddress(MO1.getIndex(), rt);
      }
    }
    break;

  case X86II::MRMDestReg: {
    MCE.emitByte(BaseOpcode);
    emitRegModRMByte(MI.getOperand(CurOp).getReg(),
                     getX86RegNum(MI.getOperand(CurOp+1).getReg()));
    CurOp += 2;
    if (CurOp != NumOps)
      emitConstant(MI.getOperand(CurOp++).getImm(), X86InstrInfo::sizeOfImm(Desc));
    break;
  }
  case X86II::MRMDestMem: {
    MCE.emitByte(BaseOpcode);
    emitMemModRMByte(MI, CurOp, getX86RegNum(MI.getOperand(CurOp+4).getReg()));
    CurOp += 5;
    if (CurOp != NumOps)
      emitConstant(MI.getOperand(CurOp++).getImm(), X86InstrInfo::sizeOfImm(Desc));
    break;
  }

  case X86II::MRMSrcReg:
    MCE.emitByte(BaseOpcode);
    emitRegModRMByte(MI.getOperand(CurOp+1).getReg(),
                     getX86RegNum(MI.getOperand(CurOp).getReg()));
    CurOp += 2;
    if (CurOp != NumOps)
      emitConstant(MI.getOperand(CurOp++).getImm(), X86InstrInfo::sizeOfImm(Desc));
    break;

  case X86II::MRMSrcMem: {
    intptr_t PCAdj = (CurOp+5 != NumOps) ? X86InstrInfo::sizeOfImm(Desc) : 0;

    MCE.emitByte(BaseOpcode);
    emitMemModRMByte(MI, CurOp+1, getX86RegNum(MI.getOperand(CurOp).getReg()),
                     PCAdj);
    CurOp += 5;
    if (CurOp != NumOps)
      emitConstant(MI.getOperand(CurOp++).getImm(), X86InstrInfo::sizeOfImm(Desc));
    break;
  }

  case X86II::MRM0r: case X86II::MRM1r:
  case X86II::MRM2r: case X86II::MRM3r:
  case X86II::MRM4r: case X86II::MRM5r:
  case X86II::MRM6r: case X86II::MRM7r: {
    MCE.emitByte(BaseOpcode);

    // Special handling of lfence and mfence. 
    if (Desc->getOpcode() == X86::LFENCE ||
        Desc->getOpcode() == X86::MFENCE)
      emitRegModRMByte((Desc->TSFlags & X86II::FormMask)-X86II::MRM0r);
    else
      emitRegModRMByte(MI.getOperand(CurOp++).getReg(),
                       (Desc->TSFlags & X86II::FormMask)-X86II::MRM0r);

    if (CurOp != NumOps) {
      const MachineOperand &MO1 = MI.getOperand(CurOp++);
      unsigned Size = X86InstrInfo::sizeOfImm(Desc);
      if (MO1.isImm())
        emitConstant(MO1.getImm(), Size);
      else {
        unsigned rt = Is64BitMode ? X86::reloc_pcrel_word
          : (IsPIC ? X86::reloc_picrel_word : X86::reloc_absolute_word);
        if (Opcode == X86::MOV64ri32)
          rt = X86::reloc_absolute_word;  // FIXME: add X86II flag?
        if (MO1.isGlobal()) {
          bool NeedStub = isa<Function>(MO1.getGlobal());
          bool Indirect = gvNeedsNonLazyPtr(MO1.getGlobal());
          emitGlobalAddress(MO1.getGlobal(), rt, MO1.getOffset(), 0,
                            NeedStub, Indirect);
        } else if (MO1.isSymbol())
          emitExternalSymbolAddress(MO1.getSymbolName(), rt);
        else if (MO1.isCPI())
          emitConstPoolAddress(MO1.getIndex(), rt);
        else if (MO1.isJTI())
          emitJumpTableAddress(MO1.getIndex(), rt);
      }
    }
    break;
  }

  case X86II::MRM0m: case X86II::MRM1m:
  case X86II::MRM2m: case X86II::MRM3m:
  case X86II::MRM4m: case X86II::MRM5m:
  case X86II::MRM6m: case X86II::MRM7m: {
    intptr_t PCAdj = (CurOp+4 != NumOps) ?
      (MI.getOperand(CurOp+4).isImm() ? X86InstrInfo::sizeOfImm(Desc) : 4) : 0;

    MCE.emitByte(BaseOpcode);
    emitMemModRMByte(MI, CurOp, (Desc->TSFlags & X86II::FormMask)-X86II::MRM0m,
                     PCAdj);
    CurOp += 4;

    if (CurOp != NumOps) {
      const MachineOperand &MO = MI.getOperand(CurOp++);
      unsigned Size = X86InstrInfo::sizeOfImm(Desc);
      if (MO.isImm())
        emitConstant(MO.getImm(), Size);
      else {
        unsigned rt = Is64BitMode ? X86::reloc_pcrel_word
          : (IsPIC ? X86::reloc_picrel_word : X86::reloc_absolute_word);
        if (Opcode == X86::MOV64mi32)
          rt = X86::reloc_absolute_word;  // FIXME: add X86II flag?
        if (MO.isGlobal()) {
          bool NeedStub = isa<Function>(MO.getGlobal());
          bool Indirect = gvNeedsNonLazyPtr(MO.getGlobal());
          emitGlobalAddress(MO.getGlobal(), rt, MO.getOffset(), 0,
                            NeedStub, Indirect);
        } else if (MO.isSymbol())
          emitExternalSymbolAddress(MO.getSymbolName(), rt);
        else if (MO.isCPI())
          emitConstPoolAddress(MO.getIndex(), rt);
        else if (MO.isJTI())
          emitJumpTableAddress(MO.getIndex(), rt);
      }
    }
    break;
  }

  case X86II::MRMInitReg:
    MCE.emitByte(BaseOpcode);
    // Duplicate register, used by things like MOV8r0 (aka xor reg,reg).
    emitRegModRMByte(MI.getOperand(CurOp).getReg(),
                     getX86RegNum(MI.getOperand(CurOp).getReg()));
    ++CurOp;
    break;
  }

  if (!Desc->isVariadic() && CurOp != NumOps) {
    cerr << "Cannot encode: ";
    MI.dump();
    cerr << '\n';
    abort();
  }
}
