//===-- X86CodeEmitter.cpp - Convert X86 code to machine code -------------===//
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
#include "llvm/LLVMContext.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/JITCodeEmitter.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetOptions.h"
using namespace llvm;

STATISTIC(NumEmitted, "Number of machine instructions emitted");

namespace {
  template<class CodeEmitter>
  class Emitter : public MachineFunctionPass {
    const X86InstrInfo  *II;
    const DataLayout    *TD;
    X86TargetMachine    &TM;
    CodeEmitter         &MCE;
    MachineModuleInfo   *MMI;
    intptr_t PICBaseOffset;
    bool Is64BitMode;
    bool IsPIC;
  public:
    static char ID;
    explicit Emitter(X86TargetMachine &tm, CodeEmitter &mce)
      : MachineFunctionPass(ID), II(0), TD(0), TM(tm),
      MCE(mce), PICBaseOffset(0), Is64BitMode(false),
      IsPIC(TM.getRelocationModel() == Reloc::PIC_) {}
    Emitter(X86TargetMachine &tm, CodeEmitter &mce,
            const X86InstrInfo &ii, const DataLayout &td, bool is64)
      : MachineFunctionPass(ID), II(&ii), TD(&td), TM(tm),
      MCE(mce), PICBaseOffset(0), Is64BitMode(is64),
      IsPIC(TM.getRelocationModel() == Reloc::PIC_) {}

    bool runOnMachineFunction(MachineFunction &MF);

    virtual const char *getPassName() const {
      return "X86 Machine Code Emitter";
    }

    void emitOpcodePrefix(uint64_t TSFlags, int MemOperand,
                          const MachineInstr &MI,
                          const MCInstrDesc *Desc) const;

    void emitVEXOpcodePrefix(uint64_t TSFlags, int MemOperand,
                             const MachineInstr &MI,
                             const MCInstrDesc *Desc) const;

    void emitSegmentOverridePrefix(uint64_t TSFlags,
                                   int MemOperand,
                                   const MachineInstr &MI) const;

    void emitInstruction(MachineInstr &MI, const MCInstrDesc *Desc);

    void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
      AU.addRequired<MachineModuleInfo>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

  private:
    void emitPCRelativeBlockAddress(MachineBasicBlock *MBB);
    void emitGlobalAddress(const GlobalValue *GV, unsigned Reloc,
                           intptr_t Disp = 0, intptr_t PCAdj = 0,
                           bool Indirect = false);
    void emitExternalSymbolAddress(const char *ES, unsigned Reloc);
    void emitConstPoolAddress(unsigned CPI, unsigned Reloc, intptr_t Disp = 0,
                              intptr_t PCAdj = 0);
    void emitJumpTableAddress(unsigned JTI, unsigned Reloc,
                              intptr_t PCAdj = 0);

    void emitDisplacementField(const MachineOperand *RelocOp, int DispVal,
                               intptr_t Adj = 0, bool IsPCRel = true);

    void emitRegModRMByte(unsigned ModRMReg, unsigned RegOpcodeField);
    void emitRegModRMByte(unsigned RegOpcodeField);
    void emitSIBByte(unsigned SS, unsigned Index, unsigned Base);
    void emitConstant(uint64_t Val, unsigned Size);

    void emitMemModRMByte(const MachineInstr &MI,
                          unsigned Op, unsigned RegOpcodeField,
                          intptr_t PCAdj = 0);

    unsigned getX86RegNum(unsigned RegNo) const {
      const TargetRegisterInfo *TRI = TM.getRegisterInfo();
      return TRI->getEncodingValue(RegNo) & 0x7;
    }

    unsigned char getVEXRegisterEncoding(const MachineInstr &MI,
                                         unsigned OpNum) const;
  };

template<class CodeEmitter>
  char Emitter<CodeEmitter>::ID = 0;
} // end anonymous namespace.

/// createX86CodeEmitterPass - Return a pass that emits the collected X86 code
/// to the specified templated MachineCodeEmitter object.
FunctionPass *llvm::createX86JITCodeEmitterPass(X86TargetMachine &TM,
                                                JITCodeEmitter &JCE) {
  return new Emitter<JITCodeEmitter>(TM, JCE);
}

template<class CodeEmitter>
bool Emitter<CodeEmitter>::runOnMachineFunction(MachineFunction &MF) {
  MMI = &getAnalysis<MachineModuleInfo>();
  MCE.setModuleInfo(MMI);

  II = TM.getInstrInfo();
  TD = TM.getDataLayout();
  Is64BitMode = TM.getSubtarget<X86Subtarget>().is64Bit();
  IsPIC = TM.getRelocationModel() == Reloc::PIC_;

  do {
    DEBUG(dbgs() << "JITTing function '" << MF.getName() << "'\n");
    MCE.startFunction(MF);
    for (MachineFunction::iterator MBB = MF.begin(), E = MF.end();
         MBB != E; ++MBB) {
      MCE.StartMachineBasicBlock(MBB);
      for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end();
           I != E; ++I) {
        const MCInstrDesc &Desc = I->getDesc();
        emitInstruction(*I, &Desc);
        // MOVPC32r is basically a call plus a pop instruction.
        if (Desc.getOpcode() == X86::MOVPC32r)
          emitInstruction(*I, &II->get(X86::POP32r));
        ++NumEmitted;  // Keep track of the # of mi's emitted
      }
    }
  } while (MCE.finishFunction(MF));

  return false;
}

/// determineREX - Determine if the MachineInstr has to be encoded with a X86-64
/// REX prefix which specifies 1) 64-bit instructions, 2) non-default operand
/// size, and 3) use of X86-64 extended registers.
static unsigned determineREX(const MachineInstr &MI) {
  unsigned REX = 0;
  const MCInstrDesc &Desc = MI.getDesc();

  // Pseudo instructions do not need REX prefix byte.
  if ((Desc.TSFlags & X86II::FormMask) == X86II::Pseudo)
    return 0;
  if (Desc.TSFlags & X86II::REX_W)
    REX |= 1 << 3;

  unsigned NumOps = Desc.getNumOperands();
  if (NumOps) {
    bool isTwoAddr = NumOps > 1 &&
      Desc.getOperandConstraint(1, MCOI::TIED_TO) != -1;

    // If it accesses SPL, BPL, SIL, or DIL, then it requires a 0x40 REX prefix.
    unsigned i = isTwoAddr ? 1 : 0;
    for (unsigned e = NumOps; i != e; ++i) {
      const MachineOperand& MO = MI.getOperand(i);
      if (MO.isReg()) {
        unsigned Reg = MO.getReg();
        if (X86II::isX86_64NonExtLowByteReg(Reg))
          REX |= 0x40;
      }
    }

    switch (Desc.TSFlags & X86II::FormMask) {
      case X86II::MRMInitReg:
        if (X86InstrInfo::isX86_64ExtendedReg(MI.getOperand(0)))
          REX |= (1 << 0) | (1 << 2);
        break;
      case X86II::MRMSrcReg: {
        if (X86InstrInfo::isX86_64ExtendedReg(MI.getOperand(0)))
          REX |= 1 << 2;
        i = isTwoAddr ? 2 : 1;
        for (unsigned e = NumOps; i != e; ++i) {
          const MachineOperand& MO = MI.getOperand(i);
          if (X86InstrInfo::isX86_64ExtendedReg(MO))
            REX |= 1 << 0;
        }
        break;
      }
      case X86II::MRMSrcMem: {
        if (X86InstrInfo::isX86_64ExtendedReg(MI.getOperand(0)))
          REX |= 1 << 2;
        unsigned Bit = 0;
        i = isTwoAddr ? 2 : 1;
        for (; i != NumOps; ++i) {
          const MachineOperand& MO = MI.getOperand(i);
          if (MO.isReg()) {
            if (X86InstrInfo::isX86_64ExtendedReg(MO))
              REX |= 1 << Bit;
            Bit++;
          }
        }
        break;
      }
      case X86II::MRM0m: case X86II::MRM1m:
      case X86II::MRM2m: case X86II::MRM3m:
      case X86II::MRM4m: case X86II::MRM5m:
      case X86II::MRM6m: case X86II::MRM7m:
      case X86II::MRMDestMem: {
        unsigned e = (isTwoAddr ? X86::AddrNumOperands+1 : X86::AddrNumOperands);
        i = isTwoAddr ? 1 : 0;
        if (NumOps > e && X86InstrInfo::isX86_64ExtendedReg(MI.getOperand(e)))
          REX |= 1 << 2;
        unsigned Bit = 0;
        for (; i != e; ++i) {
          const MachineOperand& MO = MI.getOperand(i);
          if (MO.isReg()) {
            if (X86InstrInfo::isX86_64ExtendedReg(MO))
              REX |= 1 << Bit;
            Bit++;
          }
        }
        break;
      }
      default: {
        if (X86InstrInfo::isX86_64ExtendedReg(MI.getOperand(0)))
          REX |= 1 << 0;
        i = isTwoAddr ? 2 : 1;
        for (unsigned e = NumOps; i != e; ++i) {
          const MachineOperand& MO = MI.getOperand(i);
          if (X86InstrInfo::isX86_64ExtendedReg(MO))
            REX |= 1 << 2;
        }
        break;
      }
    }
  }
  return REX;
}


/// emitPCRelativeBlockAddress - This method keeps track of the information
/// necessary to resolve the address of this block later and emits a dummy
/// value.
///
template<class CodeEmitter>
void Emitter<CodeEmitter>::emitPCRelativeBlockAddress(MachineBasicBlock *MBB) {
  // Remember where this reference was and where it is to so we can
  // deal with it later.
  MCE.addRelocation(MachineRelocation::getBB(MCE.getCurrentPCOffset(),
                                             X86::reloc_pcrel_word, MBB));
  MCE.emitWordLE(0);
}

/// emitGlobalAddress - Emit the specified address to the code stream assuming
/// this is part of a "take the address of a global" instruction.
///
template<class CodeEmitter>
void Emitter<CodeEmitter>::emitGlobalAddress(const GlobalValue *GV,
                                unsigned Reloc,
                                intptr_t Disp /* = 0 */,
                                intptr_t PCAdj /* = 0 */,
                                bool Indirect /* = false */) {
  intptr_t RelocCST = Disp;
  if (Reloc == X86::reloc_picrel_word)
    RelocCST = PICBaseOffset;
  else if (Reloc == X86::reloc_pcrel_word)
    RelocCST = PCAdj;
  MachineRelocation MR = Indirect
    ? MachineRelocation::getIndirectSymbol(MCE.getCurrentPCOffset(), Reloc,
                                           const_cast<GlobalValue *>(GV),
                                           RelocCST, false)
    : MachineRelocation::getGV(MCE.getCurrentPCOffset(), Reloc,
                               const_cast<GlobalValue *>(GV), RelocCST, false);
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
template<class CodeEmitter>
void Emitter<CodeEmitter>::emitExternalSymbolAddress(const char *ES,
                                                     unsigned Reloc) {
  intptr_t RelocCST = (Reloc == X86::reloc_picrel_word) ? PICBaseOffset : 0;

  // X86 never needs stubs because instruction selection will always pick
  // an instruction sequence that is large enough to hold any address
  // to a symbol.
  // (see X86ISelLowering.cpp, near 2039: X86TargetLowering::LowerCall)
  bool NeedStub = false;
  MCE.addRelocation(MachineRelocation::getExtSym(MCE.getCurrentPCOffset(),
                                                 Reloc, ES, RelocCST,
                                                 0, NeedStub));
  if (Reloc == X86::reloc_absolute_dword)
    MCE.emitDWordLE(0);
  else
    MCE.emitWordLE(0);
}

/// emitConstPoolAddress - Arrange for the address of an constant pool
/// to be emitted to the current location in the function, and allow it to be PC
/// relative.
template<class CodeEmitter>
void Emitter<CodeEmitter>::emitConstPoolAddress(unsigned CPI, unsigned Reloc,
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
template<class CodeEmitter>
void Emitter<CodeEmitter>::emitJumpTableAddress(unsigned JTI, unsigned Reloc,
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

inline static unsigned char ModRMByte(unsigned Mod, unsigned RegOpcode,
                                      unsigned RM) {
  assert(Mod < 4 && RegOpcode < 8 && RM < 8 && "ModRM Fields out of range!");
  return RM | (RegOpcode << 3) | (Mod << 6);
}

template<class CodeEmitter>
void Emitter<CodeEmitter>::emitRegModRMByte(unsigned ModRMReg,
                                            unsigned RegOpcodeFld){
  MCE.emitByte(ModRMByte(3, RegOpcodeFld, getX86RegNum(ModRMReg)));
}

template<class CodeEmitter>
void Emitter<CodeEmitter>::emitRegModRMByte(unsigned RegOpcodeFld) {
  MCE.emitByte(ModRMByte(3, RegOpcodeFld, 0));
}

template<class CodeEmitter>
void Emitter<CodeEmitter>::emitSIBByte(unsigned SS,
                                       unsigned Index,
                                       unsigned Base) {
  // SIB byte is in the same format as the ModRMByte...
  MCE.emitByte(ModRMByte(SS, Index, Base));
}

template<class CodeEmitter>
void Emitter<CodeEmitter>::emitConstant(uint64_t Val, unsigned Size) {
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

static bool gvNeedsNonLazyPtr(const MachineOperand &GVOp,
                              const TargetMachine &TM) {
  // For Darwin-64, simulate the linktime GOT by using the same non-lazy-pointer
  // mechanism as 32-bit mode.
  if (TM.getSubtarget<X86Subtarget>().is64Bit() &&
      !TM.getSubtarget<X86Subtarget>().isTargetDarwin())
    return false;

  // Return true if this is a reference to a stub containing the address of the
  // global, not the global itself.
  return isGlobalStubReference(GVOp.getTargetFlags());
}

template<class CodeEmitter>
void Emitter<CodeEmitter>::emitDisplacementField(const MachineOperand *RelocOp,
                                                 int DispVal,
                                                 intptr_t Adj /* = 0 */,
                                                 bool IsPCRel /* = true */) {
  // If this is a simple integer displacement that doesn't require a relocation,
  // emit it now.
  if (!RelocOp) {
    emitConstant(DispVal, 4);
    return;
  }

  // Otherwise, this is something that requires a relocation.  Emit it as such
  // now.
  unsigned RelocType = Is64BitMode ?
    (IsPCRel ? X86::reloc_pcrel_word : X86::reloc_absolute_word_sext)
    : (IsPIC ? X86::reloc_picrel_word : X86::reloc_absolute_word);
  if (RelocOp->isGlobal()) {
    // In 64-bit static small code model, we could potentially emit absolute.
    // But it's probably not beneficial. If the MCE supports using RIP directly
    // do it, otherwise fallback to absolute (this is determined by IsPCRel).
    //  89 05 00 00 00 00     mov    %eax,0(%rip)  # PC-relative
    //  89 04 25 00 00 00 00  mov    %eax,0x0      # Absolute
    bool Indirect = gvNeedsNonLazyPtr(*RelocOp, TM);
    emitGlobalAddress(RelocOp->getGlobal(), RelocType, RelocOp->getOffset(),
                      Adj, Indirect);
  } else if (RelocOp->isSymbol()) {
    emitExternalSymbolAddress(RelocOp->getSymbolName(), RelocType);
  } else if (RelocOp->isCPI()) {
    emitConstPoolAddress(RelocOp->getIndex(), RelocType,
                         RelocOp->getOffset(), Adj);
  } else {
    assert(RelocOp->isJTI() && "Unexpected machine operand!");
    emitJumpTableAddress(RelocOp->getIndex(), RelocType, Adj);
  }
}

template<class CodeEmitter>
void Emitter<CodeEmitter>::emitMemModRMByte(const MachineInstr &MI,
                                            unsigned Op,unsigned RegOpcodeField,
                                            intptr_t PCAdj) {
  const MachineOperand &Op3 = MI.getOperand(Op+3);
  int DispVal = 0;
  const MachineOperand *DispForReloc = 0;

  // Figure out what sort of displacement we have to handle here.
  if (Op3.isGlobal()) {
    DispForReloc = &Op3;
  } else if (Op3.isSymbol()) {
    DispForReloc = &Op3;
  } else if (Op3.isCPI()) {
    if (!MCE.earlyResolveAddresses() || Is64BitMode || IsPIC) {
      DispForReloc = &Op3;
    } else {
      DispVal += MCE.getConstantPoolEntryAddress(Op3.getIndex());
      DispVal += Op3.getOffset();
    }
  } else if (Op3.isJTI()) {
    if (!MCE.earlyResolveAddresses() || Is64BitMode || IsPIC) {
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

  // Handle %rip relative addressing.
  if (BaseReg == X86::RIP ||
      (Is64BitMode && DispForReloc)) { // [disp32+RIP] in X86-64 mode
    assert(IndexReg.getReg() == 0 && Is64BitMode &&
           "Invalid rip-relative address");
    MCE.emitByte(ModRMByte(0, RegOpcodeField, 5));
    emitDisplacementField(DispForReloc, DispVal, PCAdj, true);
    return;
  }

  // Indicate that the displacement will use an pcrel or absolute reference
  // by default. MCEs able to resolve addresses on-the-fly use pcrel by default
  // while others, unless explicit asked to use RIP, use absolute references.
  bool IsPCRel = MCE.earlyResolveAddresses() ? true : false;

  // Is a SIB byte needed?
  // If no BaseReg, issue a RIP relative instruction only if the MCE can
  // resolve addresses on-the-fly, otherwise use SIB (Intel Manual 2A, table
  // 2-7) and absolute references.
  unsigned BaseRegNo = -1U;
  if (BaseReg != 0 && BaseReg != X86::RIP)
    BaseRegNo = getX86RegNum(BaseReg);

  if (// The SIB byte must be used if there is an index register.
      IndexReg.getReg() == 0 &&
      // The SIB byte must be used if the base is ESP/RSP/R12, all of which
      // encode to an R/M value of 4, which indicates that a SIB byte is
      // present.
      BaseRegNo != N86::ESP &&
      // If there is no base register and we're in 64-bit mode, we need a SIB
      // byte to emit an addr that is just 'disp32' (the non-RIP relative form).
      (!Is64BitMode || BaseReg != 0)) {
    if (BaseReg == 0 ||          // [disp32]     in X86-32 mode
        BaseReg == X86::RIP) {   // [disp32+RIP] in X86-64 mode
      MCE.emitByte(ModRMByte(0, RegOpcodeField, 5));
      emitDisplacementField(DispForReloc, DispVal, PCAdj, true);
      return;
    }

    // If the base is not EBP/ESP and there is no displacement, use simple
    // indirect register encoding, this handles addresses like [EAX].  The
    // encoding for [EBP] with no displacement means [disp32] so we handle it
    // by emitting a displacement of 0 below.
    if (!DispForReloc && DispVal == 0 && BaseRegNo != N86::EBP) {
      MCE.emitByte(ModRMByte(0, RegOpcodeField, BaseRegNo));
      return;
    }

    // Otherwise, if the displacement fits in a byte, encode as [REG+disp8].
    if (!DispForReloc && isDisp8(DispVal)) {
      MCE.emitByte(ModRMByte(1, RegOpcodeField, BaseRegNo));
      emitConstant(DispVal, 1);
      return;
    }

    // Otherwise, emit the most general non-SIB encoding: [REG+disp32]
    MCE.emitByte(ModRMByte(2, RegOpcodeField, BaseRegNo));
    emitDisplacementField(DispForReloc, DispVal, PCAdj, IsPCRel);
    return;
  }

  // Otherwise we need a SIB byte, so start by outputting the ModR/M byte first.
  assert(IndexReg.getReg() != X86::ESP &&
         IndexReg.getReg() != X86::RSP && "Cannot use ESP as index reg!");

  bool ForceDisp32 = false;
  bool ForceDisp8  = false;
  if (BaseReg == 0) {
    // If there is no base register, we emit the special case SIB byte with
    // MOD=0, BASE=4, to JUST get the index, scale, and displacement.
    MCE.emitByte(ModRMByte(0, RegOpcodeField, 4));
    ForceDisp32 = true;
  } else if (DispForReloc) {
    // Emit the normal disp32 encoding.
    MCE.emitByte(ModRMByte(2, RegOpcodeField, 4));
    ForceDisp32 = true;
  } else if (DispVal == 0 && BaseRegNo != N86::EBP) {
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
  static const unsigned SSTable[] = { ~0U, 0, 1, ~0U, 2, ~0U, ~0U, ~0U, 3 };
  unsigned SS = SSTable[Scale.getImm()];

  if (BaseReg == 0) {
    // Handle the SIB byte for the case where there is no base, see Intel
    // Manual 2A, table 2-7. The displacement has already been output.
    unsigned IndexRegNo;
    if (IndexReg.getReg())
      IndexRegNo = getX86RegNum(IndexReg.getReg());
    else // Examples: [ESP+1*<noreg>+4] or [scaled idx]+disp32 (MOD=0,BASE=5)
      IndexRegNo = 4;
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
    emitDisplacementField(DispForReloc, DispVal, PCAdj, IsPCRel);
  }
}

static const MCInstrDesc *UpdateOp(MachineInstr &MI, const X86InstrInfo *II,
                                   unsigned Opcode) {
  const MCInstrDesc *Desc = &II->get(Opcode);
  MI.setDesc(*Desc);
  return Desc;
}

/// Is16BitMemOperand - Return true if the specified instruction has
/// a 16-bit memory operand. Op specifies the operand # of the memoperand.
static bool Is16BitMemOperand(const MachineInstr &MI, unsigned Op) {
  const MachineOperand &BaseReg  = MI.getOperand(Op+X86::AddrBaseReg);
  const MachineOperand &IndexReg = MI.getOperand(Op+X86::AddrIndexReg);

  if ((BaseReg.getReg() != 0 &&
       X86MCRegisterClasses[X86::GR16RegClassID].contains(BaseReg.getReg())) ||
      (IndexReg.getReg() != 0 &&
       X86MCRegisterClasses[X86::GR16RegClassID].contains(IndexReg.getReg())))
    return true;
  return false;
}

/// Is32BitMemOperand - Return true if the specified instruction has
/// a 32-bit memory operand. Op specifies the operand # of the memoperand.
static bool Is32BitMemOperand(const MachineInstr &MI, unsigned Op) {
  const MachineOperand &BaseReg  = MI.getOperand(Op+X86::AddrBaseReg);
  const MachineOperand &IndexReg = MI.getOperand(Op+X86::AddrIndexReg);

  if ((BaseReg.getReg() != 0 &&
       X86MCRegisterClasses[X86::GR32RegClassID].contains(BaseReg.getReg())) ||
      (IndexReg.getReg() != 0 &&
       X86MCRegisterClasses[X86::GR32RegClassID].contains(IndexReg.getReg())))
    return true;
  return false;
}

/// Is64BitMemOperand - Return true if the specified instruction has
/// a 64-bit memory operand. Op specifies the operand # of the memoperand.
#ifndef NDEBUG
static bool Is64BitMemOperand(const MachineInstr &MI, unsigned Op) {
  const MachineOperand &BaseReg  = MI.getOperand(Op+X86::AddrBaseReg);
  const MachineOperand &IndexReg = MI.getOperand(Op+X86::AddrIndexReg);

  if ((BaseReg.getReg() != 0 &&
       X86MCRegisterClasses[X86::GR64RegClassID].contains(BaseReg.getReg())) ||
      (IndexReg.getReg() != 0 &&
       X86MCRegisterClasses[X86::GR64RegClassID].contains(IndexReg.getReg())))
    return true;
  return false;
}
#endif

template<class CodeEmitter>
void Emitter<CodeEmitter>::emitOpcodePrefix(uint64_t TSFlags,
                                            int MemOperand,
                                            const MachineInstr &MI,
                                            const MCInstrDesc *Desc) const {
  // Emit the lock opcode prefix as needed.
  if (Desc->TSFlags & X86II::LOCK)
    MCE.emitByte(0xF0);

  // Emit segment override opcode prefix as needed.
  emitSegmentOverridePrefix(TSFlags, MemOperand, MI);

  // Emit the repeat opcode prefix as needed.
  if ((Desc->TSFlags & X86II::Op0Mask) == X86II::REP)
    MCE.emitByte(0xF3);

  // Emit the address size opcode prefix as needed.
  bool need_address_override;
  if (TSFlags & X86II::AdSize) {
    need_address_override = true;
  } else if (MemOperand == -1) {
    need_address_override = false;
  } else if (Is64BitMode) {
    assert(!Is16BitMemOperand(MI, MemOperand));
    need_address_override = Is32BitMemOperand(MI, MemOperand);
  } else {
    assert(!Is64BitMemOperand(MI, MemOperand));
    need_address_override = Is16BitMemOperand(MI, MemOperand);
  }

  if (need_address_override)
    MCE.emitByte(0x67);

  // Emit the operand size opcode prefix as needed.
  if (TSFlags & X86II::OpSize)
    MCE.emitByte(0x66);

  bool Need0FPrefix = false;
  switch (Desc->TSFlags & X86II::Op0Mask) {
    case X86II::TB:  // Two-byte opcode prefix
    case X86II::T8:  // 0F 38
    case X86II::TA:  // 0F 3A
    case X86II::A6:  // 0F A6
    case X86II::A7:  // 0F A7
      Need0FPrefix = true;
      break;
    case X86II::REP: break; // already handled.
    case X86II::T8XS: // F3 0F 38
    case X86II::XS:   // F3 0F
      MCE.emitByte(0xF3);
      Need0FPrefix = true;
      break;
    case X86II::T8XD: // F2 0F 38
    case X86II::TAXD: // F2 0F 3A
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
    default: llvm_unreachable("Invalid prefix!");
    case 0: break;  // No prefix!
  }

  // Handle REX prefix.
  if (Is64BitMode) {
    if (unsigned REX = determineREX(MI))
      MCE.emitByte(0x40 | REX);
  }

  // 0x0F escape code must be emitted just before the opcode.
  if (Need0FPrefix)
    MCE.emitByte(0x0F);

  switch (Desc->TSFlags & X86II::Op0Mask) {
    case X86II::T8XD:  // F2 0F 38
    case X86II::T8XS:  // F3 0F 38
    case X86II::T8:    // 0F 38
      MCE.emitByte(0x38);
      break;
    case X86II::TAXD:  // F2 0F 38
    case X86II::TA:    // 0F 3A
      MCE.emitByte(0x3A);
      break;
    case X86II::A6:    // 0F A6
      MCE.emitByte(0xA6);
      break;
    case X86II::A7:    // 0F A7
      MCE.emitByte(0xA7);
      break;
  }
}

// On regular x86, both XMM0-XMM7 and XMM8-XMM15 are encoded in the range
// 0-7 and the difference between the 2 groups is given by the REX prefix.
// In the VEX prefix, registers are seen sequencially from 0-15 and encoded
// in 1's complement form, example:
//
//  ModRM field => XMM9 => 1
//  VEX.VVVV    => XMM9 => ~9
//
// See table 4-35 of Intel AVX Programming Reference for details.
template<class CodeEmitter>
unsigned char
Emitter<CodeEmitter>::getVEXRegisterEncoding(const MachineInstr &MI,
                                             unsigned OpNum) const {
  unsigned SrcReg = MI.getOperand(OpNum).getReg();
  unsigned SrcRegNum = getX86RegNum(MI.getOperand(OpNum).getReg());
  if (X86II::isX86_64ExtendedReg(SrcReg))
    SrcRegNum |= 8;

  // The registers represented through VEX_VVVV should
  // be encoded in 1's complement form.
  return (~SrcRegNum) & 0xf;
}

/// EmitSegmentOverridePrefix - Emit segment override opcode prefix as needed
template<class CodeEmitter>
void Emitter<CodeEmitter>::emitSegmentOverridePrefix(uint64_t TSFlags,
                                                 int MemOperand,
                                                 const MachineInstr &MI) const {
  switch (TSFlags & X86II::SegOvrMask) {
    default: llvm_unreachable("Invalid segment!");
    case 0:
      // No segment override, check for explicit one on memory operand.
      if (MemOperand != -1) {   // If the instruction has a memory operand.
        switch (MI.getOperand(MemOperand+X86::AddrSegmentReg).getReg()) {
          default: llvm_unreachable("Unknown segment register!");
          case 0: break;
          case X86::CS: MCE.emitByte(0x2E); break;
          case X86::SS: MCE.emitByte(0x36); break;
          case X86::DS: MCE.emitByte(0x3E); break;
          case X86::ES: MCE.emitByte(0x26); break;
          case X86::FS: MCE.emitByte(0x64); break;
          case X86::GS: MCE.emitByte(0x65); break;
        }
      }
      break;
    case X86II::FS:
      MCE.emitByte(0x64);
      break;
    case X86II::GS:
      MCE.emitByte(0x65);
      break;
  }
}

template<class CodeEmitter>
void Emitter<CodeEmitter>::emitVEXOpcodePrefix(uint64_t TSFlags,
                                               int MemOperand,
                                               const MachineInstr &MI,
                                               const MCInstrDesc *Desc) const {
  bool HasVEX_4V = (TSFlags >> X86II::VEXShift) & X86II::VEX_4V;
  bool HasVEX_4VOp3 = (TSFlags >> X86II::VEXShift) & X86II::VEX_4VOp3;

  // VEX_R: opcode externsion equivalent to REX.R in
  // 1's complement (inverted) form
  //
  //  1: Same as REX_R=0 (must be 1 in 32-bit mode)
  //  0: Same as REX_R=1 (64 bit mode only)
  //
  unsigned char VEX_R = 0x1;

  // VEX_X: equivalent to REX.X, only used when a
  // register is used for index in SIB Byte.
  //
  //  1: Same as REX.X=0 (must be 1 in 32-bit mode)
  //  0: Same as REX.X=1 (64-bit mode only)
  unsigned char VEX_X = 0x1;

  // VEX_B:
  //
  //  1: Same as REX_B=0 (ignored in 32-bit mode)
  //  0: Same as REX_B=1 (64 bit mode only)
  //
  unsigned char VEX_B = 0x1;

  // VEX_W: opcode specific (use like REX.W, or used for
  // opcode extension, or ignored, depending on the opcode byte)
  unsigned char VEX_W = 0;

  // XOP: Use XOP prefix byte 0x8f instead of VEX.
  unsigned char XOP = 0;

  // VEX_5M (VEX m-mmmmm field):
  //
  //  0b00000: Reserved for future use
  //  0b00001: implied 0F leading opcode
  //  0b00010: implied 0F 38 leading opcode bytes
  //  0b00011: implied 0F 3A leading opcode bytes
  //  0b00100-0b11111: Reserved for future use
  //  0b01000: XOP map select - 08h instructions with imm byte
  //  0b10001: XOP map select - 09h instructions with no imm byte
  unsigned char VEX_5M = 0x1;

  // VEX_4V (VEX vvvv field): a register specifier
  // (in 1's complement form) or 1111 if unused.
  unsigned char VEX_4V = 0xf;

  // VEX_L (Vector Length):
  //
  //  0: scalar or 128-bit vector
  //  1: 256-bit vector
  //
  unsigned char VEX_L = 0;

  // VEX_PP: opcode extension providing equivalent
  // functionality of a SIMD prefix
  //
  //  0b00: None
  //  0b01: 66
  //  0b10: F3
  //  0b11: F2
  //
  unsigned char VEX_PP = 0;

  // Encode the operand size opcode prefix as needed.
  if (TSFlags & X86II::OpSize)
    VEX_PP = 0x01;

  if ((TSFlags >> X86II::VEXShift) & X86II::VEX_W)
    VEX_W = 1;

  if ((TSFlags >> X86II::VEXShift) & X86II::XOP)
    XOP = 1;

  if ((TSFlags >> X86II::VEXShift) & X86II::VEX_L)
    VEX_L = 1;

  switch (TSFlags & X86II::Op0Mask) {
    default: llvm_unreachable("Invalid prefix!");
    case X86II::T8:  // 0F 38
      VEX_5M = 0x2;
      break;
    case X86II::TA:  // 0F 3A
      VEX_5M = 0x3;
      break;
    case X86II::T8XS: // F3 0F 38
      VEX_PP = 0x2;
      VEX_5M = 0x2;
      break;
    case X86II::T8XD: // F2 0F 38
      VEX_PP = 0x3;
      VEX_5M = 0x2;
      break;
    case X86II::TAXD: // F2 0F 3A
      VEX_PP = 0x3;
      VEX_5M = 0x3;
      break;
    case X86II::XS:  // F3 0F
      VEX_PP = 0x2;
      break;
    case X86II::XD:  // F2 0F
      VEX_PP = 0x3;
      break;
    case X86II::XOP8:
      VEX_5M = 0x8;
      break;
    case X86II::XOP9:
      VEX_5M = 0x9;
      break;
    case X86II::A6:  // Bypass: Not used by VEX
    case X86II::A7:  // Bypass: Not used by VEX
    case X86II::TB:  // Bypass: Not used by VEX
    case 0:
      break;  // No prefix!
  }


  // Classify VEX_B, VEX_4V, VEX_R, VEX_X
  unsigned NumOps = Desc->getNumOperands();
  unsigned CurOp = 0;
  if (NumOps > 1 && Desc->getOperandConstraint(1, MCOI::TIED_TO) == 0)
    ++CurOp;
  else if (NumOps > 3 && Desc->getOperandConstraint(2, MCOI::TIED_TO) == 0) {
    assert(Desc->getOperandConstraint(NumOps - 1, MCOI::TIED_TO) == 1);
    // Special case for GATHER with 2 TIED_TO operands
    // Skip the first 2 operands: dst, mask_wb
    CurOp += 2;
  }

  switch (TSFlags & X86II::FormMask) {
    case X86II::MRMInitReg:
      // Duplicate register.
      if (X86II::isX86_64ExtendedReg(MI.getOperand(CurOp).getReg()))
        VEX_R = 0x0;

      if (HasVEX_4V)
        VEX_4V = getVEXRegisterEncoding(MI, CurOp);
      if (X86II::isX86_64ExtendedReg(MI.getOperand(CurOp).getReg()))
        VEX_B = 0x0;
      if (HasVEX_4VOp3)
        VEX_4V = getVEXRegisterEncoding(MI, CurOp);
      break;
    case X86II::MRMDestMem: {
      // MRMDestMem instructions forms:
      //  MemAddr, src1(ModR/M)
      //  MemAddr, src1(VEX_4V), src2(ModR/M)
      //  MemAddr, src1(ModR/M), imm8
      //
      if (X86II::isX86_64ExtendedReg(MI.getOperand(X86::AddrBaseReg).getReg()))
        VEX_B = 0x0;
      if (X86II::isX86_64ExtendedReg(MI.getOperand(X86::AddrIndexReg).getReg()))
        VEX_X = 0x0;

      CurOp = X86::AddrNumOperands;
      if (HasVEX_4V)
        VEX_4V = getVEXRegisterEncoding(MI, CurOp++);

      const MachineOperand &MO = MI.getOperand(CurOp);
      if (MO.isReg() && X86II::isX86_64ExtendedReg(MO.getReg()))
        VEX_R = 0x0;
      break;
    }
    case X86II::MRMSrcMem:
      // MRMSrcMem instructions forms:
      //  src1(ModR/M), MemAddr
      //  src1(ModR/M), src2(VEX_4V), MemAddr
      //  src1(ModR/M), MemAddr, imm8
      //  src1(ModR/M), MemAddr, src2(VEX_I8IMM)
      //
      //  FMA4:
      //  dst(ModR/M.reg), src1(VEX_4V), src2(ModR/M), src3(VEX_I8IMM)
      //  dst(ModR/M.reg), src1(VEX_4V), src2(VEX_I8IMM), src3(ModR/M),
      if (X86II::isX86_64ExtendedReg(MI.getOperand(0).getReg()))
        VEX_R = 0x0;

      if (HasVEX_4V)
        VEX_4V = getVEXRegisterEncoding(MI, 1);

      if (X86II::isX86_64ExtendedReg(
                          MI.getOperand(MemOperand+X86::AddrBaseReg).getReg()))
        VEX_B = 0x0;
      if (X86II::isX86_64ExtendedReg(
                          MI.getOperand(MemOperand+X86::AddrIndexReg).getReg()))
        VEX_X = 0x0;

      if (HasVEX_4VOp3)
        VEX_4V = getVEXRegisterEncoding(MI, X86::AddrNumOperands+1);
      break;
    case X86II::MRM0m: case X86II::MRM1m:
    case X86II::MRM2m: case X86II::MRM3m:
    case X86II::MRM4m: case X86II::MRM5m:
    case X86II::MRM6m: case X86II::MRM7m: {
      // MRM[0-9]m instructions forms:
      //  MemAddr
      //  src1(VEX_4V), MemAddr
      if (HasVEX_4V)
        VEX_4V = getVEXRegisterEncoding(MI, 0);

      if (X86II::isX86_64ExtendedReg(
                          MI.getOperand(MemOperand+X86::AddrBaseReg).getReg()))
        VEX_B = 0x0;
      if (X86II::isX86_64ExtendedReg(
                          MI.getOperand(MemOperand+X86::AddrIndexReg).getReg()))
        VEX_X = 0x0;
      break;
    }
    case X86II::MRMSrcReg:
      // MRMSrcReg instructions forms:
      //  dst(ModR/M), src1(VEX_4V), src2(ModR/M), src3(VEX_I8IMM)
      //  dst(ModR/M), src1(ModR/M)
      //  dst(ModR/M), src1(ModR/M), imm8
      //
      if (X86II::isX86_64ExtendedReg(MI.getOperand(CurOp).getReg()))
        VEX_R = 0x0;
      CurOp++;

      if (HasVEX_4V)
        VEX_4V = getVEXRegisterEncoding(MI, CurOp++);
      if (X86II::isX86_64ExtendedReg(MI.getOperand(CurOp).getReg()))
        VEX_B = 0x0;
      CurOp++;
      if (HasVEX_4VOp3)
        VEX_4V = getVEXRegisterEncoding(MI, CurOp);
      break;
    case X86II::MRMDestReg:
      // MRMDestReg instructions forms:
      //  dst(ModR/M), src(ModR/M)
      //  dst(ModR/M), src(ModR/M), imm8
      if (X86II::isX86_64ExtendedReg(MI.getOperand(0).getReg()))
        VEX_B = 0x0;
      if (X86II::isX86_64ExtendedReg(MI.getOperand(1).getReg()))
        VEX_R = 0x0;
      break;
    case X86II::MRM0r: case X86II::MRM1r:
    case X86II::MRM2r: case X86II::MRM3r:
    case X86II::MRM4r: case X86II::MRM5r:
    case X86II::MRM6r: case X86II::MRM7r:
      // MRM0r-MRM7r instructions forms:
      //  dst(VEX_4V), src(ModR/M), imm8
      VEX_4V = getVEXRegisterEncoding(MI, 0);
      if (X86II::isX86_64ExtendedReg(MI.getOperand(1).getReg()))
        VEX_B = 0x0;
      break;
    default: // RawFrm
      break;
  }

  // Emit segment override opcode prefix as needed.
  emitSegmentOverridePrefix(TSFlags, MemOperand, MI);

  // VEX opcode prefix can have 2 or 3 bytes
  //
  //  3 bytes:
  //    +-----+ +--------------+ +-------------------+
  //    | C4h | | RXB | m-mmmm | | W | vvvv | L | pp |
  //    +-----+ +--------------+ +-------------------+
  //  2 bytes:
  //    +-----+ +-------------------+
  //    | C5h | | R | vvvv | L | pp |
  //    +-----+ +-------------------+
  //
  unsigned char LastByte = VEX_PP | (VEX_L << 2) | (VEX_4V << 3);

  if (VEX_B && VEX_X && !VEX_W && !XOP && (VEX_5M == 1)) { // 2 byte VEX prefix
    MCE.emitByte(0xC5);
    MCE.emitByte(LastByte | (VEX_R << 7));
    return;
  }

  // 3 byte VEX prefix
  MCE.emitByte(XOP ? 0x8F : 0xC4);
  MCE.emitByte(VEX_R << 7 | VEX_X << 6 | VEX_B << 5 | VEX_5M);
  MCE.emitByte(LastByte | (VEX_W << 7));
}

template<class CodeEmitter>
void Emitter<CodeEmitter>::emitInstruction(MachineInstr &MI,
                                           const MCInstrDesc *Desc) {
  DEBUG(dbgs() << MI);

  // If this is a pseudo instruction, lower it.
  switch (Desc->getOpcode()) {
  case X86::ADD16rr_DB:      Desc = UpdateOp(MI, II, X86::OR16rr); break;
  case X86::ADD32rr_DB:      Desc = UpdateOp(MI, II, X86::OR32rr); break;
  case X86::ADD64rr_DB:      Desc = UpdateOp(MI, II, X86::OR64rr); break;
  case X86::ADD16ri_DB:      Desc = UpdateOp(MI, II, X86::OR16ri); break;
  case X86::ADD32ri_DB:      Desc = UpdateOp(MI, II, X86::OR32ri); break;
  case X86::ADD64ri32_DB:    Desc = UpdateOp(MI, II, X86::OR64ri32); break;
  case X86::ADD16ri8_DB:     Desc = UpdateOp(MI, II, X86::OR16ri8); break;
  case X86::ADD32ri8_DB:     Desc = UpdateOp(MI, II, X86::OR32ri8); break;
  case X86::ADD64ri8_DB:     Desc = UpdateOp(MI, II, X86::OR64ri8); break;
  case X86::ACQUIRE_MOV8rm:  Desc = UpdateOp(MI, II, X86::MOV8rm); break;
  case X86::ACQUIRE_MOV16rm: Desc = UpdateOp(MI, II, X86::MOV16rm); break;
  case X86::ACQUIRE_MOV32rm: Desc = UpdateOp(MI, II, X86::MOV32rm); break;
  case X86::ACQUIRE_MOV64rm: Desc = UpdateOp(MI, II, X86::MOV64rm); break;
  case X86::RELEASE_MOV8mr:  Desc = UpdateOp(MI, II, X86::MOV8mr); break;
  case X86::RELEASE_MOV16mr: Desc = UpdateOp(MI, II, X86::MOV16mr); break;
  case X86::RELEASE_MOV32mr: Desc = UpdateOp(MI, II, X86::MOV32mr); break;
  case X86::RELEASE_MOV64mr: Desc = UpdateOp(MI, II, X86::MOV64mr); break;
  }


  MCE.processDebugLoc(MI.getDebugLoc(), true);

  unsigned Opcode = Desc->Opcode;

  // If this is a two-address instruction, skip one of the register operands.
  unsigned NumOps = Desc->getNumOperands();
  unsigned CurOp = 0;
  if (NumOps > 1 && Desc->getOperandConstraint(1, MCOI::TIED_TO) == 0)
    ++CurOp;
  else if (NumOps > 3 && Desc->getOperandConstraint(2, MCOI::TIED_TO) == 0) {
    assert(Desc->getOperandConstraint(NumOps - 1, MCOI::TIED_TO) == 1);
    // Special case for GATHER with 2 TIED_TO operands
    // Skip the first 2 operands: dst, mask_wb
    CurOp += 2;
  }

  uint64_t TSFlags = Desc->TSFlags;

  // Is this instruction encoded using the AVX VEX prefix?
  bool HasVEXPrefix = (TSFlags >> X86II::VEXShift) & X86II::VEX;
  // It uses the VEX.VVVV field?
  bool HasVEX_4V = (TSFlags >> X86II::VEXShift) & X86II::VEX_4V;
  bool HasVEX_4VOp3 = (TSFlags >> X86II::VEXShift) & X86II::VEX_4VOp3;
  bool HasMemOp4 = (TSFlags >> X86II::VEXShift) & X86II::MemOp4;
  const unsigned MemOp4_I8IMMOperand = 2;

  // Determine where the memory operand starts, if present.
  int MemoryOperand = X86II::getMemoryOperandNo(TSFlags, Opcode);
  if (MemoryOperand != -1) MemoryOperand += CurOp;

  if (!HasVEXPrefix)
    emitOpcodePrefix(TSFlags, MemoryOperand, MI, Desc);
  else
    emitVEXOpcodePrefix(TSFlags, MemoryOperand, MI, Desc);

  unsigned char BaseOpcode = X86II::getBaseOpcodeFor(Desc->TSFlags);
  switch (TSFlags & X86II::FormMask) {
  default:
    llvm_unreachable("Unknown FormMask value in X86 MachineCodeEmitter!");
  case X86II::Pseudo:
    // Remember the current PC offset, this is the PIC relocation
    // base address.
    switch (Opcode) {
    default:
      llvm_unreachable("pseudo instructions should be removed before code"
                       " emission");
    // Do nothing for Int_MemBarrier - it's just a comment.  Add a debug
    // to make it slightly easier to see.
    case X86::Int_MemBarrier:
      DEBUG(dbgs() << "#MEMBARRIER\n");
      break;

    case TargetOpcode::INLINEASM:
      // We allow inline assembler nodes with empty bodies - they can
      // implicitly define registers, which is ok for JIT.
      if (MI.getOperand(0).getSymbolName()[0])
        report_fatal_error("JIT does not support inline asm!");
      break;
    case TargetOpcode::PROLOG_LABEL:
    case TargetOpcode::GC_LABEL:
    case TargetOpcode::EH_LABEL:
      MCE.emitLabel(MI.getOperand(0).getMCSymbol());
      break;

    case TargetOpcode::IMPLICIT_DEF:
    case TargetOpcode::KILL:
      break;
    case X86::MOVPC32r: {
      // This emits the "call" portion of this pseudo instruction.
      MCE.emitByte(BaseOpcode);
      emitConstant(0, X86II::getSizeOfImm(Desc->TSFlags));
      // Remember PIC base.
      PICBaseOffset = (intptr_t) MCE.getCurrentPCOffset();
      X86JITInfo *JTI = TM.getJITInfo();
      JTI->setPICBase(MCE.getCurrentPCValue());
      break;
    }
    }
    CurOp = NumOps;
    break;
  case X86II::RawFrm: {
    MCE.emitByte(BaseOpcode);

    if (CurOp == NumOps)
      break;

    const MachineOperand &MO = MI.getOperand(CurOp++);

    DEBUG(dbgs() << "RawFrm CurOp " << CurOp << "\n");
    DEBUG(dbgs() << "isMBB " << MO.isMBB() << "\n");
    DEBUG(dbgs() << "isGlobal " << MO.isGlobal() << "\n");
    DEBUG(dbgs() << "isSymbol " << MO.isSymbol() << "\n");
    DEBUG(dbgs() << "isImm " << MO.isImm() << "\n");

    if (MO.isMBB()) {
      emitPCRelativeBlockAddress(MO.getMBB());
      break;
    }

    if (MO.isGlobal()) {
      emitGlobalAddress(MO.getGlobal(), X86::reloc_pcrel_word,
                        MO.getOffset(), 0);
      break;
    }

    if (MO.isSymbol()) {
      emitExternalSymbolAddress(MO.getSymbolName(), X86::reloc_pcrel_word);
      break;
    }

    // FIXME: Only used by hackish MCCodeEmitter, remove when dead.
    if (MO.isJTI()) {
      emitJumpTableAddress(MO.getIndex(), X86::reloc_pcrel_word);
      break;
    }

    assert(MO.isImm() && "Unknown RawFrm operand!");
    if (Opcode == X86::CALLpcrel32 || Opcode == X86::CALL64pcrel32) {
      // Fix up immediate operand for pc relative calls.
      intptr_t Imm = (intptr_t)MO.getImm();
      Imm = Imm - MCE.getCurrentPCValue() - 4;
      emitConstant(Imm, X86II::getSizeOfImm(Desc->TSFlags));
    } else
      emitConstant(MO.getImm(), X86II::getSizeOfImm(Desc->TSFlags));
    break;
  }

  case X86II::AddRegFrm: {
    MCE.emitByte(BaseOpcode +
                 getX86RegNum(MI.getOperand(CurOp++).getReg()));

    if (CurOp == NumOps)
      break;

    const MachineOperand &MO1 = MI.getOperand(CurOp++);
    unsigned Size = X86II::getSizeOfImm(Desc->TSFlags);
    if (MO1.isImm()) {
      emitConstant(MO1.getImm(), Size);
      break;
    }

    unsigned rt = Is64BitMode ? X86::reloc_pcrel_word
      : (IsPIC ? X86::reloc_picrel_word : X86::reloc_absolute_word);
    if (Opcode == X86::MOV64ri64i32)
      rt = X86::reloc_absolute_word;  // FIXME: add X86II flag?
    // This should not occur on Darwin for relocatable objects.
    if (Opcode == X86::MOV64ri)
      rt = X86::reloc_absolute_dword;  // FIXME: add X86II flag?
    if (MO1.isGlobal()) {
      bool Indirect = gvNeedsNonLazyPtr(MO1, TM);
      emitGlobalAddress(MO1.getGlobal(), rt, MO1.getOffset(), 0,
                        Indirect);
    } else if (MO1.isSymbol())
      emitExternalSymbolAddress(MO1.getSymbolName(), rt);
    else if (MO1.isCPI())
      emitConstPoolAddress(MO1.getIndex(), rt);
    else if (MO1.isJTI())
      emitJumpTableAddress(MO1.getIndex(), rt);
    break;
  }

  case X86II::MRMDestReg: {
    MCE.emitByte(BaseOpcode);
    emitRegModRMByte(MI.getOperand(CurOp).getReg(),
                     getX86RegNum(MI.getOperand(CurOp+1).getReg()));
    CurOp += 2;
    break;
  }
  case X86II::MRMDestMem: {
    MCE.emitByte(BaseOpcode);

    unsigned SrcRegNum = CurOp + X86::AddrNumOperands;
    if (HasVEX_4V) // Skip 1st src (which is encoded in VEX_VVVV)
      SrcRegNum++;
    emitMemModRMByte(MI, CurOp,
                     getX86RegNum(MI.getOperand(SrcRegNum).getReg()));
    CurOp = SrcRegNum + 1;
    break;
  }

  case X86II::MRMSrcReg: {
    MCE.emitByte(BaseOpcode);

    unsigned SrcRegNum = CurOp+1;
    if (HasVEX_4V) // Skip 1st src (which is encoded in VEX_VVVV)
      ++SrcRegNum;

    if (HasMemOp4) // Skip 2nd src (which is encoded in I8IMM)
      ++SrcRegNum;

    emitRegModRMByte(MI.getOperand(SrcRegNum).getReg(),
                     getX86RegNum(MI.getOperand(CurOp).getReg()));
    // 2 operands skipped with HasMemOp4, compensate accordingly
    CurOp = HasMemOp4 ? SrcRegNum : SrcRegNum + 1;
    if (HasVEX_4VOp3)
      ++CurOp;
    break;
  }
  case X86II::MRMSrcMem: {
    int AddrOperands = X86::AddrNumOperands;
    unsigned FirstMemOp = CurOp+1;
    if (HasVEX_4V) {
      ++AddrOperands;
      ++FirstMemOp;  // Skip the register source (which is encoded in VEX_VVVV).
    }
    if (HasMemOp4) // Skip second register source (encoded in I8IMM)
      ++FirstMemOp;

    MCE.emitByte(BaseOpcode);

    intptr_t PCAdj = (CurOp + AddrOperands + 1 != NumOps) ?
      X86II::getSizeOfImm(Desc->TSFlags) : 0;
    emitMemModRMByte(MI, FirstMemOp,
                     getX86RegNum(MI.getOperand(CurOp).getReg()),PCAdj);
    CurOp += AddrOperands + 1;
    if (HasVEX_4VOp3)
      ++CurOp;
    break;
  }

  case X86II::MRM0r: case X86II::MRM1r:
  case X86II::MRM2r: case X86II::MRM3r:
  case X86II::MRM4r: case X86II::MRM5r:
  case X86II::MRM6r: case X86II::MRM7r: {
    if (HasVEX_4V) // Skip the register dst (which is encoded in VEX_VVVV).
      ++CurOp;
    MCE.emitByte(BaseOpcode);
    emitRegModRMByte(MI.getOperand(CurOp++).getReg(),
                     (Desc->TSFlags & X86II::FormMask)-X86II::MRM0r);

    if (CurOp == NumOps)
      break;

    const MachineOperand &MO1 = MI.getOperand(CurOp++);
    unsigned Size = X86II::getSizeOfImm(Desc->TSFlags);
    if (MO1.isImm()) {
      emitConstant(MO1.getImm(), Size);
      break;
    }

    unsigned rt = Is64BitMode ? X86::reloc_pcrel_word
      : (IsPIC ? X86::reloc_picrel_word : X86::reloc_absolute_word);
    if (Opcode == X86::MOV64ri32)
      rt = X86::reloc_absolute_word_sext;  // FIXME: add X86II flag?
    if (MO1.isGlobal()) {
      bool Indirect = gvNeedsNonLazyPtr(MO1, TM);
      emitGlobalAddress(MO1.getGlobal(), rt, MO1.getOffset(), 0,
                        Indirect);
    } else if (MO1.isSymbol())
      emitExternalSymbolAddress(MO1.getSymbolName(), rt);
    else if (MO1.isCPI())
      emitConstPoolAddress(MO1.getIndex(), rt);
    else if (MO1.isJTI())
      emitJumpTableAddress(MO1.getIndex(), rt);
    break;
  }

  case X86II::MRM0m: case X86II::MRM1m:
  case X86II::MRM2m: case X86II::MRM3m:
  case X86II::MRM4m: case X86II::MRM5m:
  case X86II::MRM6m: case X86II::MRM7m: {
    if (HasVEX_4V) // Skip the register dst (which is encoded in VEX_VVVV).
      ++CurOp;
    intptr_t PCAdj = (CurOp + X86::AddrNumOperands != NumOps) ?
      (MI.getOperand(CurOp+X86::AddrNumOperands).isImm() ?
          X86II::getSizeOfImm(Desc->TSFlags) : 4) : 0;

    MCE.emitByte(BaseOpcode);
    emitMemModRMByte(MI, CurOp, (Desc->TSFlags & X86II::FormMask)-X86II::MRM0m,
                     PCAdj);
    CurOp += X86::AddrNumOperands;

    if (CurOp == NumOps)
      break;

    const MachineOperand &MO = MI.getOperand(CurOp++);
    unsigned Size = X86II::getSizeOfImm(Desc->TSFlags);
    if (MO.isImm()) {
      emitConstant(MO.getImm(), Size);
      break;
    }

    unsigned rt = Is64BitMode ? X86::reloc_pcrel_word
      : (IsPIC ? X86::reloc_picrel_word : X86::reloc_absolute_word);
    if (Opcode == X86::MOV64mi32)
      rt = X86::reloc_absolute_word_sext;  // FIXME: add X86II flag?
    if (MO.isGlobal()) {
      bool Indirect = gvNeedsNonLazyPtr(MO, TM);
      emitGlobalAddress(MO.getGlobal(), rt, MO.getOffset(), 0,
                        Indirect);
    } else if (MO.isSymbol())
      emitExternalSymbolAddress(MO.getSymbolName(), rt);
    else if (MO.isCPI())
      emitConstPoolAddress(MO.getIndex(), rt);
    else if (MO.isJTI())
      emitJumpTableAddress(MO.getIndex(), rt);
    break;
  }

  case X86II::MRMInitReg:
    MCE.emitByte(BaseOpcode);
    // Duplicate register, used by things like MOV8r0 (aka xor reg,reg).
    emitRegModRMByte(MI.getOperand(CurOp).getReg(),
                     getX86RegNum(MI.getOperand(CurOp).getReg()));
    ++CurOp;
    break;

  case X86II::MRM_C1:
    MCE.emitByte(BaseOpcode);
    MCE.emitByte(0xC1);
    break;
  case X86II::MRM_C8:
    MCE.emitByte(BaseOpcode);
    MCE.emitByte(0xC8);
    break;
  case X86II::MRM_C9:
    MCE.emitByte(BaseOpcode);
    MCE.emitByte(0xC9);
    break;
  case X86II::MRM_E8:
    MCE.emitByte(BaseOpcode);
    MCE.emitByte(0xE8);
    break;
  case X86II::MRM_F0:
    MCE.emitByte(BaseOpcode);
    MCE.emitByte(0xF0);
    break;
  }

  while (CurOp != NumOps && NumOps - CurOp <= 2) {
    // The last source register of a 4 operand instruction in AVX is encoded
    // in bits[7:4] of a immediate byte.
    if ((TSFlags >> X86II::VEXShift) & X86II::VEX_I8IMM) {
      const MachineOperand &MO = MI.getOperand(HasMemOp4 ? MemOp4_I8IMMOperand
                                                         : CurOp);
      ++CurOp;
      unsigned RegNum = getX86RegNum(MO.getReg()) << 4;
      if (X86II::isX86_64ExtendedReg(MO.getReg()))
        RegNum |= 1 << 7;
      // If there is an additional 5th operand it must be an immediate, which
      // is encoded in bits[3:0]
      if (CurOp != NumOps) {
        const MachineOperand &MIMM = MI.getOperand(CurOp++);
        if (MIMM.isImm()) {
          unsigned Val = MIMM.getImm();
          assert(Val < 16 && "Immediate operand value out of range");
          RegNum |= Val;
        }
      }
      emitConstant(RegNum, 1);
    } else {
      emitConstant(MI.getOperand(CurOp++).getImm(),
                   X86II::getSizeOfImm(Desc->TSFlags));
    }
  }

  if (!MI.isVariadic() && CurOp != NumOps) {
#ifndef NDEBUG
    dbgs() << "Cannot encode all operands of: " << MI << "\n";
#endif
    llvm_unreachable(0);
  }

  MCE.processDebugLoc(MI.getDebugLoc(), false);
}
