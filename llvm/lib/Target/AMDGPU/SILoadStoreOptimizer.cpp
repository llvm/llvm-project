//===- SILoadStoreOptimizer.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass tries to fuse DS instructions with close by immediate offsets.
// This will fuse operations such as
//  ds_read_b32 v0, v2 offset:16
//  ds_read_b32 v1, v2 offset:32
// ==>
//   ds_read2_b32 v[0:1], v2, offset0:4 offset1:8
//
// The same is done for certain SMEM and VMEM opcodes, e.g.:
//  s_buffer_load_dword s4, s[0:3], 4
//  s_buffer_load_dword s5, s[0:3], 8
// ==>
//  s_buffer_load_dwordx2 s[4:5], s[0:3], 4
//
// This pass also tries to promote constant offset to the immediate by
// adjusting the base. It tries to use a base from the nearby instructions that
// allows it to have a 13bit constant offset and then promotes the 13bit offset
// to the immediate.
// E.g.
//  s_movk_i32 s0, 0x1800
//  v_add_co_u32_e32 v0, vcc, s0, v2
//  v_addc_co_u32_e32 v1, vcc, 0, v6, vcc
//
//  s_movk_i32 s0, 0x1000
//  v_add_co_u32_e32 v5, vcc, s0, v2
//  v_addc_co_u32_e32 v6, vcc, 0, v6, vcc
//  global_load_dwordx2 v[5:6], v[5:6], off
//  global_load_dwordx2 v[0:1], v[0:1], off
// =>
//  s_movk_i32 s0, 0x1000
//  v_add_co_u32_e32 v5, vcc, s0, v2
//  v_addc_co_u32_e32 v6, vcc, 0, v6, vcc
//  global_load_dwordx2 v[5:6], v[5:6], off
//  global_load_dwordx2 v[0:1], v[5:6], off offset:2048
//
// Future improvements:
//
// - This is currently missing stores of constants because loading
//   the constant into the data register is placed between the stores, although
//   this is arguably a scheduling problem.
//
// - Live interval recomputing seems inefficient. This currently only matches
//   one pair, and recomputes live intervals and moves on to the next pair. It
//   would be better to compute a list of all merges that need to occur.
//
// - With a list of instructions to process, we can also merge more. If a
//   cluster of loads have offsets that are too large to fit in the 8-bit
//   offsets, but are close enough to fit in the 8 bits, we can add to the base
//   pointer and use the new reduced offsets.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iterator>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "si-load-store-opt"

namespace {
enum InstClassEnum {
  UNKNOWN,
  DS_READ,
  DS_WRITE,
  S_BUFFER_LOAD_IMM,
  BUFFER_LOAD,
  BUFFER_STORE,
  MIMG,
  TBUFFER_LOAD,
  TBUFFER_STORE,
};

enum RegisterEnum {
  SBASE = 0x1,
  SRSRC = 0x2,
  SOFFSET = 0x4,
  VADDR = 0x8,
  ADDR = 0x10,
  SSAMP = 0x20,
};

class SILoadStoreOptimizer : public MachineFunctionPass {
  struct CombineInfo {
    MachineBasicBlock::iterator I;
    MachineBasicBlock::iterator Paired;
    unsigned EltSize;
    unsigned Offset0;
    unsigned Offset1;
    unsigned Width0;
    unsigned Width1;
    unsigned Format0;
    unsigned Format1;
    unsigned BaseOff;
    unsigned DMask0;
    unsigned DMask1;
    InstClassEnum InstClass;
    bool GLC0;
    bool GLC1;
    bool SLC0;
    bool SLC1;
    bool DLC0;
    bool DLC1;
    bool UseST64;
    SmallVector<MachineInstr *, 8> InstsToMove;
    int AddrIdx[5];
    const MachineOperand *AddrReg[5];
    unsigned NumAddresses;

    bool hasSameBaseAddress(const MachineInstr &MI) {
      for (unsigned i = 0; i < NumAddresses; i++) {
        const MachineOperand &AddrRegNext = MI.getOperand(AddrIdx[i]);

        if (AddrReg[i]->isImm() || AddrRegNext.isImm()) {
          if (AddrReg[i]->isImm() != AddrRegNext.isImm() ||
              AddrReg[i]->getImm() != AddrRegNext.getImm()) {
            return false;
          }
          continue;
        }

        // Check same base pointer. Be careful of subregisters, which can occur
        // with vectors of pointers.
        if (AddrReg[i]->getReg() != AddrRegNext.getReg() ||
            AddrReg[i]->getSubReg() != AddrRegNext.getSubReg()) {
         return false;
        }
      }
      return true;
    }

    bool hasMergeableAddress(const MachineRegisterInfo &MRI) {
      for (unsigned i = 0; i < NumAddresses; ++i) {
        const MachineOperand *AddrOp = AddrReg[i];
        // Immediates are always OK.
        if (AddrOp->isImm())
          continue;

        // Don't try to merge addresses that aren't either immediates or registers.
        // TODO: Should be possible to merge FrameIndexes and maybe some other
        // non-register
        if (!AddrOp->isReg())
          return false;

        // TODO: We should be able to merge physical reg addreses.
        if (Register::isPhysicalRegister(AddrOp->getReg()))
          return false;

        // If an address has only one use then there will be on other
        // instructions with the same address, so we can't merge this one.
        if (MRI.hasOneNonDBGUse(AddrOp->getReg()))
          return false;
      }
      return true;
    }

    void setMI(MachineBasicBlock::iterator MI, const SIInstrInfo &TII,
               const GCNSubtarget &STM);
    void setPaired(MachineBasicBlock::iterator MI, const SIInstrInfo &TII);
  };

  struct BaseRegisters {
    unsigned LoReg = 0;
    unsigned HiReg = 0;

    unsigned LoSubReg = 0;
    unsigned HiSubReg = 0;
  };

  struct MemAddress {
    BaseRegisters Base;
    int64_t Offset = 0;
  };

  using MemInfoMap = DenseMap<MachineInstr *, MemAddress>;

private:
  const GCNSubtarget *STM = nullptr;
  const SIInstrInfo *TII = nullptr;
  const SIRegisterInfo *TRI = nullptr;
  const MCSubtargetInfo *STI = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  AliasAnalysis *AA = nullptr;
  bool OptimizeAgain;

  static bool dmasksCanBeCombined(const CombineInfo &CI,
                                  const SIInstrInfo &TII);
  static bool offsetsCanBeCombined(CombineInfo &CI, const MCSubtargetInfo &STI);
  static bool widthsFit(const GCNSubtarget &STM, const CombineInfo &CI);
  static unsigned getNewOpcode(const CombineInfo &CI);
  static std::pair<unsigned, unsigned> getSubRegIdxs(const CombineInfo &CI);
  const TargetRegisterClass *getTargetRegisterClass(const CombineInfo &CI);

  bool findMatchingInst(CombineInfo &CI);

  unsigned read2Opcode(unsigned EltSize) const;
  unsigned read2ST64Opcode(unsigned EltSize) const;
  MachineBasicBlock::iterator mergeRead2Pair(CombineInfo &CI);

  unsigned write2Opcode(unsigned EltSize) const;
  unsigned write2ST64Opcode(unsigned EltSize) const;
  MachineBasicBlock::iterator mergeWrite2Pair(CombineInfo &CI);
  MachineBasicBlock::iterator mergeImagePair(CombineInfo &CI);
  MachineBasicBlock::iterator mergeSBufferLoadImmPair(CombineInfo &CI);
  MachineBasicBlock::iterator mergeBufferLoadPair(CombineInfo &CI);
  MachineBasicBlock::iterator mergeBufferStorePair(CombineInfo &CI);
  MachineBasicBlock::iterator mergeTBufferLoadPair(CombineInfo &CI);
  MachineBasicBlock::iterator mergeTBufferStorePair(CombineInfo &CI);

  void updateBaseAndOffset(MachineInstr &I, unsigned NewBase,
                           int32_t NewOffset) const;
  unsigned computeBase(MachineInstr &MI, const MemAddress &Addr) const;
  MachineOperand createRegOrImm(int32_t Val, MachineInstr &MI) const;
  Optional<int32_t> extractConstOffset(const MachineOperand &Op) const;
  void processBaseWithConstOffset(const MachineOperand &Base, MemAddress &Addr) const;
  /// Promotes constant offset to the immediate by adjusting the base. It
  /// tries to use a base from the nearby instructions that allows it to have
  /// a 13bit constant offset which gets promoted to the immediate.
  bool promoteConstantOffsetToImm(MachineInstr &CI,
                                  MemInfoMap &Visited,
                                  SmallPtrSet<MachineInstr *, 4> &Promoted) const;
  void addInstToMergeableList(const CombineInfo &CI,
                  std::list<std::list<CombineInfo> > &MergeableInsts) const;
  bool collectMergeableInsts(MachineBasicBlock &MBB,
                  std::list<std::list<CombineInfo> > &MergeableInsts) const;

public:
  static char ID;

  SILoadStoreOptimizer() : MachineFunctionPass(ID) {
    initializeSILoadStoreOptimizerPass(*PassRegistry::getPassRegistry());
  }

  void removeCombinedInst(std::list<CombineInfo> &MergeList,
                                         const MachineInstr &MI);
  bool optimizeInstsWithSameBaseAddr(std::list<CombineInfo> &MergeList,
                                     bool &OptimizeListAgain);
  bool optimizeBlock(std::list<std::list<CombineInfo> > &MergeableInsts);

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "SI Load Store Optimizer"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<AAResultsWrapperPass>();

    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

static unsigned getOpcodeWidth(const MachineInstr &MI, const SIInstrInfo &TII) {
  const unsigned Opc = MI.getOpcode();

  if (TII.isMUBUF(Opc)) {
    // FIXME: Handle d16 correctly
    return AMDGPU::getMUBUFElements(Opc);
  }
  if (TII.isMIMG(MI)) {
    uint64_t DMaskImm =
        TII.getNamedOperand(MI, AMDGPU::OpName::dmask)->getImm();
    return countPopulation(DMaskImm);
  }
  if (TII.isMTBUF(Opc)) {
    return AMDGPU::getMTBUFElements(Opc);
  }

  switch (Opc) {
  case AMDGPU::S_BUFFER_LOAD_DWORD_IMM:
    return 1;
  case AMDGPU::S_BUFFER_LOAD_DWORDX2_IMM:
    return 2;
  case AMDGPU::S_BUFFER_LOAD_DWORDX4_IMM:
    return 4;
  default:
    return 0;
  }
}

/// Maps instruction opcode to enum InstClassEnum.
static InstClassEnum getInstClass(unsigned Opc, const SIInstrInfo &TII) {
  switch (Opc) {
  default:
    if (TII.isMUBUF(Opc)) {
      switch (AMDGPU::getMUBUFBaseOpcode(Opc)) {
      default:
        return UNKNOWN;
      case AMDGPU::BUFFER_LOAD_DWORD_OFFEN:
      case AMDGPU::BUFFER_LOAD_DWORD_OFFEN_exact:
      case AMDGPU::BUFFER_LOAD_DWORD_OFFSET:
      case AMDGPU::BUFFER_LOAD_DWORD_OFFSET_exact:
        return BUFFER_LOAD;
      case AMDGPU::BUFFER_STORE_DWORD_OFFEN:
      case AMDGPU::BUFFER_STORE_DWORD_OFFEN_exact:
      case AMDGPU::BUFFER_STORE_DWORD_OFFSET:
      case AMDGPU::BUFFER_STORE_DWORD_OFFSET_exact:
        return BUFFER_STORE;
      }
    }
    if (TII.isMIMG(Opc)) {
      // Ignore instructions encoded without vaddr.
      if (AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::vaddr) == -1)
        return UNKNOWN;
      // TODO: Support IMAGE_GET_RESINFO and IMAGE_GET_LOD.
      if (TII.get(Opc).mayStore() || !TII.get(Opc).mayLoad() ||
          TII.isGather4(Opc))
        return UNKNOWN;
      return MIMG;
    }
    if (TII.isMTBUF(Opc)) {
      switch (AMDGPU::getMTBUFBaseOpcode(Opc)) {
      default:
        return UNKNOWN;
      case AMDGPU::TBUFFER_LOAD_FORMAT_X_OFFEN:
      case AMDGPU::TBUFFER_LOAD_FORMAT_X_OFFEN_exact:
      case AMDGPU::TBUFFER_LOAD_FORMAT_X_OFFSET:
      case AMDGPU::TBUFFER_LOAD_FORMAT_X_OFFSET_exact:
        return TBUFFER_LOAD;
      case AMDGPU::TBUFFER_STORE_FORMAT_X_OFFEN:
      case AMDGPU::TBUFFER_STORE_FORMAT_X_OFFEN_exact:
      case AMDGPU::TBUFFER_STORE_FORMAT_X_OFFSET:
      case AMDGPU::TBUFFER_STORE_FORMAT_X_OFFSET_exact:
        return TBUFFER_STORE;
      }
    }
    return UNKNOWN;
  case AMDGPU::S_BUFFER_LOAD_DWORD_IMM:
  case AMDGPU::S_BUFFER_LOAD_DWORDX2_IMM:
  case AMDGPU::S_BUFFER_LOAD_DWORDX4_IMM:
    return S_BUFFER_LOAD_IMM;
  case AMDGPU::DS_READ_B32:
  case AMDGPU::DS_READ_B32_gfx9:
  case AMDGPU::DS_READ_B64:
  case AMDGPU::DS_READ_B64_gfx9:
    return DS_READ;
  case AMDGPU::DS_WRITE_B32:
  case AMDGPU::DS_WRITE_B32_gfx9:
  case AMDGPU::DS_WRITE_B64:
  case AMDGPU::DS_WRITE_B64_gfx9:
    return DS_WRITE;
  }
}

/// Determines instruction subclass from opcode. Only instructions
/// of the same subclass can be merged together.
static unsigned getInstSubclass(unsigned Opc, const SIInstrInfo &TII) {
  switch (Opc) {
  default:
    if (TII.isMUBUF(Opc))
      return AMDGPU::getMUBUFBaseOpcode(Opc);
    if (TII.isMIMG(Opc)) {
      const AMDGPU::MIMGInfo *Info = AMDGPU::getMIMGInfo(Opc);
      assert(Info);
      return Info->BaseOpcode;
    }
    if (TII.isMTBUF(Opc))
      return AMDGPU::getMTBUFBaseOpcode(Opc);
    return -1;
  case AMDGPU::DS_READ_B32:
  case AMDGPU::DS_READ_B32_gfx9:
  case AMDGPU::DS_READ_B64:
  case AMDGPU::DS_READ_B64_gfx9:
  case AMDGPU::DS_WRITE_B32:
  case AMDGPU::DS_WRITE_B32_gfx9:
  case AMDGPU::DS_WRITE_B64:
  case AMDGPU::DS_WRITE_B64_gfx9:
    return Opc;
  case AMDGPU::S_BUFFER_LOAD_DWORD_IMM:
  case AMDGPU::S_BUFFER_LOAD_DWORDX2_IMM:
  case AMDGPU::S_BUFFER_LOAD_DWORDX4_IMM:
    return AMDGPU::S_BUFFER_LOAD_DWORD_IMM;
  }
}

static unsigned getRegs(unsigned Opc, const SIInstrInfo &TII) {
  if (TII.isMUBUF(Opc)) {
    unsigned result = 0;

    if (AMDGPU::getMUBUFHasVAddr(Opc)) {
      result |= VADDR;
    }

    if (AMDGPU::getMUBUFHasSrsrc(Opc)) {
      result |= SRSRC;
    }

    if (AMDGPU::getMUBUFHasSoffset(Opc)) {
      result |= SOFFSET;
    }

    return result;
  }

  if (TII.isMIMG(Opc)) {
    unsigned result = VADDR | SRSRC;
    const AMDGPU::MIMGInfo *Info = AMDGPU::getMIMGInfo(Opc);
    if (Info && AMDGPU::getMIMGBaseOpcodeInfo(Info->BaseOpcode)->Sampler)
      result |= SSAMP;

    return result;
  }
  if (TII.isMTBUF(Opc)) {
    unsigned result = 0;

    if (AMDGPU::getMTBUFHasVAddr(Opc)) {
      result |= VADDR;
    }

    if (AMDGPU::getMTBUFHasSrsrc(Opc)) {
      result |= SRSRC;
    }

    if (AMDGPU::getMTBUFHasSoffset(Opc)) {
      result |= SOFFSET;
    }

    return result;
  }

  switch (Opc) {
  default:
    return 0;
  case AMDGPU::S_BUFFER_LOAD_DWORD_IMM:
  case AMDGPU::S_BUFFER_LOAD_DWORDX2_IMM:
  case AMDGPU::S_BUFFER_LOAD_DWORDX4_IMM:
    return SBASE;
  case AMDGPU::DS_READ_B32:
  case AMDGPU::DS_READ_B64:
  case AMDGPU::DS_READ_B32_gfx9:
  case AMDGPU::DS_READ_B64_gfx9:
  case AMDGPU::DS_WRITE_B32:
  case AMDGPU::DS_WRITE_B64:
  case AMDGPU::DS_WRITE_B32_gfx9:
  case AMDGPU::DS_WRITE_B64_gfx9:
    return ADDR;
  }
}

void SILoadStoreOptimizer::CombineInfo::setMI(MachineBasicBlock::iterator MI,
                                              const SIInstrInfo &TII,
                                              const GCNSubtarget &STM) {
  I = MI;
  unsigned Opc = MI->getOpcode();
  InstClass = getInstClass(Opc, TII);

  if (InstClass == UNKNOWN)
    return;

  switch (InstClass) {
  case DS_READ:
   EltSize =
          (Opc == AMDGPU::DS_READ_B64 || Opc == AMDGPU::DS_READ_B64_gfx9) ? 8
                                                                          : 4;
   break;
  case DS_WRITE:
    EltSize =
          (Opc == AMDGPU::DS_WRITE_B64 || Opc == AMDGPU::DS_WRITE_B64_gfx9) ? 8
                                                                            : 4;
    break;
  case S_BUFFER_LOAD_IMM:
    EltSize = AMDGPU::getSMRDEncodedOffset(STM, 4);
    break;
  default:
    EltSize = 4;
    break;
  }

  if (InstClass == MIMG) {
    DMask0 = TII.getNamedOperand(*I, AMDGPU::OpName::dmask)->getImm();
  } else {
    int OffsetIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::offset);
    Offset0 = I->getOperand(OffsetIdx).getImm();
  }

  if (InstClass == TBUFFER_LOAD || InstClass == TBUFFER_STORE)
    Format0 = TII.getNamedOperand(*I, AMDGPU::OpName::format)->getImm();

  Width0 = getOpcodeWidth(*I, TII);

  if ((InstClass == DS_READ) || (InstClass == DS_WRITE)) {
    Offset0 &= 0xffff;
  } else if (InstClass != MIMG) {
    GLC0 = TII.getNamedOperand(*I, AMDGPU::OpName::glc)->getImm();
    if (InstClass != S_BUFFER_LOAD_IMM) {
      SLC0 = TII.getNamedOperand(*I, AMDGPU::OpName::slc)->getImm();
    }
    DLC0 = TII.getNamedOperand(*I, AMDGPU::OpName::dlc)->getImm();
  }

  unsigned AddrOpName[5] = {0};
  NumAddresses = 0;
  const unsigned Regs = getRegs(I->getOpcode(), TII);

  if (Regs & ADDR) {
    AddrOpName[NumAddresses++] = AMDGPU::OpName::addr;
  }

  if (Regs & SBASE) {
    AddrOpName[NumAddresses++] = AMDGPU::OpName::sbase;
  }

  if (Regs & SRSRC) {
    AddrOpName[NumAddresses++] = AMDGPU::OpName::srsrc;
  }

  if (Regs & SOFFSET) {
    AddrOpName[NumAddresses++] = AMDGPU::OpName::soffset;
  }

  if (Regs & VADDR) {
    AddrOpName[NumAddresses++] = AMDGPU::OpName::vaddr;
  }

  if (Regs & SSAMP) {
    AddrOpName[NumAddresses++] = AMDGPU::OpName::ssamp;
  }

  for (unsigned i = 0; i < NumAddresses; i++) {
    AddrIdx[i] = AMDGPU::getNamedOperandIdx(I->getOpcode(), AddrOpName[i]);
    AddrReg[i] = &I->getOperand(AddrIdx[i]);
  }

  InstsToMove.clear();
}

void SILoadStoreOptimizer::CombineInfo::setPaired(MachineBasicBlock::iterator MI,
                                                  const SIInstrInfo &TII) {
  Paired = MI;
  assert(InstClass == getInstClass(Paired->getOpcode(), TII));

  if (InstClass == MIMG) {
    DMask1 = TII.getNamedOperand(*Paired, AMDGPU::OpName::dmask)->getImm();
  } else {
    int OffsetIdx =
        AMDGPU::getNamedOperandIdx(I->getOpcode(), AMDGPU::OpName::offset);
    Offset1 = Paired->getOperand(OffsetIdx).getImm();
  }

  if (InstClass == TBUFFER_LOAD || InstClass == TBUFFER_STORE)
    Format1 = TII.getNamedOperand(*Paired, AMDGPU::OpName::format)->getImm();

  Width1 = getOpcodeWidth(*Paired, TII);
  if ((InstClass == DS_READ) || (InstClass == DS_WRITE)) {
    Offset1 &= 0xffff;
  } else if (InstClass != MIMG) {
    GLC1 = TII.getNamedOperand(*Paired, AMDGPU::OpName::glc)->getImm();
    if (InstClass != S_BUFFER_LOAD_IMM) {
      SLC1 = TII.getNamedOperand(*Paired, AMDGPU::OpName::slc)->getImm();
    }
    DLC1 = TII.getNamedOperand(*Paired, AMDGPU::OpName::dlc)->getImm();
  }
}

} // end anonymous namespace.

INITIALIZE_PASS_BEGIN(SILoadStoreOptimizer, DEBUG_TYPE,
                      "SI Load Store Optimizer", false, false)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_END(SILoadStoreOptimizer, DEBUG_TYPE, "SI Load Store Optimizer",
                    false, false)

char SILoadStoreOptimizer::ID = 0;

char &llvm::SILoadStoreOptimizerID = SILoadStoreOptimizer::ID;

FunctionPass *llvm::createSILoadStoreOptimizerPass() {
  return new SILoadStoreOptimizer();
}

static void moveInstsAfter(MachineBasicBlock::iterator I,
                           ArrayRef<MachineInstr *> InstsToMove) {
  MachineBasicBlock *MBB = I->getParent();
  ++I;
  for (MachineInstr *MI : InstsToMove) {
    MI->removeFromParent();
    MBB->insert(I, MI);
  }
}

static void addDefsUsesToList(const MachineInstr &MI,
                              DenseSet<unsigned> &RegDefs,
                              DenseSet<unsigned> &PhysRegUses) {
  for (const MachineOperand &Op : MI.operands()) {
    if (Op.isReg()) {
      if (Op.isDef())
        RegDefs.insert(Op.getReg());
      else if (Op.readsReg() && Register::isPhysicalRegister(Op.getReg()))
        PhysRegUses.insert(Op.getReg());
    }
  }
}

static bool memAccessesCanBeReordered(MachineBasicBlock::iterator A,
                                      MachineBasicBlock::iterator B,
                                      AliasAnalysis *AA) {
  // RAW or WAR - cannot reorder
  // WAW - cannot reorder
  // RAR - safe to reorder
  return !(A->mayStore() || B->mayStore()) || !A->mayAlias(AA, *B, true);
}

// Add MI and its defs to the lists if MI reads one of the defs that are
// already in the list. Returns true in that case.
static bool addToListsIfDependent(MachineInstr &MI, DenseSet<unsigned> &RegDefs,
                                  DenseSet<unsigned> &PhysRegUses,
                                  SmallVectorImpl<MachineInstr *> &Insts) {
  for (MachineOperand &Use : MI.operands()) {
    // If one of the defs is read, then there is a use of Def between I and the
    // instruction that I will potentially be merged with. We will need to move
    // this instruction after the merged instructions.
    //
    // Similarly, if there is a def which is read by an instruction that is to
    // be moved for merging, then we need to move the def-instruction as well.
    // This can only happen for physical registers such as M0; virtual
    // registers are in SSA form.
    if (Use.isReg() &&
        ((Use.readsReg() && RegDefs.count(Use.getReg())) ||
         (Use.isDef() && RegDefs.count(Use.getReg())) ||
         (Use.isDef() && Register::isPhysicalRegister(Use.getReg()) &&
          PhysRegUses.count(Use.getReg())))) {
      Insts.push_back(&MI);
      addDefsUsesToList(MI, RegDefs, PhysRegUses);
      return true;
    }
  }

  return false;
}

static bool canMoveInstsAcrossMemOp(MachineInstr &MemOp,
                                    ArrayRef<MachineInstr *> InstsToMove,
                                    AliasAnalysis *AA) {
  assert(MemOp.mayLoadOrStore());

  for (MachineInstr *InstToMove : InstsToMove) {
    if (!InstToMove->mayLoadOrStore())
      continue;
    if (!memAccessesCanBeReordered(MemOp, *InstToMove, AA))
      return false;
  }
  return true;
}

// This function assumes that \p A and \p B have are identical except for
// size and offset, and they referecne adjacent memory.
static MachineMemOperand *combineKnownAdjacentMMOs(MachineFunction &MF,
                                                   const MachineMemOperand *A,
                                                   const MachineMemOperand *B) {
  unsigned MinOffset = std::min(A->getOffset(), B->getOffset());
  unsigned Size = A->getSize() + B->getSize();
  // This function adds the offset parameter to the existing offset for A,
  // so we pass 0 here as the offset and then manually set it to the correct
  // value after the call.
  MachineMemOperand *MMO = MF.getMachineMemOperand(A, 0, Size);
  MMO->setOffset(MinOffset);
  return MMO;
}

bool SILoadStoreOptimizer::dmasksCanBeCombined(const CombineInfo &CI, const SIInstrInfo &TII) {
  assert(CI.InstClass == MIMG);

  // Ignore instructions with tfe/lwe set.
  const auto *TFEOp = TII.getNamedOperand(*CI.I, AMDGPU::OpName::tfe);
  const auto *LWEOp = TII.getNamedOperand(*CI.I, AMDGPU::OpName::lwe);

  if ((TFEOp && TFEOp->getImm()) || (LWEOp && LWEOp->getImm()))
    return false;

  // Check other optional immediate operands for equality.
  unsigned OperandsToMatch[] = {AMDGPU::OpName::glc, AMDGPU::OpName::slc,
                                AMDGPU::OpName::d16, AMDGPU::OpName::unorm,
                                AMDGPU::OpName::da,  AMDGPU::OpName::r128};

  for (auto op : OperandsToMatch) {
    int Idx = AMDGPU::getNamedOperandIdx(CI.I->getOpcode(), op);
    if (AMDGPU::getNamedOperandIdx(CI.Paired->getOpcode(), op) != Idx)
      return false;
    if (Idx != -1 &&
        CI.I->getOperand(Idx).getImm() != CI.Paired->getOperand(Idx).getImm())
      return false;
  }

  // Check DMask for overlaps.
  unsigned MaxMask = std::max(CI.DMask0, CI.DMask1);
  unsigned MinMask = std::min(CI.DMask0, CI.DMask1);

  unsigned AllowedBitsForMin = llvm::countTrailingZeros(MaxMask);
  if ((1u << AllowedBitsForMin) <= MinMask)
    return false;

  return true;
}

static unsigned getBufferFormatWithCompCount(unsigned OldFormat,
                                       unsigned ComponentCount,
                                       const MCSubtargetInfo &STI) {
  if (ComponentCount > 4)
    return 0;

  const llvm::AMDGPU::GcnBufferFormatInfo *OldFormatInfo =
      llvm::AMDGPU::getGcnBufferFormatInfo(OldFormat, STI);
  if (!OldFormatInfo)
    return 0;

  const llvm::AMDGPU::GcnBufferFormatInfo *NewFormatInfo =
      llvm::AMDGPU::getGcnBufferFormatInfo(OldFormatInfo->BitsPerComp,
                                           ComponentCount,
                                           OldFormatInfo->NumFormat, STI);

  if (!NewFormatInfo)
    return 0;

  assert(NewFormatInfo->NumFormat == OldFormatInfo->NumFormat &&
         NewFormatInfo->BitsPerComp == OldFormatInfo->BitsPerComp);

  return NewFormatInfo->Format;
}

bool SILoadStoreOptimizer::offsetsCanBeCombined(CombineInfo &CI,
                                                const MCSubtargetInfo &STI) {
  assert(CI.InstClass != MIMG);

  // XXX - Would the same offset be OK? Is there any reason this would happen or
  // be useful?
  if (CI.Offset0 == CI.Offset1)
    return false;

  // This won't be valid if the offset isn't aligned.
  if ((CI.Offset0 % CI.EltSize != 0) || (CI.Offset1 % CI.EltSize != 0))
    return false;

  if (CI.InstClass == TBUFFER_LOAD || CI.InstClass == TBUFFER_STORE) {

    const llvm::AMDGPU::GcnBufferFormatInfo *Info0 =
        llvm::AMDGPU::getGcnBufferFormatInfo(CI.Format0, STI);
    if (!Info0)
      return false;
    const llvm::AMDGPU::GcnBufferFormatInfo *Info1 =
        llvm::AMDGPU::getGcnBufferFormatInfo(CI.Format1, STI);
    if (!Info1)
      return false;

    if (Info0->BitsPerComp != Info1->BitsPerComp ||
        Info0->NumFormat != Info1->NumFormat)
      return false;

    // TODO: Should be possible to support more formats, but if format loads
    // are not dword-aligned, the merged load might not be valid.
    if (Info0->BitsPerComp != 32)
      return false;

    if (getBufferFormatWithCompCount(CI.Format0, CI.Width0 + CI.Width1, STI) == 0)
      return false;
  }

  unsigned EltOffset0 = CI.Offset0 / CI.EltSize;
  unsigned EltOffset1 = CI.Offset1 / CI.EltSize;
  CI.UseST64 = false;
  CI.BaseOff = 0;

  // Handle SMEM and VMEM instructions.
  if ((CI.InstClass != DS_READ) && (CI.InstClass != DS_WRITE)) {
    return (EltOffset0 + CI.Width0 == EltOffset1 ||
            EltOffset1 + CI.Width1 == EltOffset0) &&
           CI.GLC0 == CI.GLC1 && CI.DLC0 == CI.DLC1 &&
           (CI.InstClass == S_BUFFER_LOAD_IMM || CI.SLC0 == CI.SLC1);
  }

  // If the offset in elements doesn't fit in 8-bits, we might be able to use
  // the stride 64 versions.
  if ((EltOffset0 % 64 == 0) && (EltOffset1 % 64) == 0 &&
      isUInt<8>(EltOffset0 / 64) && isUInt<8>(EltOffset1 / 64)) {
    CI.Offset0 = EltOffset0 / 64;
    CI.Offset1 = EltOffset1 / 64;
    CI.UseST64 = true;
    return true;
  }

  // Check if the new offsets fit in the reduced 8-bit range.
  if (isUInt<8>(EltOffset0) && isUInt<8>(EltOffset1)) {
    CI.Offset0 = EltOffset0;
    CI.Offset1 = EltOffset1;
    return true;
  }

  // Try to shift base address to decrease offsets.
  unsigned OffsetDiff = std::abs((int)EltOffset1 - (int)EltOffset0);
  CI.BaseOff = std::min(CI.Offset0, CI.Offset1);

  if ((OffsetDiff % 64 == 0) && isUInt<8>(OffsetDiff / 64)) {
    CI.Offset0 = (EltOffset0 - CI.BaseOff / CI.EltSize) / 64;
    CI.Offset1 = (EltOffset1 - CI.BaseOff / CI.EltSize) / 64;
    CI.UseST64 = true;
    return true;
  }

  if (isUInt<8>(OffsetDiff)) {
    CI.Offset0 = EltOffset0 - CI.BaseOff / CI.EltSize;
    CI.Offset1 = EltOffset1 - CI.BaseOff / CI.EltSize;
    return true;
  }

  return false;
}

bool SILoadStoreOptimizer::widthsFit(const GCNSubtarget &STM,
                                     const CombineInfo &CI) {
  const unsigned Width = (CI.Width0 + CI.Width1);
  switch (CI.InstClass) {
  default:
    return (Width <= 4) && (STM.hasDwordx3LoadStores() || (Width != 3));
  case S_BUFFER_LOAD_IMM:
    switch (Width) {
    default:
      return false;
    case 2:
    case 4:
      return true;
    }
  }
}

bool SILoadStoreOptimizer::findMatchingInst(CombineInfo &CI) {
  MachineBasicBlock *MBB = CI.I->getParent();
  MachineBasicBlock::iterator E = MBB->end();
  MachineBasicBlock::iterator MBBI = CI.I;

  const unsigned Opc = CI.I->getOpcode();
  const InstClassEnum InstClass = getInstClass(Opc, *TII);

  if (InstClass == UNKNOWN) {
    return false;
  }
  const unsigned InstSubclass = getInstSubclass(Opc, *TII);

  // Do not merge VMEM buffer instructions with "swizzled" bit set.
  int Swizzled =
      AMDGPU::getNamedOperandIdx(CI.I->getOpcode(), AMDGPU::OpName::swz);
  if (Swizzled != -1 && CI.I->getOperand(Swizzled).getImm())
    return false;

  ++MBBI;

  DenseSet<unsigned> RegDefsToMove;
  DenseSet<unsigned> PhysRegUsesToMove;
  addDefsUsesToList(*CI.I, RegDefsToMove, PhysRegUsesToMove);

  for (; MBBI != E; ++MBBI) {

    if ((getInstClass(MBBI->getOpcode(), *TII) != InstClass) ||
        (getInstSubclass(MBBI->getOpcode(), *TII) != InstSubclass)) {
      // This is not a matching instruction, but we can keep looking as
      // long as one of these conditions are met:
      // 1. It is safe to move I down past MBBI.
      // 2. It is safe to move MBBI down past the instruction that I will
      //    be merged into.

      if (MBBI->hasUnmodeledSideEffects()) {
        // We can't re-order this instruction with respect to other memory
        // operations, so we fail both conditions mentioned above.
        return false;
      }

      if (MBBI->mayLoadOrStore() &&
          (!memAccessesCanBeReordered(*CI.I, *MBBI, AA) ||
           !canMoveInstsAcrossMemOp(*MBBI, CI.InstsToMove, AA))) {
        // We fail condition #1, but we may still be able to satisfy condition
        // #2.  Add this instruction to the move list and then we will check
        // if condition #2 holds once we have selected the matching instruction.
        CI.InstsToMove.push_back(&*MBBI);
        addDefsUsesToList(*MBBI, RegDefsToMove, PhysRegUsesToMove);
        continue;
      }

      // When we match I with another DS instruction we will be moving I down
      // to the location of the matched instruction any uses of I will need to
      // be moved down as well.
      addToListsIfDependent(*MBBI, RegDefsToMove, PhysRegUsesToMove,
                            CI.InstsToMove);
      continue;
    }

    // Don't merge volatiles.
    if (MBBI->hasOrderedMemoryRef())
      return false;

    int Swizzled =
        AMDGPU::getNamedOperandIdx(MBBI->getOpcode(), AMDGPU::OpName::swz);
    if (Swizzled != -1 && MBBI->getOperand(Swizzled).getImm())
      return false;

    // Handle a case like
    //   DS_WRITE_B32 addr, v, idx0
    //   w = DS_READ_B32 addr, idx0
    //   DS_WRITE_B32 addr, f(w), idx1
    // where the DS_READ_B32 ends up in InstsToMove and therefore prevents
    // merging of the two writes.
    if (addToListsIfDependent(*MBBI, RegDefsToMove, PhysRegUsesToMove,
                              CI.InstsToMove))
      continue;

    bool Match = CI.hasSameBaseAddress(*MBBI);

    if (Match) {
      CI.setPaired(MBBI, *TII);

      // Check both offsets (or masks for MIMG) can be combined and fit in the
      // reduced range.
      bool canBeCombined =
          CI.InstClass == MIMG
              ? dmasksCanBeCombined(CI, *TII)
              : widthsFit(*STM, CI) && offsetsCanBeCombined(CI, *STI);

      // We also need to go through the list of instructions that we plan to
      // move and make sure they are all safe to move down past the merged
      // instruction.
      if (canBeCombined && canMoveInstsAcrossMemOp(*MBBI, CI.InstsToMove, AA))
        return true;
    }

    // We've found a load/store that we couldn't merge for some reason.
    // We could potentially keep looking, but we'd need to make sure that
    // it was safe to move I and also all the instruction in InstsToMove
    // down past this instruction.
    // check if we can move I across MBBI and if we can move all I's users
    if (!memAccessesCanBeReordered(*CI.I, *MBBI, AA) ||
        !canMoveInstsAcrossMemOp(*MBBI, CI.InstsToMove, AA))
      break;
  }
  return false;
}

unsigned SILoadStoreOptimizer::read2Opcode(unsigned EltSize) const {
  if (STM->ldsRequiresM0Init())
    return (EltSize == 4) ? AMDGPU::DS_READ2_B32 : AMDGPU::DS_READ2_B64;
  return (EltSize == 4) ? AMDGPU::DS_READ2_B32_gfx9 : AMDGPU::DS_READ2_B64_gfx9;
}

unsigned SILoadStoreOptimizer::read2ST64Opcode(unsigned EltSize) const {
  if (STM->ldsRequiresM0Init())
    return (EltSize == 4) ? AMDGPU::DS_READ2ST64_B32 : AMDGPU::DS_READ2ST64_B64;

  return (EltSize == 4) ? AMDGPU::DS_READ2ST64_B32_gfx9
                        : AMDGPU::DS_READ2ST64_B64_gfx9;
}

MachineBasicBlock::iterator
SILoadStoreOptimizer::mergeRead2Pair(CombineInfo &CI) {
  MachineBasicBlock *MBB = CI.I->getParent();

  // Be careful, since the addresses could be subregisters themselves in weird
  // cases, like vectors of pointers.
  const auto *AddrReg = TII->getNamedOperand(*CI.I, AMDGPU::OpName::addr);

  const auto *Dest0 = TII->getNamedOperand(*CI.I, AMDGPU::OpName::vdst);
  const auto *Dest1 = TII->getNamedOperand(*CI.Paired, AMDGPU::OpName::vdst);

  unsigned NewOffset0 = CI.Offset0;
  unsigned NewOffset1 = CI.Offset1;
  unsigned Opc =
      CI.UseST64 ? read2ST64Opcode(CI.EltSize) : read2Opcode(CI.EltSize);

  unsigned SubRegIdx0 = (CI.EltSize == 4) ? AMDGPU::sub0 : AMDGPU::sub0_sub1;
  unsigned SubRegIdx1 = (CI.EltSize == 4) ? AMDGPU::sub1 : AMDGPU::sub2_sub3;

  if (NewOffset0 > NewOffset1) {
    // Canonicalize the merged instruction so the smaller offset comes first.
    std::swap(NewOffset0, NewOffset1);
    std::swap(SubRegIdx0, SubRegIdx1);
  }

  assert((isUInt<8>(NewOffset0) && isUInt<8>(NewOffset1)) &&
         (NewOffset0 != NewOffset1) && "Computed offset doesn't fit");

  const MCInstrDesc &Read2Desc = TII->get(Opc);

  const TargetRegisterClass *SuperRC =
      (CI.EltSize == 4) ? &AMDGPU::VReg_64RegClass : &AMDGPU::VReg_128RegClass;
  Register DestReg = MRI->createVirtualRegister(SuperRC);

  DebugLoc DL = CI.I->getDebugLoc();

  Register BaseReg = AddrReg->getReg();
  unsigned BaseSubReg = AddrReg->getSubReg();
  unsigned BaseRegFlags = 0;
  if (CI.BaseOff) {
    Register ImmReg = MRI->createVirtualRegister(&AMDGPU::SReg_32RegClass);
    BuildMI(*MBB, CI.Paired, DL, TII->get(AMDGPU::S_MOV_B32), ImmReg)
        .addImm(CI.BaseOff);

    BaseReg = MRI->createVirtualRegister(&AMDGPU::VGPR_32RegClass);
    BaseRegFlags = RegState::Kill;

    TII->getAddNoCarry(*MBB, CI.Paired, DL, BaseReg)
        .addReg(ImmReg)
        .addReg(AddrReg->getReg(), 0, BaseSubReg)
        .addImm(0); // clamp bit
    BaseSubReg = 0;
  }

  MachineInstrBuilder Read2 =
      BuildMI(*MBB, CI.Paired, DL, Read2Desc, DestReg)
          .addReg(BaseReg, BaseRegFlags, BaseSubReg) // addr
          .addImm(NewOffset0)                        // offset0
          .addImm(NewOffset1)                        // offset1
          .addImm(0)                                 // gds
          .cloneMergedMemRefs({&*CI.I, &*CI.Paired});

  (void)Read2;

  const MCInstrDesc &CopyDesc = TII->get(TargetOpcode::COPY);

  // Copy to the old destination registers.
  BuildMI(*MBB, CI.Paired, DL, CopyDesc)
      .add(*Dest0) // Copy to same destination including flags and sub reg.
      .addReg(DestReg, 0, SubRegIdx0);
  MachineInstr *Copy1 = BuildMI(*MBB, CI.Paired, DL, CopyDesc)
                            .add(*Dest1)
                            .addReg(DestReg, RegState::Kill, SubRegIdx1);

  moveInstsAfter(Copy1, CI.InstsToMove);

  CI.I->eraseFromParent();
  CI.Paired->eraseFromParent();

  LLVM_DEBUG(dbgs() << "Inserted read2: " << *Read2 << '\n');
  return Read2;
}

unsigned SILoadStoreOptimizer::write2Opcode(unsigned EltSize) const {
  if (STM->ldsRequiresM0Init())
    return (EltSize == 4) ? AMDGPU::DS_WRITE2_B32 : AMDGPU::DS_WRITE2_B64;
  return (EltSize == 4) ? AMDGPU::DS_WRITE2_B32_gfx9
                        : AMDGPU::DS_WRITE2_B64_gfx9;
}

unsigned SILoadStoreOptimizer::write2ST64Opcode(unsigned EltSize) const {
  if (STM->ldsRequiresM0Init())
    return (EltSize == 4) ? AMDGPU::DS_WRITE2ST64_B32
                          : AMDGPU::DS_WRITE2ST64_B64;

  return (EltSize == 4) ? AMDGPU::DS_WRITE2ST64_B32_gfx9
                        : AMDGPU::DS_WRITE2ST64_B64_gfx9;
}

MachineBasicBlock::iterator
SILoadStoreOptimizer::mergeWrite2Pair(CombineInfo &CI) {
  MachineBasicBlock *MBB = CI.I->getParent();

  // Be sure to use .addOperand(), and not .addReg() with these. We want to be
  // sure we preserve the subregister index and any register flags set on them.
  const MachineOperand *AddrReg =
      TII->getNamedOperand(*CI.I, AMDGPU::OpName::addr);
  const MachineOperand *Data0 =
      TII->getNamedOperand(*CI.I, AMDGPU::OpName::data0);
  const MachineOperand *Data1 =
      TII->getNamedOperand(*CI.Paired, AMDGPU::OpName::data0);

  unsigned NewOffset0 = CI.Offset0;
  unsigned NewOffset1 = CI.Offset1;
  unsigned Opc =
      CI.UseST64 ? write2ST64Opcode(CI.EltSize) : write2Opcode(CI.EltSize);

  if (NewOffset0 > NewOffset1) {
    // Canonicalize the merged instruction so the smaller offset comes first.
    std::swap(NewOffset0, NewOffset1);
    std::swap(Data0, Data1);
  }

  assert((isUInt<8>(NewOffset0) && isUInt<8>(NewOffset1)) &&
         (NewOffset0 != NewOffset1) && "Computed offset doesn't fit");

  const MCInstrDesc &Write2Desc = TII->get(Opc);
  DebugLoc DL = CI.I->getDebugLoc();

  Register BaseReg = AddrReg->getReg();
  unsigned BaseSubReg = AddrReg->getSubReg();
  unsigned BaseRegFlags = 0;
  if (CI.BaseOff) {
    Register ImmReg = MRI->createVirtualRegister(&AMDGPU::SReg_32RegClass);
    BuildMI(*MBB, CI.Paired, DL, TII->get(AMDGPU::S_MOV_B32), ImmReg)
        .addImm(CI.BaseOff);

    BaseReg = MRI->createVirtualRegister(&AMDGPU::VGPR_32RegClass);
    BaseRegFlags = RegState::Kill;

    TII->getAddNoCarry(*MBB, CI.Paired, DL, BaseReg)
        .addReg(ImmReg)
        .addReg(AddrReg->getReg(), 0, BaseSubReg)
        .addImm(0); // clamp bit
    BaseSubReg = 0;
  }

  MachineInstrBuilder Write2 =
      BuildMI(*MBB, CI.Paired, DL, Write2Desc)
          .addReg(BaseReg, BaseRegFlags, BaseSubReg) // addr
          .add(*Data0)                               // data0
          .add(*Data1)                               // data1
          .addImm(NewOffset0)                        // offset0
          .addImm(NewOffset1)                        // offset1
          .addImm(0)                                 // gds
          .cloneMergedMemRefs({&*CI.I, &*CI.Paired});

  moveInstsAfter(Write2, CI.InstsToMove);

  CI.I->eraseFromParent();
  CI.Paired->eraseFromParent();

  LLVM_DEBUG(dbgs() << "Inserted write2 inst: " << *Write2 << '\n');
  return Write2;
}

MachineBasicBlock::iterator
SILoadStoreOptimizer::mergeImagePair(CombineInfo &CI) {
  MachineBasicBlock *MBB = CI.I->getParent();
  DebugLoc DL = CI.I->getDebugLoc();
  const unsigned Opcode = getNewOpcode(CI);

  const TargetRegisterClass *SuperRC = getTargetRegisterClass(CI);

  Register DestReg = MRI->createVirtualRegister(SuperRC);
  unsigned MergedDMask = CI.DMask0 | CI.DMask1;
  unsigned DMaskIdx =
      AMDGPU::getNamedOperandIdx(CI.I->getOpcode(), AMDGPU::OpName::dmask);

  auto MIB = BuildMI(*MBB, CI.Paired, DL, TII->get(Opcode), DestReg);
  for (unsigned I = 1, E = (*CI.I).getNumOperands(); I != E; ++I) {
    if (I == DMaskIdx)
      MIB.addImm(MergedDMask);
    else
      MIB.add((*CI.I).getOperand(I));
  }

  // It shouldn't be possible to get this far if the two instructions
  // don't have a single memoperand, because MachineInstr::mayAlias()
  // will return true if this is the case.
  assert(CI.I->hasOneMemOperand() && CI.Paired->hasOneMemOperand());

  const MachineMemOperand *MMOa = *CI.I->memoperands_begin();
  const MachineMemOperand *MMOb = *CI.Paired->memoperands_begin();

  MachineInstr *New = MIB.addMemOperand(combineKnownAdjacentMMOs(*MBB->getParent(), MMOa, MMOb));

  std::pair<unsigned, unsigned> SubRegIdx = getSubRegIdxs(CI);
  const unsigned SubRegIdx0 = std::get<0>(SubRegIdx);
  const unsigned SubRegIdx1 = std::get<1>(SubRegIdx);

  // Copy to the old destination registers.
  const MCInstrDesc &CopyDesc = TII->get(TargetOpcode::COPY);
  const auto *Dest0 = TII->getNamedOperand(*CI.I, AMDGPU::OpName::vdata);
  const auto *Dest1 = TII->getNamedOperand(*CI.Paired, AMDGPU::OpName::vdata);

  BuildMI(*MBB, CI.Paired, DL, CopyDesc)
      .add(*Dest0) // Copy to same destination including flags and sub reg.
      .addReg(DestReg, 0, SubRegIdx0);
  MachineInstr *Copy1 = BuildMI(*MBB, CI.Paired, DL, CopyDesc)
                            .add(*Dest1)
                            .addReg(DestReg, RegState::Kill, SubRegIdx1);

  moveInstsAfter(Copy1, CI.InstsToMove);

  CI.I->eraseFromParent();
  CI.Paired->eraseFromParent();
  return New;
}

MachineBasicBlock::iterator
SILoadStoreOptimizer::mergeSBufferLoadImmPair(CombineInfo &CI) {
  MachineBasicBlock *MBB = CI.I->getParent();
  DebugLoc DL = CI.I->getDebugLoc();
  const unsigned Opcode = getNewOpcode(CI);

  const TargetRegisterClass *SuperRC = getTargetRegisterClass(CI);

  Register DestReg = MRI->createVirtualRegister(SuperRC);
  unsigned MergedOffset = std::min(CI.Offset0, CI.Offset1);

  // It shouldn't be possible to get this far if the two instructions
  // don't have a single memoperand, because MachineInstr::mayAlias()
  // will return true if this is the case.
  assert(CI.I->hasOneMemOperand() && CI.Paired->hasOneMemOperand());

  const MachineMemOperand *MMOa = *CI.I->memoperands_begin();
  const MachineMemOperand *MMOb = *CI.Paired->memoperands_begin();

  MachineInstr *New =
    BuildMI(*MBB, CI.Paired, DL, TII->get(Opcode), DestReg)
        .add(*TII->getNamedOperand(*CI.I, AMDGPU::OpName::sbase))
        .addImm(MergedOffset) // offset
        .addImm(CI.GLC0)      // glc
        .addImm(CI.DLC0)      // dlc
        .addMemOperand(combineKnownAdjacentMMOs(*MBB->getParent(), MMOa, MMOb));

  std::pair<unsigned, unsigned> SubRegIdx = getSubRegIdxs(CI);
  const unsigned SubRegIdx0 = std::get<0>(SubRegIdx);
  const unsigned SubRegIdx1 = std::get<1>(SubRegIdx);

  // Copy to the old destination registers.
  const MCInstrDesc &CopyDesc = TII->get(TargetOpcode::COPY);
  const auto *Dest0 = TII->getNamedOperand(*CI.I, AMDGPU::OpName::sdst);
  const auto *Dest1 = TII->getNamedOperand(*CI.Paired, AMDGPU::OpName::sdst);

  BuildMI(*MBB, CI.Paired, DL, CopyDesc)
      .add(*Dest0) // Copy to same destination including flags and sub reg.
      .addReg(DestReg, 0, SubRegIdx0);
  MachineInstr *Copy1 = BuildMI(*MBB, CI.Paired, DL, CopyDesc)
                            .add(*Dest1)
                            .addReg(DestReg, RegState::Kill, SubRegIdx1);

  moveInstsAfter(Copy1, CI.InstsToMove);

  CI.I->eraseFromParent();
  CI.Paired->eraseFromParent();
  return New;
}

MachineBasicBlock::iterator
SILoadStoreOptimizer::mergeBufferLoadPair(CombineInfo &CI) {
  MachineBasicBlock *MBB = CI.I->getParent();
  DebugLoc DL = CI.I->getDebugLoc();

  const unsigned Opcode = getNewOpcode(CI);

  const TargetRegisterClass *SuperRC = getTargetRegisterClass(CI);

  // Copy to the new source register.
  Register DestReg = MRI->createVirtualRegister(SuperRC);
  unsigned MergedOffset = std::min(CI.Offset0, CI.Offset1);

  auto MIB = BuildMI(*MBB, CI.Paired, DL, TII->get(Opcode), DestReg);

  const unsigned Regs = getRegs(Opcode, *TII);

  if (Regs & VADDR)
    MIB.add(*TII->getNamedOperand(*CI.I, AMDGPU::OpName::vaddr));

  // It shouldn't be possible to get this far if the two instructions
  // don't have a single memoperand, because MachineInstr::mayAlias()
  // will return true if this is the case.
  assert(CI.I->hasOneMemOperand() && CI.Paired->hasOneMemOperand());

  const MachineMemOperand *MMOa = *CI.I->memoperands_begin();
  const MachineMemOperand *MMOb = *CI.Paired->memoperands_begin();

  MachineInstr *New =
    MIB.add(*TII->getNamedOperand(*CI.I, AMDGPU::OpName::srsrc))
        .add(*TII->getNamedOperand(*CI.I, AMDGPU::OpName::soffset))
        .addImm(MergedOffset) // offset
        .addImm(CI.GLC0)      // glc
        .addImm(CI.SLC0)      // slc
        .addImm(0)            // tfe
        .addImm(CI.DLC0)      // dlc
        .addImm(0)            // swz
        .addMemOperand(combineKnownAdjacentMMOs(*MBB->getParent(), MMOa, MMOb));

  std::pair<unsigned, unsigned> SubRegIdx = getSubRegIdxs(CI);
  const unsigned SubRegIdx0 = std::get<0>(SubRegIdx);
  const unsigned SubRegIdx1 = std::get<1>(SubRegIdx);

  // Copy to the old destination registers.
  const MCInstrDesc &CopyDesc = TII->get(TargetOpcode::COPY);
  const auto *Dest0 = TII->getNamedOperand(*CI.I, AMDGPU::OpName::vdata);
  const auto *Dest1 = TII->getNamedOperand(*CI.Paired, AMDGPU::OpName::vdata);

  BuildMI(*MBB, CI.Paired, DL, CopyDesc)
      .add(*Dest0) // Copy to same destination including flags and sub reg.
      .addReg(DestReg, 0, SubRegIdx0);
  MachineInstr *Copy1 = BuildMI(*MBB, CI.Paired, DL, CopyDesc)
                            .add(*Dest1)
                            .addReg(DestReg, RegState::Kill, SubRegIdx1);

  moveInstsAfter(Copy1, CI.InstsToMove);

  CI.I->eraseFromParent();
  CI.Paired->eraseFromParent();
  return New;
}

MachineBasicBlock::iterator
SILoadStoreOptimizer::mergeTBufferLoadPair(CombineInfo &CI) {
  MachineBasicBlock *MBB = CI.I->getParent();
  DebugLoc DL = CI.I->getDebugLoc();

  const unsigned Opcode = getNewOpcode(CI);

  const TargetRegisterClass *SuperRC = getTargetRegisterClass(CI);

  // Copy to the new source register.
  Register DestReg = MRI->createVirtualRegister(SuperRC);
  unsigned MergedOffset = std::min(CI.Offset0, CI.Offset1);

  auto MIB = BuildMI(*MBB, CI.Paired, DL, TII->get(Opcode), DestReg);

  const unsigned Regs = getRegs(Opcode, *TII);

  if (Regs & VADDR)
    MIB.add(*TII->getNamedOperand(*CI.I, AMDGPU::OpName::vaddr));

  unsigned JoinedFormat =
      getBufferFormatWithCompCount(CI.Format0, CI.Width0 + CI.Width1, *STI);

  // It shouldn't be possible to get this far if the two instructions
  // don't have a single memoperand, because MachineInstr::mayAlias()
  // will return true if this is the case.
  assert(CI.I->hasOneMemOperand() && CI.Paired->hasOneMemOperand());

  const MachineMemOperand *MMOa = *CI.I->memoperands_begin();
  const MachineMemOperand *MMOb = *CI.Paired->memoperands_begin();

  MachineInstr *New =
      MIB.add(*TII->getNamedOperand(*CI.I, AMDGPU::OpName::srsrc))
          .add(*TII->getNamedOperand(*CI.I, AMDGPU::OpName::soffset))
          .addImm(MergedOffset) // offset
          .addImm(JoinedFormat) // format
          .addImm(CI.GLC0)      // glc
          .addImm(CI.SLC0)      // slc
          .addImm(0)            // tfe
          .addImm(CI.DLC0)      // dlc
          .addImm(0)            // swz
          .addMemOperand(
              combineKnownAdjacentMMOs(*MBB->getParent(), MMOa, MMOb));

  std::pair<unsigned, unsigned> SubRegIdx = getSubRegIdxs(CI);
  const unsigned SubRegIdx0 = std::get<0>(SubRegIdx);
  const unsigned SubRegIdx1 = std::get<1>(SubRegIdx);

  // Copy to the old destination registers.
  const MCInstrDesc &CopyDesc = TII->get(TargetOpcode::COPY);
  const auto *Dest0 = TII->getNamedOperand(*CI.I, AMDGPU::OpName::vdata);
  const auto *Dest1 = TII->getNamedOperand(*CI.Paired, AMDGPU::OpName::vdata);

  BuildMI(*MBB, CI.Paired, DL, CopyDesc)
      .add(*Dest0) // Copy to same destination including flags and sub reg.
      .addReg(DestReg, 0, SubRegIdx0);
  MachineInstr *Copy1 = BuildMI(*MBB, CI.Paired, DL, CopyDesc)
                            .add(*Dest1)
                            .addReg(DestReg, RegState::Kill, SubRegIdx1);

  moveInstsAfter(Copy1, CI.InstsToMove);

  CI.I->eraseFromParent();
  CI.Paired->eraseFromParent();
  return New;
}

MachineBasicBlock::iterator
SILoadStoreOptimizer::mergeTBufferStorePair(CombineInfo &CI) {
  MachineBasicBlock *MBB = CI.I->getParent();
  DebugLoc DL = CI.I->getDebugLoc();

  const unsigned Opcode = getNewOpcode(CI);

  std::pair<unsigned, unsigned> SubRegIdx = getSubRegIdxs(CI);
  const unsigned SubRegIdx0 = std::get<0>(SubRegIdx);
  const unsigned SubRegIdx1 = std::get<1>(SubRegIdx);

  // Copy to the new source register.
  const TargetRegisterClass *SuperRC = getTargetRegisterClass(CI);
  Register SrcReg = MRI->createVirtualRegister(SuperRC);

  const auto *Src0 = TII->getNamedOperand(*CI.I, AMDGPU::OpName::vdata);
  const auto *Src1 = TII->getNamedOperand(*CI.Paired, AMDGPU::OpName::vdata);

  BuildMI(*MBB, CI.Paired, DL, TII->get(AMDGPU::REG_SEQUENCE), SrcReg)
      .add(*Src0)
      .addImm(SubRegIdx0)
      .add(*Src1)
      .addImm(SubRegIdx1);

  auto MIB = BuildMI(*MBB, CI.Paired, DL, TII->get(Opcode))
                 .addReg(SrcReg, RegState::Kill);

  const unsigned Regs = getRegs(Opcode, *TII);

  if (Regs & VADDR)
    MIB.add(*TII->getNamedOperand(*CI.I, AMDGPU::OpName::vaddr));

  unsigned JoinedFormat =
      getBufferFormatWithCompCount(CI.Format0, CI.Width0 + CI.Width1, *STI);

  // It shouldn't be possible to get this far if the two instructions
  // don't have a single memoperand, because MachineInstr::mayAlias()
  // will return true if this is the case.
  assert(CI.I->hasOneMemOperand() && CI.Paired->hasOneMemOperand());

  const MachineMemOperand *MMOa = *CI.I->memoperands_begin();
  const MachineMemOperand *MMOb = *CI.Paired->memoperands_begin();

  MachineInstr *New =
      MIB.add(*TII->getNamedOperand(*CI.I, AMDGPU::OpName::srsrc))
          .add(*TII->getNamedOperand(*CI.I, AMDGPU::OpName::soffset))
          .addImm(std::min(CI.Offset0, CI.Offset1)) // offset
          .addImm(JoinedFormat)                     // format
          .addImm(CI.GLC0)                          // glc
          .addImm(CI.SLC0)                          // slc
          .addImm(0)                                // tfe
          .addImm(CI.DLC0)                          // dlc
          .addImm(0)                                // swz
          .addMemOperand(
              combineKnownAdjacentMMOs(*MBB->getParent(), MMOa, MMOb));

  moveInstsAfter(MIB, CI.InstsToMove);

  CI.I->eraseFromParent();
  CI.Paired->eraseFromParent();
  return New;
}

unsigned SILoadStoreOptimizer::getNewOpcode(const CombineInfo &CI) {
  const unsigned Width = CI.Width0 + CI.Width1;

  switch (CI.InstClass) {
  default:
    assert(CI.InstClass == BUFFER_LOAD || CI.InstClass == BUFFER_STORE);
    // FIXME: Handle d16 correctly
    return AMDGPU::getMUBUFOpcode(AMDGPU::getMUBUFBaseOpcode(CI.I->getOpcode()),
                                  Width);
  case TBUFFER_LOAD:
  case TBUFFER_STORE:
    return AMDGPU::getMTBUFOpcode(AMDGPU::getMTBUFBaseOpcode(CI.I->getOpcode()),
                                  Width);

  case UNKNOWN:
    llvm_unreachable("Unknown instruction class");
  case S_BUFFER_LOAD_IMM:
    switch (Width) {
    default:
      return 0;
    case 2:
      return AMDGPU::S_BUFFER_LOAD_DWORDX2_IMM;
    case 4:
      return AMDGPU::S_BUFFER_LOAD_DWORDX4_IMM;
    }
  case MIMG:
    assert("No overlaps" && (countPopulation(CI.DMask0 | CI.DMask1) == Width));
    return AMDGPU::getMaskedMIMGOp(CI.I->getOpcode(), Width);
  }
}

std::pair<unsigned, unsigned>
SILoadStoreOptimizer::getSubRegIdxs(const CombineInfo &CI) {

  if (CI.Width0 == 0 || CI.Width1 == 0 || CI.Width0 + CI.Width1 > 4)
    return std::make_pair(0, 0);

  bool ReverseOrder;
  if (CI.InstClass == MIMG) {
    assert((countPopulation(CI.DMask0 | CI.DMask1) == CI.Width0 + CI.Width1) &&
           "No overlaps");
    ReverseOrder = CI.DMask0 > CI.DMask1;
  } else
    ReverseOrder = CI.Offset0 > CI.Offset1;

  static const unsigned Idxs[4][4] = {
      {AMDGPU::sub0, AMDGPU::sub0_sub1, AMDGPU::sub0_sub1_sub2, AMDGPU::sub0_sub1_sub2_sub3},
      {AMDGPU::sub1, AMDGPU::sub1_sub2, AMDGPU::sub1_sub2_sub3, 0},
      {AMDGPU::sub2, AMDGPU::sub2_sub3, 0, 0},
      {AMDGPU::sub3, 0, 0, 0},
  };
  unsigned Idx0;
  unsigned Idx1;

  assert(CI.Width0 >= 1 && CI.Width0 <= 3);
  assert(CI.Width1 >= 1 && CI.Width1 <= 3);

  if (ReverseOrder) {
    Idx1 = Idxs[0][CI.Width1 - 1];
    Idx0 = Idxs[CI.Width1][CI.Width0 - 1];
  } else {
    Idx0 = Idxs[0][CI.Width0 - 1];
    Idx1 = Idxs[CI.Width0][CI.Width1 - 1];
  }

  return std::make_pair(Idx0, Idx1);
}

const TargetRegisterClass *
SILoadStoreOptimizer::getTargetRegisterClass(const CombineInfo &CI) {
  if (CI.InstClass == S_BUFFER_LOAD_IMM) {
    switch (CI.Width0 + CI.Width1) {
    default:
      return nullptr;
    case 2:
      return &AMDGPU::SReg_64_XEXECRegClass;
    case 4:
      return &AMDGPU::SGPR_128RegClass;
    case 8:
      return &AMDGPU::SReg_256RegClass;
    case 16:
      return &AMDGPU::SReg_512RegClass;
    }
  } else {
    switch (CI.Width0 + CI.Width1) {
    default:
      return nullptr;
    case 2:
      return &AMDGPU::VReg_64RegClass;
    case 3:
      return &AMDGPU::VReg_96RegClass;
    case 4:
      return &AMDGPU::VReg_128RegClass;
    }
  }
}

MachineBasicBlock::iterator
SILoadStoreOptimizer::mergeBufferStorePair(CombineInfo &CI) {
  MachineBasicBlock *MBB = CI.I->getParent();
  DebugLoc DL = CI.I->getDebugLoc();

  const unsigned Opcode = getNewOpcode(CI);

  std::pair<unsigned, unsigned> SubRegIdx = getSubRegIdxs(CI);
  const unsigned SubRegIdx0 = std::get<0>(SubRegIdx);
  const unsigned SubRegIdx1 = std::get<1>(SubRegIdx);

  // Copy to the new source register.
  const TargetRegisterClass *SuperRC = getTargetRegisterClass(CI);
  Register SrcReg = MRI->createVirtualRegister(SuperRC);

  const auto *Src0 = TII->getNamedOperand(*CI.I, AMDGPU::OpName::vdata);
  const auto *Src1 = TII->getNamedOperand(*CI.Paired, AMDGPU::OpName::vdata);

  BuildMI(*MBB, CI.Paired, DL, TII->get(AMDGPU::REG_SEQUENCE), SrcReg)
      .add(*Src0)
      .addImm(SubRegIdx0)
      .add(*Src1)
      .addImm(SubRegIdx1);

  auto MIB = BuildMI(*MBB, CI.Paired, DL, TII->get(Opcode))
                 .addReg(SrcReg, RegState::Kill);

  const unsigned Regs = getRegs(Opcode, *TII);

  if (Regs & VADDR)
    MIB.add(*TII->getNamedOperand(*CI.I, AMDGPU::OpName::vaddr));


  // It shouldn't be possible to get this far if the two instructions
  // don't have a single memoperand, because MachineInstr::mayAlias()
  // will return true if this is the case.
  assert(CI.I->hasOneMemOperand() && CI.Paired->hasOneMemOperand());

  const MachineMemOperand *MMOa = *CI.I->memoperands_begin();
  const MachineMemOperand *MMOb = *CI.Paired->memoperands_begin();

  MachineInstr *New =
    MIB.add(*TII->getNamedOperand(*CI.I, AMDGPU::OpName::srsrc))
        .add(*TII->getNamedOperand(*CI.I, AMDGPU::OpName::soffset))
        .addImm(std::min(CI.Offset0, CI.Offset1)) // offset
        .addImm(CI.GLC0)      // glc
        .addImm(CI.SLC0)      // slc
        .addImm(0)            // tfe
        .addImm(CI.DLC0)      // dlc
        .addImm(0)            // swz
        .addMemOperand(combineKnownAdjacentMMOs(*MBB->getParent(), MMOa, MMOb));

  moveInstsAfter(MIB, CI.InstsToMove);

  CI.I->eraseFromParent();
  CI.Paired->eraseFromParent();
  return New;
}

MachineOperand
SILoadStoreOptimizer::createRegOrImm(int32_t Val, MachineInstr &MI) const {
  APInt V(32, Val, true);
  if (TII->isInlineConstant(V))
    return MachineOperand::CreateImm(Val);

  Register Reg = MRI->createVirtualRegister(&AMDGPU::SReg_32RegClass);
  MachineInstr *Mov =
  BuildMI(*MI.getParent(), MI.getIterator(), MI.getDebugLoc(),
          TII->get(AMDGPU::S_MOV_B32), Reg)
    .addImm(Val);
  (void)Mov;
  LLVM_DEBUG(dbgs() << "    "; Mov->dump());
  return MachineOperand::CreateReg(Reg, false);
}

// Compute base address using Addr and return the final register.
unsigned SILoadStoreOptimizer::computeBase(MachineInstr &MI,
                                           const MemAddress &Addr) const {
  MachineBasicBlock *MBB = MI.getParent();
  MachineBasicBlock::iterator MBBI = MI.getIterator();
  DebugLoc DL = MI.getDebugLoc();

  assert((TRI->getRegSizeInBits(Addr.Base.LoReg, *MRI) == 32 ||
          Addr.Base.LoSubReg) &&
         "Expected 32-bit Base-Register-Low!!");

  assert((TRI->getRegSizeInBits(Addr.Base.HiReg, *MRI) == 32 ||
          Addr.Base.HiSubReg) &&
         "Expected 32-bit Base-Register-Hi!!");

  LLVM_DEBUG(dbgs() << "  Re-Computed Anchor-Base:\n");
  MachineOperand OffsetLo = createRegOrImm(static_cast<int32_t>(Addr.Offset), MI);
  MachineOperand OffsetHi =
    createRegOrImm(static_cast<int32_t>(Addr.Offset >> 32), MI);

  const auto *CarryRC = TRI->getRegClass(AMDGPU::SReg_1_XEXECRegClassID);
  Register CarryReg = MRI->createVirtualRegister(CarryRC);
  Register DeadCarryReg = MRI->createVirtualRegister(CarryRC);

  Register DestSub0 = MRI->createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  Register DestSub1 = MRI->createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  MachineInstr *LoHalf =
    BuildMI(*MBB, MBBI, DL, TII->get(AMDGPU::V_ADD_I32_e64), DestSub0)
      .addReg(CarryReg, RegState::Define)
      .addReg(Addr.Base.LoReg, 0, Addr.Base.LoSubReg)
      .add(OffsetLo)
      .addImm(0); // clamp bit
  (void)LoHalf;
  LLVM_DEBUG(dbgs() << "    "; LoHalf->dump(););

  MachineInstr *HiHalf =
  BuildMI(*MBB, MBBI, DL, TII->get(AMDGPU::V_ADDC_U32_e64), DestSub1)
    .addReg(DeadCarryReg, RegState::Define | RegState::Dead)
    .addReg(Addr.Base.HiReg, 0, Addr.Base.HiSubReg)
    .add(OffsetHi)
    .addReg(CarryReg, RegState::Kill)
    .addImm(0); // clamp bit
  (void)HiHalf;
  LLVM_DEBUG(dbgs() << "    "; HiHalf->dump(););

  Register FullDestReg = MRI->createVirtualRegister(&AMDGPU::VReg_64RegClass);
  MachineInstr *FullBase =
    BuildMI(*MBB, MBBI, DL, TII->get(TargetOpcode::REG_SEQUENCE), FullDestReg)
      .addReg(DestSub0)
      .addImm(AMDGPU::sub0)
      .addReg(DestSub1)
      .addImm(AMDGPU::sub1);
  (void)FullBase;
  LLVM_DEBUG(dbgs() << "    "; FullBase->dump(); dbgs() << "\n";);

  return FullDestReg;
}

// Update base and offset with the NewBase and NewOffset in MI.
void SILoadStoreOptimizer::updateBaseAndOffset(MachineInstr &MI,
                                               unsigned NewBase,
                                               int32_t NewOffset) const {
  auto Base = TII->getNamedOperand(MI, AMDGPU::OpName::vaddr);
  Base->setReg(NewBase);
  Base->setIsKill(false);
  TII->getNamedOperand(MI, AMDGPU::OpName::offset)->setImm(NewOffset);
}

Optional<int32_t>
SILoadStoreOptimizer::extractConstOffset(const MachineOperand &Op) const {
  if (Op.isImm())
    return Op.getImm();

  if (!Op.isReg())
    return None;

  MachineInstr *Def = MRI->getUniqueVRegDef(Op.getReg());
  if (!Def || Def->getOpcode() != AMDGPU::S_MOV_B32 ||
      !Def->getOperand(1).isImm())
    return None;

  return Def->getOperand(1).getImm();
}

// Analyze Base and extracts:
//  - 32bit base registers, subregisters
//  - 64bit constant offset
// Expecting base computation as:
//   %OFFSET0:sgpr_32 = S_MOV_B32 8000
//   %LO:vgpr_32, %c:sreg_64_xexec =
//       V_ADD_I32_e64 %BASE_LO:vgpr_32, %103:sgpr_32,
//   %HI:vgpr_32, = V_ADDC_U32_e64 %BASE_HI:vgpr_32, 0, killed %c:sreg_64_xexec
//   %Base:vreg_64 =
//       REG_SEQUENCE %LO:vgpr_32, %subreg.sub0, %HI:vgpr_32, %subreg.sub1
void SILoadStoreOptimizer::processBaseWithConstOffset(const MachineOperand &Base,
                                                      MemAddress &Addr) const {
  if (!Base.isReg())
    return;

  MachineInstr *Def = MRI->getUniqueVRegDef(Base.getReg());
  if (!Def || Def->getOpcode() != AMDGPU::REG_SEQUENCE
      || Def->getNumOperands() != 5)
    return;

  MachineOperand BaseLo = Def->getOperand(1);
  MachineOperand BaseHi = Def->getOperand(3);
  if (!BaseLo.isReg() || !BaseHi.isReg())
    return;

  MachineInstr *BaseLoDef = MRI->getUniqueVRegDef(BaseLo.getReg());
  MachineInstr *BaseHiDef = MRI->getUniqueVRegDef(BaseHi.getReg());

  if (!BaseLoDef || BaseLoDef->getOpcode() != AMDGPU::V_ADD_I32_e64 ||
      !BaseHiDef || BaseHiDef->getOpcode() != AMDGPU::V_ADDC_U32_e64)
    return;

  const auto *Src0 = TII->getNamedOperand(*BaseLoDef, AMDGPU::OpName::src0);
  const auto *Src1 = TII->getNamedOperand(*BaseLoDef, AMDGPU::OpName::src1);

  auto Offset0P = extractConstOffset(*Src0);
  if (Offset0P)
    BaseLo = *Src1;
  else {
    if (!(Offset0P = extractConstOffset(*Src1)))
      return;
    BaseLo = *Src0;
  }

  Src0 = TII->getNamedOperand(*BaseHiDef, AMDGPU::OpName::src0);
  Src1 = TII->getNamedOperand(*BaseHiDef, AMDGPU::OpName::src1);

  if (Src0->isImm())
    std::swap(Src0, Src1);

  if (!Src1->isImm())
    return;

  uint64_t Offset1 = Src1->getImm();
  BaseHi = *Src0;

  Addr.Base.LoReg = BaseLo.getReg();
  Addr.Base.HiReg = BaseHi.getReg();
  Addr.Base.LoSubReg = BaseLo.getSubReg();
  Addr.Base.HiSubReg = BaseHi.getSubReg();
  Addr.Offset = (*Offset0P & 0x00000000ffffffff) | (Offset1 << 32);
}

bool SILoadStoreOptimizer::promoteConstantOffsetToImm(
    MachineInstr &MI,
    MemInfoMap &Visited,
    SmallPtrSet<MachineInstr *, 4> &AnchorList) const {

  if (!(MI.mayLoad() ^ MI.mayStore()))
    return false;

  // TODO: Support flat and scratch.
  if (AMDGPU::getGlobalSaddrOp(MI.getOpcode()) < 0)
    return false;

  if (MI.mayLoad() && TII->getNamedOperand(MI, AMDGPU::OpName::vdata) != NULL)
    return false;

  if (AnchorList.count(&MI))
    return false;

  LLVM_DEBUG(dbgs() << "\nTryToPromoteConstantOffsetToImmFor "; MI.dump());

  if (TII->getNamedOperand(MI, AMDGPU::OpName::offset)->getImm()) {
    LLVM_DEBUG(dbgs() << "  Const-offset is already promoted.\n";);
    return false;
  }

  // Step1: Find the base-registers and a 64bit constant offset.
  MachineOperand &Base = *TII->getNamedOperand(MI, AMDGPU::OpName::vaddr);
  MemAddress MAddr;
  if (Visited.find(&MI) == Visited.end()) {
    processBaseWithConstOffset(Base, MAddr);
    Visited[&MI] = MAddr;
  } else
    MAddr = Visited[&MI];

  if (MAddr.Offset == 0) {
    LLVM_DEBUG(dbgs() << "  Failed to extract constant-offset or there are no"
                         " constant offsets that can be promoted.\n";);
    return false;
  }

  LLVM_DEBUG(dbgs() << "  BASE: {" << MAddr.Base.HiReg << ", "
             << MAddr.Base.LoReg << "} Offset: " << MAddr.Offset << "\n\n";);

  // Step2: Traverse through MI's basic block and find an anchor(that has the
  // same base-registers) with the highest 13bit distance from MI's offset.
  // E.g. (64bit loads)
  // bb:
  //   addr1 = &a + 4096;   load1 = load(addr1,  0)
  //   addr2 = &a + 6144;   load2 = load(addr2,  0)
  //   addr3 = &a + 8192;   load3 = load(addr3,  0)
  //   addr4 = &a + 10240;  load4 = load(addr4,  0)
  //   addr5 = &a + 12288;  load5 = load(addr5,  0)
  //
  // Starting from the first load, the optimization will try to find a new base
  // from which (&a + 4096) has 13 bit distance. Both &a + 6144 and &a + 8192
  // has 13bit distance from &a + 4096. The heuristic considers &a + 8192
  // as the new-base(anchor) because of the maximum distance which can
  // accomodate more intermediate bases presumeably.
  //
  // Step3: move (&a + 8192) above load1. Compute and promote offsets from
  // (&a + 8192) for load1, load2, load4.
  //   addr = &a + 8192
  //   load1 = load(addr,       -4096)
  //   load2 = load(addr,       -2048)
  //   load3 = load(addr,       0)
  //   load4 = load(addr,       2048)
  //   addr5 = &a + 12288;  load5 = load(addr5,  0)
  //
  MachineInstr *AnchorInst = nullptr;
  MemAddress AnchorAddr;
  uint32_t MaxDist = std::numeric_limits<uint32_t>::min();
  SmallVector<std::pair<MachineInstr *, int64_t>, 4> InstsWCommonBase;

  MachineBasicBlock *MBB = MI.getParent();
  MachineBasicBlock::iterator E = MBB->end();
  MachineBasicBlock::iterator MBBI = MI.getIterator();
  ++MBBI;
  const SITargetLowering *TLI =
    static_cast<const SITargetLowering *>(STM->getTargetLowering());

  for ( ; MBBI != E; ++MBBI) {
    MachineInstr &MINext = *MBBI;
    // TODO: Support finding an anchor(with same base) from store addresses or
    // any other load addresses where the opcodes are different.
    if (MINext.getOpcode() != MI.getOpcode() ||
        TII->getNamedOperand(MINext, AMDGPU::OpName::offset)->getImm())
      continue;

    const MachineOperand &BaseNext =
      *TII->getNamedOperand(MINext, AMDGPU::OpName::vaddr);
    MemAddress MAddrNext;
    if (Visited.find(&MINext) == Visited.end()) {
      processBaseWithConstOffset(BaseNext, MAddrNext);
      Visited[&MINext] = MAddrNext;
    } else
      MAddrNext = Visited[&MINext];

    if (MAddrNext.Base.LoReg != MAddr.Base.LoReg ||
        MAddrNext.Base.HiReg != MAddr.Base.HiReg ||
        MAddrNext.Base.LoSubReg != MAddr.Base.LoSubReg ||
        MAddrNext.Base.HiSubReg != MAddr.Base.HiSubReg)
      continue;

    InstsWCommonBase.push_back(std::make_pair(&MINext, MAddrNext.Offset));

    int64_t Dist = MAddr.Offset - MAddrNext.Offset;
    TargetLoweringBase::AddrMode AM;
    AM.HasBaseReg = true;
    AM.BaseOffs = Dist;
    if (TLI->isLegalGlobalAddressingMode(AM) &&
        (uint32_t)std::abs(Dist) > MaxDist) {
      MaxDist = std::abs(Dist);

      AnchorAddr = MAddrNext;
      AnchorInst = &MINext;
    }
  }

  if (AnchorInst) {
    LLVM_DEBUG(dbgs() << "  Anchor-Inst(with max-distance from Offset): ";
               AnchorInst->dump());
    LLVM_DEBUG(dbgs() << "  Anchor-Offset from BASE: "
               <<  AnchorAddr.Offset << "\n\n");

    // Instead of moving up, just re-compute anchor-instruction's base address.
    unsigned Base = computeBase(MI, AnchorAddr);

    updateBaseAndOffset(MI, Base, MAddr.Offset - AnchorAddr.Offset);
    LLVM_DEBUG(dbgs() << "  After promotion: "; MI.dump(););

    for (auto P : InstsWCommonBase) {
      TargetLoweringBase::AddrMode AM;
      AM.HasBaseReg = true;
      AM.BaseOffs = P.second - AnchorAddr.Offset;

      if (TLI->isLegalGlobalAddressingMode(AM)) {
        LLVM_DEBUG(dbgs() << "  Promote Offset(" << P.second;
                   dbgs() << ")"; P.first->dump());
        updateBaseAndOffset(*P.first, Base, P.second - AnchorAddr.Offset);
        LLVM_DEBUG(dbgs() << "     After promotion: "; P.first->dump());
      }
    }
    AnchorList.insert(AnchorInst);
    return true;
  }

  return false;
}

void SILoadStoreOptimizer::addInstToMergeableList(const CombineInfo &CI,
                 std::list<std::list<CombineInfo> > &MergeableInsts) const {
  for (std::list<CombineInfo> &AddrList : MergeableInsts) {
    if (AddrList.front().InstClass == CI.InstClass &&
        AddrList.front().hasSameBaseAddress(*CI.I)) {
      AddrList.emplace_back(CI);
      return;
    }
  }

  // Base address not found, so add a new list.
  MergeableInsts.emplace_back(1, CI);
}

bool SILoadStoreOptimizer::collectMergeableInsts(MachineBasicBlock &MBB,
                 std::list<std::list<CombineInfo> > &MergeableInsts) const {
  bool Modified = false;
  // Contain the list
  MemInfoMap Visited;
  // Contains the list of instructions for which constant offsets are being
  // promoted to the IMM.
  SmallPtrSet<MachineInstr *, 4> AnchorList;

  // Sort potential mergeable instructions into lists.  One list per base address.
  for (MachineInstr &MI : MBB.instrs()) {
    // We run this before checking if an address is mergeable, because it can produce
    // better code even if the instructions aren't mergeable.
    if (promoteConstantOffsetToImm(MI, Visited, AnchorList))
      Modified = true;

    const InstClassEnum InstClass = getInstClass(MI.getOpcode(), *TII);
    if (InstClass == UNKNOWN)
      continue;

    // Don't combine if volatile.
    if (MI.hasOrderedMemoryRef())
      continue;

    CombineInfo CI;
    CI.setMI(MI, *TII, *STM);

    if (!CI.hasMergeableAddress(*MRI))
      continue;

    addInstToMergeableList(CI, MergeableInsts);
  }
  return Modified;
}

// Scan through looking for adjacent LDS operations with constant offsets from
// the same base register. We rely on the scheduler to do the hard work of
// clustering nearby loads, and assume these are all adjacent.
bool SILoadStoreOptimizer::optimizeBlock(
                       std::list<std::list<CombineInfo> > &MergeableInsts) {
  bool Modified = false;

  for (std::list<CombineInfo> &MergeList : MergeableInsts) {
    if (MergeList.size() < 2)
      continue;

    bool OptimizeListAgain = false;
    if (!optimizeInstsWithSameBaseAddr(MergeList, OptimizeListAgain)) {
      // We weren't able to make any changes, so clear the list so we don't
      // process the same instructions the next time we try to optimize this
      // block.
      MergeList.clear();
      continue;
    }

    // We made changes, but also determined that there were no more optimization
    // opportunities, so we don't need to reprocess the list
    if (!OptimizeListAgain)
      MergeList.clear();

    OptimizeAgain |= OptimizeListAgain;
    Modified = true;
  }
  return Modified;
}

void
SILoadStoreOptimizer::removeCombinedInst(std::list<CombineInfo> &MergeList,
                                         const MachineInstr &MI) {

  for (auto CI = MergeList.begin(), E = MergeList.end(); CI != E; ++CI) {
    if (&*CI->I == &MI) {
      MergeList.erase(CI);
      return;
    }
  }
}

bool
SILoadStoreOptimizer::optimizeInstsWithSameBaseAddr(
                                          std::list<CombineInfo> &MergeList,
                                          bool &OptimizeListAgain) {
  bool Modified = false;
  for (auto I = MergeList.begin(); I != MergeList.end(); ++I) {
    CombineInfo &CI = *I;

    switch (CI.InstClass) {
    default:
      break;
    case DS_READ:
      if (findMatchingInst(CI)) {
        Modified = true;
        removeCombinedInst(MergeList, *CI.Paired);
        MachineBasicBlock::iterator NewMI = mergeRead2Pair(CI);
        CI.setMI(NewMI, *TII, *STM);
      }
      break;
    case DS_WRITE:
      if (findMatchingInst(CI)) {
        Modified = true;
        removeCombinedInst(MergeList, *CI.Paired);
        MachineBasicBlock::iterator NewMI = mergeWrite2Pair(CI);
        CI.setMI(NewMI, *TII, *STM);
      }
      break;
    case S_BUFFER_LOAD_IMM:
      if (findMatchingInst(CI)) {
        Modified = true;
        removeCombinedInst(MergeList, *CI.Paired);
        MachineBasicBlock::iterator NewMI = mergeSBufferLoadImmPair(CI);
        CI.setMI(NewMI, *TII, *STM);
        OptimizeListAgain |= (CI.Width0 + CI.Width1) < 16;
      }
      break;
    case BUFFER_LOAD:
      if (findMatchingInst(CI)) {
        Modified = true;
        removeCombinedInst(MergeList, *CI.Paired);
        MachineBasicBlock::iterator NewMI = mergeBufferLoadPair(CI);
        CI.setMI(NewMI, *TII, *STM);
        OptimizeListAgain |= (CI.Width0 + CI.Width1) < 4;
      }
      break;
    case BUFFER_STORE:
      if (findMatchingInst(CI)) {
        Modified = true;
        removeCombinedInst(MergeList, *CI.Paired);
        MachineBasicBlock::iterator NewMI = mergeBufferStorePair(CI);
        CI.setMI(NewMI, *TII, *STM);
        OptimizeListAgain |= (CI.Width0 + CI.Width1) < 4;
      }
      break;
    case MIMG:
      if (findMatchingInst(CI)) {
        Modified = true;
        removeCombinedInst(MergeList, *CI.Paired);
        MachineBasicBlock::iterator NewMI = mergeImagePair(CI);
        CI.setMI(NewMI, *TII, *STM);
        OptimizeListAgain |= (CI.Width0 + CI.Width1) < 4;
      }
      break;
    case TBUFFER_LOAD:
      if (findMatchingInst(CI)) {
        Modified = true;
        removeCombinedInst(MergeList, *CI.Paired);
        MachineBasicBlock::iterator NewMI = mergeTBufferLoadPair(CI);
        CI.setMI(NewMI, *TII, *STM);
        OptimizeListAgain |= (CI.Width0 + CI.Width1) < 4;
      }
      break;
    case TBUFFER_STORE:
      if (findMatchingInst(CI)) {
        Modified = true;
        removeCombinedInst(MergeList, *CI.Paired);
        MachineBasicBlock::iterator NewMI = mergeTBufferStorePair(CI);
        CI.setMI(NewMI, *TII, *STM);
        OptimizeListAgain |= (CI.Width0 + CI.Width1) < 4;
      }
      break;
    }
    // Clear the InstsToMove after we have finished searching so we don't have
    // stale values left over if we search for this CI again in another pass
    // over the block.
    CI.InstsToMove.clear();
  }

  return Modified;
}

bool SILoadStoreOptimizer::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  STM = &MF.getSubtarget<GCNSubtarget>();
  if (!STM->loadStoreOptEnabled())
    return false;

  TII = STM->getInstrInfo();
  TRI = &TII->getRegisterInfo();
  STI = &MF.getSubtarget<MCSubtargetInfo>();

  MRI = &MF.getRegInfo();
  AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();

  assert(MRI->isSSA() && "Must be run on SSA");

  LLVM_DEBUG(dbgs() << "Running SILoadStoreOptimizer\n");

  bool Modified = false;


  for (MachineBasicBlock &MBB : MF) {
    std::list<std::list<CombineInfo> > MergeableInsts;
    // First pass: Collect list of all instructions we know how to merge.
    Modified |= collectMergeableInsts(MBB, MergeableInsts);
    do {
      OptimizeAgain = false;
      Modified |= optimizeBlock(MergeableInsts);
    } while (OptimizeAgain);
  }

  return Modified;
}
