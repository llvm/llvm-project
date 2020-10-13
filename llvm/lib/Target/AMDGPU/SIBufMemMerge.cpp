//===-- SIBufMemMerge.cpp -------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
// This pass attempts to merge buffer memory operations
// For instance:
// s_buffer_load_dword s34, s[28:31], 0x9
// s_buffer_load_dword s35, s[28:31], 0xa
// ==>
// s_buffer_load_dwordx2 s[34:35], s[28:31], 0x9
//

#include "AMDGPU.h"
#include "AMDGPURegPressAnalysis.h"
#include "AMDGPUSubtarget.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
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
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include <cassert>
#include <iterator>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "si-bufmem-merge"

namespace {
  // \brief Struct representing the available values in the densemap
  struct SimpleMI {
    MachineInstr *Inst;
    int OriginalOrder;
    int Phase;
    
    static const int DWORD_SIZE_IN_BITS = 32;
    static const int BYTES_PER_DWORD = 4;

    SimpleMI(MachineInstr *MI) : Inst(MI), OriginalOrder(-1), Phase(-1) {
      assert((isSentinel() || canHandle(MI)) && "MachineInstr can't be handled");
    }
    SimpleMI(MachineInstr *MI, int CurrPhase) : Inst(MI), OriginalOrder(-1),
                                                Phase(CurrPhase) {
      assert((isSentinel() || canHandle(MI)) && "MachineInstr can't be handled");
    }
    
    bool isSentinel() const {
      return Inst == DenseMapInfo<MachineInstr *>::getEmptyKey() ||
      Inst == DenseMapInfo<MachineInstr *>::getTombstoneKey();
    }
    
    static bool canHandle(MachineInstr *Inst) {
      return
      Inst->getOpcode() == AMDGPU::S_BUFFER_LOAD_DWORD_IMM ||
      Inst->getOpcode() == AMDGPU::S_BUFFER_LOAD_DWORDX2_IMM ||
      Inst->getOpcode() == AMDGPU::S_BUFFER_LOAD_DWORDX4_IMM ||
      Inst->getOpcode() == AMDGPU::S_BUFFER_LOAD_DWORDX8_IMM;
    }
    
    unsigned getSBase(const SIInstrInfo *TII) const {
      return TII->getNamedOperand(*Inst, AMDGPU::OpName::sbase)->getReg();
    }
    unsigned getGLC(const SIInstrInfo *TII) const {
      return TII->getNamedOperand(*Inst, AMDGPU::OpName::glc)->getImm();
    }
    /// Get the offset encoded in the instruction
    unsigned getOffset(const SIInstrInfo *TII) const {
      return TII->getNamedOperand(*Inst, AMDGPU::OpName::offset)->getImm();
    }
    /// Get the offset in dwords
    unsigned getDwordOffset(const SIInstrInfo *TII, const GCNSubtarget *ST)
      const {
      unsigned Offset = TII->getNamedOperand(*Inst,
                                             AMDGPU::OpName::offset)->getImm();
      return Offset / AMDGPU::getSMRDEncodedOffset(*ST, BYTES_PER_DWORD, true).getValue();
    }
    MachineOperand *getDest(const SIInstrInfo *TII) const {
      return TII->getNamedOperand(*Inst, AMDGPU::OpName::sdst);
    }
    const TargetRegisterClass *getDestRC(const SIInstrInfo *TII) const {
      return TII->getOpRegClass(*Inst,
                                AMDGPU::getNamedOperandIdx(Inst->getOpcode(),
                                                           AMDGPU::OpName::sdst));
    }
    unsigned getSize(const SIInstrInfo *TII) const {
      const TargetRegisterClass *RC = getDestRC(TII);
      return TII->getRegisterInfo().getRegSizeInBits(*RC) / DWORD_SIZE_IN_BITS;
    }
  };
}

namespace {

class SIBufMemMerge : public MachineFunctionPass {

  
private:
  const GCNSubtarget *ST = nullptr;
  const SIInstrInfo *TII = nullptr;
  const SIRegisterInfo *TRI = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  AMDGPURegPressAnalysis *RP = nullptr;
  bool HighPressure = false;

public:
  static char ID;

  SIBufMemMerge() : MachineFunctionPass(ID) {
    initializeSIBufMemMergePass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "SI Memory operation coalescer";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<AMDGPURegPressAnalysis>();

    MachineFunctionPass::getAnalysisUsage(AU);
  }

  unsigned getMaxSize(unsigned int size);
  void processSubSection(const SmallVector<SimpleMI, 8> &Candidates,
                         unsigned StartIdx, unsigned EndIdx, unsigned Size,
                         const MachineOperand& SBase, const MachineOperand& GLC);
  bool processList(SmallVector<SmallVector<SimpleMI, 8>, 8> &CandidateList);
  bool optimizeBlock(MachineBasicBlock &MBB);
};

} // end anonymous namespace.

INITIALIZE_PASS_BEGIN(SIBufMemMerge, DEBUG_TYPE,
                      "SI Memory operation coalescer", false, false)
INITIALIZE_PASS_DEPENDENCY(AMDGPURegPressAnalysis)
INITIALIZE_PASS_END(SIBufMemMerge, DEBUG_TYPE,
                    "SI Memory operation coalescer", false, false)

char SIBufMemMerge::ID = 0;

char &llvm::SIBufMemMergeID = SIBufMemMerge::ID;

FunctionPass *llvm::createSIBufMemMergePass() {
  return new SIBufMemMerge();
}

namespace llvm {
template <> struct DenseMapInfo<SimpleMI> {
  static inline SimpleMI getEmptyKey() {
    return DenseMapInfo<MachineInstr *>::getEmptyKey();
  }
  static inline SimpleMI getTombstoneKey() {
    return DenseMapInfo<MachineInstr *>::getTombstoneKey();
  }
  static unsigned getHashValue(SimpleMI Val);
  static bool isEqual(SimpleMI LHS, SimpleMI RHS);
};
}

unsigned DenseMapInfo<SimpleMI>::getHashValue(SimpleMI Val) {
  MachineInstr *Inst = Val.Inst;
  MachineFunction *MF = Inst->getParent()->getParent();
  const GCNSubtarget &ST = MF->getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();
  
  if (Inst->getOpcode() == AMDGPU::S_BUFFER_LOAD_DWORD_IMM ||
      Inst->getOpcode() == AMDGPU::S_BUFFER_LOAD_DWORDX2_IMM ||
      Inst->getOpcode() == AMDGPU::S_BUFFER_LOAD_DWORDX4_IMM ||
      Inst->getOpcode() == AMDGPU::S_BUFFER_LOAD_DWORDX8_IMM) {
    const unsigned SBase = Val.getSBase(TII);
    const int64_t GLC = Val.getGLC(TII);
    const unsigned Size = Val.getSize(TII);
    const unsigned Phase = Val.Phase;
    
    return hash_combine(Inst->getOpcode(), SBase, GLC, Size, Phase);
  }
  
  llvm_unreachable_internal();
  return 0;
}

bool DenseMapInfo<SimpleMI>::isEqual(SimpleMI LHS, SimpleMI RHS) {
  MachineInstr *LHSI = LHS.Inst, *RHSI = RHS.Inst;
  
  if (LHS.isSentinel() || RHS.isSentinel())
    return LHSI == RHSI;
  
  if (LHSI->getOpcode() != RHSI->getOpcode())
    return false;
  if (LHSI->isIdenticalTo(*RHSI))
    return true;

  MachineFunction *MF = LHSI->getParent()->getParent();
  const GCNSubtarget &ST = MF->getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();
  
  // We are actually only interested in certain operands as we expect some to
  // differ
  if (LHSI->getOpcode() == AMDGPU::S_BUFFER_LOAD_DWORD_IMM ||
      LHSI->getOpcode() == AMDGPU::S_BUFFER_LOAD_DWORDX2_IMM ||
      LHSI->getOpcode() == AMDGPU::S_BUFFER_LOAD_DWORDX4_IMM ||
      LHSI->getOpcode() == AMDGPU::S_BUFFER_LOAD_DWORDX8_IMM) {
    return LHS.getSBase(TII) == RHS.getSBase(TII) &&
           LHS.getGLC(TII) == RHS.getGLC(TII) &&
           LHS.getSize(TII) == RHS.getSize(TII) &&
           LHS.Phase == RHS.Phase;
  }

  return false;
}

unsigned SIBufMemMerge::getMaxSize(unsigned int size) {
  // Limit the size to dwordx4 if HighPressure is set
  if (!HighPressure) {
    if (size >= 16) return 16;
    if (size >= 8) return 8;
  }

  if (size >= 4) return 4;
  if (size >= 2) return 2;
  if (size >= 1) return 1;
  return 0;
}

void SIBufMemMerge::processSubSection(const SmallVector<SimpleMI, 8> &Candidates,
                                      unsigned int StartIdx, unsigned int EndIdx,
                                      unsigned int Size,
                                      const llvm::MachineOperand &SBase,
                                      const llvm::MachineOperand &GLC) {
  
  // Loop through the indices creating the largest size we can from the
  // remaining elements
  while(Size) {
    LLVM_DEBUG(
      dbgs() << "Processing : \n";
      for( auto I = StartIdx; I <= EndIdx ; ++I)
        dbgs() << "\t" << *Candidates[I].Inst << "\n";);
    
    unsigned nextSize;
    const MCInstrDesc *MergeInstr = nullptr;
    // Sizes are all the same for a set of candidates
    unsigned eltSize = Candidates[StartIdx].getSize(TII);
    while ((nextSize = getMaxSize(Size))) {
      if (nextSize == eltSize) {
        // Don't need a replacement here as the replacement size and the
        // original size are the same
        LLVM_DEBUG(dbgs() << "Skipping coalesced replacement as same size: "
                     << *Candidates[StartIdx].Inst);
        ++StartIdx;
        Size -= nextSize;
        continue;
      }
      LLVM_DEBUG(dbgs() << "Creating new coalesced instruction of X" << nextSize
                   << "\n");
      const TargetRegisterClass *SuperRC = nullptr;
      switch(nextSize) {
        default:
          llvm_unreachable_internal();
          break;
        case 16:
          SuperRC = &AMDGPU::SGPR_512RegClass;
          MergeInstr = &TII->get(AMDGPU::S_BUFFER_LOAD_DWORDX16_IMM);
          break;
        case 8:
          SuperRC = &AMDGPU::SGPR_256RegClass;
          MergeInstr = &TII->get(AMDGPU::S_BUFFER_LOAD_DWORDX8_IMM);
          break;
        case 4:
          SuperRC = &AMDGPU::SGPR_128RegClass;
          MergeInstr = &TII->get(AMDGPU::S_BUFFER_LOAD_DWORDX4_IMM);
          break;
        case 2:
          SuperRC = &AMDGPU::SReg_64_XEXECRegClass;
          MergeInstr = &TII->get(AMDGPU::S_BUFFER_LOAD_DWORDX2_IMM);
          break;
        case 1:
          MergeInstr = &TII->get(AMDGPU::S_BUFFER_LOAD_DWORD_IMM);
          SuperRC = &AMDGPU::SReg_32RegClass;
          break;
      }
      unsigned DestReg = MRI->createVirtualRegister(SuperRC);
      // Find the insert point for the instructions being coalesced
      // This is the minimum OriginalOrder value within StartIdx to EndIdx
      auto InsertElt = std::min_element(Candidates.begin() + StartIdx,
                                        Candidates.begin() + StartIdx +
                                                             (nextSize/eltSize),
                                     [] (SimpleMI const& a, SimpleMI const& b)
                                     {
                                       return a.OriginalOrder < b.OriginalOrder;
                                     });
      DebugLoc DL = InsertElt->Inst->getDebugLoc();
      MachineBasicBlock *MBB = InsertElt->Inst->getParent();

      MachineInstrBuilder NewMergeInst =
        BuildMI(*MBB, InsertElt->Inst, DL, *MergeInstr, DestReg)
          .addReg(SBase.getReg())                      // addr
          .addImm(Candidates[StartIdx].getOffset(TII)) // offset
          .addImm(GLC.getImm())                        // glc
          .addImm(0);                                  // slc
      
      (void)NewMergeInst;
      
      // Copy the new dest register to the old ones
      unsigned SubOffset = 0;
      const MCInstrDesc &CopyDesc = TII->get(TargetOpcode::COPY);

      // We know that the old sub registers for this subrange will be contained
      // entirely as this is one the constraints on construction
      auto RI = TII->getRegisterInfo();
      auto I = StartIdx;
      while (SubOffset <  nextSize) {
        unsigned SubSize =
          RI.getRegSizeInBits(*Candidates[I].getDestRC(TII)) / 32;
        unsigned SubRegIdx = RI.getSubRegFromChannel(SubOffset, SubSize);
        const auto *Dest = Candidates[I].getDest(TII);
        BuildMI(*MBB, InsertElt->Inst, DL, CopyDesc)
             .add(*Dest)
             .addReg(DestReg, 0, SubRegIdx);
        SubOffset += Candidates[I].getSize(TII);
        ++I;
        
      }
      // Erase the instructions already dealt with
      for (auto J = StartIdx; J < I; J++) {
        Candidates[J].Inst->eraseFromParent();
      }
      LLVM_DEBUG(dbgs() << "Inserted coalesced instruction for DwordX"
                   << nextSize << " " << *NewMergeInst << "\n");
      Size -= nextSize;
      StartIdx = I;
    }
  }
}

bool SIBufMemMerge::processList(
  SmallVector<SmallVector<SimpleMI, 8>, 8> &CandidateList) {

  bool Modified = false;
  
  for ( auto EachList : CandidateList) {
    // Preserve the insert point
    int Order = 0;
    for (auto &Elt : EachList)
      Elt.OriginalOrder = Order++;

    LLVM_DEBUG(dbgs() << "Processing the following list\n");
    LLVM_DEBUG(for (auto Elt : EachList) dbgs() << Elt.OriginalOrder << " : "
                                           << *(Elt.Inst) << "\n";);
    
    std::sort(EachList.begin(), EachList.end(), [this](SimpleMI a, SimpleMI b) {
      return a.getOffset(TII) < b.getOffset(TII);
    });
    
    const auto *SBase = TII->getNamedOperand(*EachList.front().Inst,
                                             AMDGPU::OpName::sbase);
    const auto *GLC = TII->getNamedOperand(*EachList.front().Inst,
                                           AMDGPU::OpName::glc);
    
    bool StartingRun = true;
    // Sections is StartIdx, EndIdx, Size
    SmallVector<std::tuple<unsigned, unsigned, unsigned>, 8> Sections;
    
    unsigned OffsetStart = 0, PreviousOffset = 0;
    unsigned StartIdx = 0, EndIdx = 0;
    unsigned CurrentSize = 0, PreviousSize = 0;
    bool PushSection = true;
    for ( SimpleMI Elt : EachList ) {
      unsigned CurrentOffset = Elt.getDwordOffset(TII, ST);
      CurrentSize = Elt.getSize(TII);
      PushSection = true;
      if (StartingRun) {
        OffsetStart = PreviousOffset = CurrentOffset;
        PreviousSize = CurrentSize;
        StartingRun = false;
        continue;
      }
      if (CurrentOffset == PreviousOffset + PreviousSize) {
        PreviousOffset = CurrentOffset;
        PreviousSize = CurrentSize;
        ++EndIdx;
        continue;
      }
      
      // A gap found
      Sections.push_back(std::make_tuple(StartIdx, EndIdx,
                                         PreviousOffset + CurrentSize -
                                                          OffsetStart));
      OffsetStart = PreviousOffset = CurrentOffset;
      StartIdx = EndIdx = EndIdx + 1;
      PushSection = false;
    }
    // Check we haven't got a unpushed section pending
    if (PushSection)
      Sections.push_back(std::make_tuple(StartIdx, EndIdx,
                                         PreviousOffset + CurrentSize -
                                                          OffsetStart));
    
    
    // Now have a list of contiguous runs in Sections vector
    // Process each one
    for (auto SubSection : Sections) {
      unsigned StartIdx, EndIdx, Size;
      std::tie(StartIdx, EndIdx, Size) = SubSection;
      // Process all subsections with more than one operation
      if (EndIdx > StartIdx) {
        processSubSection(EachList, StartIdx, EndIdx, Size, *SBase, *GLC);
        Modified = true;
      }
    }
  }

  return Modified;
}

// We coalesce candidate instructions within a basic block
// Scan through the block and identify candidate instructions and add to the
// worklist if they are valid
bool SIBufMemMerge::optimizeBlock(MachineBasicBlock &MBB) {
  bool Modified = false;
  DenseMap<SimpleMI, unsigned> CandidateMap;
  SmallVector<SmallVector<SimpleMI, 8>, 8> CandidateList;
  unsigned int NextIdx = 0;
  unsigned int Phase = 0;
  
  for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end(); I != E; ++I) {
    MachineInstr &MI = *I;

    // Any instruction that stores prevents any futher coalescing
    if (MI.mayStore()) {
      LLVM_DEBUG(dbgs() << "Found an intermediate store instruction " << MI << "\n");
      Phase += 1; // Stop adding any instructions to existing lists - new ones
                  // required after blocking instruction
      continue;
    }
 
    // Not volatile
    if (MI.hasOrderedMemoryRef())
      continue;

    switch(MI.getOpcode()) {
    case AMDGPU::S_BUFFER_LOAD_DWORD_IMM:
    case AMDGPU::S_BUFFER_LOAD_DWORDX2_IMM:
    case AMDGPU::S_BUFFER_LOAD_DWORDX4_IMM:
    case AMDGPU::S_BUFFER_LOAD_DWORDX8_IMM:
      SimpleMI SMI(&MI, Phase);
      auto CandidateIdx = CandidateMap.find(SMI);
      if (CandidateIdx == CandidateMap.end()) {
        // Generate a new list and add to the map
        CandidateList.push_back(SmallVector<SimpleMI,8>());
        CandidateList.back().push_back(SMI);
        CandidateMap[SMI] = NextIdx++;
      } else {
        CandidateList[CandidateIdx->getSecond()].push_back(SMI);
      }
      break;
    }
  }
  
  Modified |= processList(CandidateList);
  return Modified;
}

bool SIBufMemMerge::runOnMachineFunction(MachineFunction &MF) {

  if (skipFunction(MF.getFunction()))
    return false;

  MRI = &MF.getRegInfo();
  ST = &MF.getSubtarget<GCNSubtarget>();
  TII = ST->getInstrInfo();
  TRI = &TII->getRegisterInfo();
  RP = &getAnalysis<AMDGPURegPressAnalysis>();

  const Function &F = MF.getFunction();
  unsigned MergeThreshold = AMDGPU::getIntegerAttribute(F, "amdgpu-bufmem-merge-threshold", 85);

  LLVM_DEBUG(dbgs() << "Running SIBufMemMerge\n");

  bool Modified = false;

  GCNRegPressure MaxPressure = RP->getMaxPressure();
  HighPressure = false;
  if (MaxPressure.getSGPRTuplesWeight() > MergeThreshold) {
    // Don't do so much merging - can cause more fragmentation and increase spilling
    HighPressure = true;
  }

  LLVM_DEBUG(
    dbgs() << "Max pressure : ";
    MaxPressure.print(dbgs(), ST);
  );

  for (MachineBasicBlock &MBB : MF)
    Modified |= optimizeBlock(MBB);

  return Modified;

}
