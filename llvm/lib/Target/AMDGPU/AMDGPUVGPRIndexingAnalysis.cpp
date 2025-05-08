#if LLPC_BUILD_NPI
//===- AMDGPUVGPRIndexingAnalysis.cpp ---- VGPR indexing analysis ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// Determine properties of VGPR indexing instructions.
/// Currently, only determine VGPR usage intervals.
///
/// The results of this analysis are used to determine waitcnts and to
/// add implicit defs/uses for promoted private objects
///
//===----------------------------------------------------------------------===//

#include "AMDGPUVGPRIndexingAnalysis.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

AnalysisKey AMDGPUVGPRIndexingAnalysis::Key;

#define DEBUG_TYPE "amdgpu-vgpr-indexing-analysis"

char AMDGPUIndexingInfoWrapper::ID = 0;

char &llvm::AMDGPUIndexingInfoWrapperID = AMDGPUIndexingInfoWrapper::ID;

INITIALIZE_PASS(AMDGPUIndexingInfoWrapper, DEBUG_TYPE,
                "AMDGPU VGPR Indexing Analysis", false, true)

static constexpr unsigned NumIdxRegs = 4;

std::optional<std::reference_wrapper<const AMDGPULaneSharedIdxInfo>>
AMDGPUIndexingInfo::getLaneSharedIdxInfo(const MachineInstr *MI) const {
  if (const auto &It = LaneSharedIdxInfos.find(MI);
      It != LaneSharedIdxInfos.end())
    return std::cref(It->second);
  return std::nullopt;
}

std::optional<std::reference_wrapper<const AMDGPUPrivateObjectIdxInfo>>
AMDGPUIndexingInfo::getPrivateObjectIdxInfo(const MachineInstr *MI) const {
  if (const auto &It = PrivateObjectIdxInfos.find(MI);
      It != PrivateObjectIdxInfos.end())
    return std::cref(It->second);
  return std::nullopt;
}

namespace {
class AnalysisImpl {
public:
  AnalysisImpl(
      const SIRegisterInfo &TRI, const MachineRegisterInfo &MRI,
      std::unordered_map<const MachineInstr *, AMDGPULaneSharedIdxInfo> &MapLS,
      std::unordered_map<const MachineInstr *, AMDGPUPrivateObjectIdxInfo>
          &MapPO,
      unsigned LaneSharedSize)
      : TRI(TRI), MRI(MRI), LaneSharedIdxInfos(MapLS),
        PrivateObjectIdxInfos(MapPO), LaneSharedSize(LaneSharedSize) {}
  void compute(const MachineFunction &MF);

private:
  const SIRegisterInfo &TRI;
  const MachineRegisterInfo &MRI;
  std::unordered_map<const MachineInstr *, AMDGPULaneSharedIdxInfo>
      &LaneSharedIdxInfos;
  std::unordered_map<const MachineInstr *, AMDGPUPrivateObjectIdxInfo>
      &PrivateObjectIdxInfos;
  unsigned LaneSharedSize;
  // Track the possible immediate value stored in gpr-idx registers
  std::optional<int64_t> GprIdxImmedVals[NumIdxRegs];

  void analyzeIdxInst(const MachineInstr &MI);
  std::optional<std::pair<unsigned, unsigned>>
  computeUsedRegs(const MachineInstr &MI, unsigned Idx, bool IsLaneShared);
  void addPrivateObjectIdxInfo(
      const MachineInstr *MI,
      std::optional<std::pair<unsigned, unsigned>> UsedRegs);
  void
  addLaneSharedIdxInfo(const MachineInstr *MI,
                       std::optional<std::pair<unsigned, unsigned>> UsedRegs);
};
} // namespace

AMDGPUVGPRIndexingAnalysis::Result
AMDGPUVGPRIndexingAnalysis::run(MachineFunction &MF,
                                MachineFunctionAnalysisManager &MFAM) {
  // TODO-GFX13: Port other VGPR indexing passes to NPM
  return AMDGPUIndexingInfo(0);
}

FunctionPass *llvm::createAMDGPUIndexingInfoWrapperPass() {
  return new AMDGPUIndexingInfoWrapper();
}

bool AMDGPUIndexingInfoWrapper::runOnMachineFunction(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();

  if (!ST.hasVGPRIndexingRegisters())
    return false;

  const SIMachineFunctionInfo &MFI = *MF.getInfo<SIMachineFunctionInfo>();
  unsigned LaneSharedSize = MFI.getLaneSharedVGPRSize() / 4u;
  IndexingInfo = AMDGPUIndexingInfo(LaneSharedSize);
  AnalysisImpl Impl(*ST.getRegisterInfo(), MF.getRegInfo(),
                    IndexingInfo.LaneSharedIdxInfos,
                    IndexingInfo.PrivateObjectIdxInfos, LaneSharedSize);
  Impl.compute(MF);

  return false;
}

std::pair<unsigned, unsigned>
llvm::getAMDGPUPrivateObjectNodeInfo(const MDNode *Obj) {
  assert(Obj && "Expected promoted private object");
  unsigned Offset =
      cast<ConstantInt>(
          cast<ConstantAsMetadata>(Obj->getOperand(0))->getValue())
          ->getZExtValue();
  unsigned Size = cast<ConstantInt>(
                      cast<ConstantAsMetadata>(Obj->getOperand(1))->getValue())
                      ->getZExtValue();
  return std::make_pair(Offset, Size);
}

// Return metadata describing the object the passed instruction refers to,
// if any, or nullptr otherwise.
static const MDNode *getPromotedPrivateObject(const MachineInstr &MI) {
  if (MI.getOpcode() != AMDGPU::V_LOAD_IDX &&
      MI.getOpcode() != AMDGPU::V_STORE_IDX)
    return nullptr;

  const MachineMemOperand *MMO = *MI.memoperands_begin();
  assert(MMO);
  const Value *Ptr = MMO->getValue();
  if (!Ptr)
    return nullptr;

  Ptr = getUnderlyingObjectAggressive(Ptr);
  const auto *Alloca = dyn_cast<AllocaInst>(Ptr);
  if (!Alloca)
    return nullptr;

  const MDNode *Obj = Alloca->getMetadata("amdgpu.allocated.vgprs");
  if (!Obj)
    return nullptr;

  return Obj;
}

void AnalysisImpl::addPrivateObjectIdxInfo(
    const MachineInstr *MI,
    std::optional<std::pair<unsigned, unsigned>> UsedRegs) {
  const MDNode *Obj = getPromotedPrivateObject(*MI);
  assert(Obj && "Expected promoted private object");
  auto [Offset, Size] = getAMDGPUPrivateObjectNodeInfo(Obj);
  PrivateObjectIdxInfos.insert({MI, {UsedRegs, Obj, Size, Offset}});
}

void AnalysisImpl::addLaneSharedIdxInfo(
    const MachineInstr *MI,
    std::optional<std::pair<unsigned, unsigned>> UsedRegs) {
  LaneSharedIdxInfos.insert({MI, {UsedRegs}});
}

// The interval (I) of used regs begins at the first known VGPR used by MI,
// including LaneSharedSize and offset relative to other private objects (if
// applicable). The interval ends at I.first + [size of memory access].
std::optional<std::pair<unsigned, unsigned>>
AnalysisImpl::computeUsedRegs(const MachineInstr &MI, unsigned Idx,
                              bool IsLaneShared) {
  if (!GprIdxImmedVals[Idx].has_value())
    return std::nullopt;
  std::pair<unsigned, unsigned> Result;
  Result.first = IsLaneShared ? 0 : LaneSharedSize;
  auto OffsetSrcIdx =
      AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::offset);
  auto Offset = MI.getOperand(OffsetSrcIdx).getImm();
  Result.first += GprIdxImmedVals[Idx].value() + Offset;
  assert(MI.hasOneMemOperand() && "V_LOAD/STORE_IDX must have one MMO");
  MachineMemOperand *MMO = *MI.memoperands_begin();
  auto Size = MMO->getSizeInBits().getValue();
  Result.second = Result.first + divideCeil(Size, 32);
  return Result;
}

void AnalysisImpl::compute(const MachineFunction &MF) {
  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB.instrs())
      analyzeIdxInst(MI);
  }
}

void AnalysisImpl::analyzeIdxInst(const MachineInstr &MI) {
  // Track the gpr-idx value in case that is an immed.
  // TODO-GFX13: need to handle s_add_gpr_idx_u32 when it is added.
  if (MI.getOpcode() == AMDGPU::S_SET_GPR_IDX_U32) {
    auto SrcOpnd = MI.getOperand(1);
    unsigned Idx = MI.getOperand(0).getReg() - AMDGPU::IDX0;
    assert(Idx < NumIdxRegs);
    if (SrcOpnd.isImm())
      GprIdxImmedVals[Idx] = SrcOpnd.getImm();
    else
      GprIdxImmedVals[Idx] = {};
  } else if (MI.getOpcode() == AMDGPU::V_STORE_IDX ||
             MI.getOpcode() == AMDGPU::V_LOAD_IDX) {
    auto IdxSrcIdx =
        AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::idx);
    unsigned Idx = MI.getOperand(IdxSrcIdx).getReg() - AMDGPU::IDX0;
    assert(MI.hasOneMemOperand());
    const MachineMemOperand *MMO = *MI.memoperands_begin();
    bool IsLaneShared = MMO->getAddrSpace() == AMDGPUAS::LANE_SHARED;
    auto UsedRegs = computeUsedRegs(MI, Idx, IsLaneShared);
#ifndef NDEBUG
    // Do not allow vector registers as implicit operands of v_load/store_idx
    // on laneshared because it is not clear whether those registers are
    // laneshared or wave-private.
    if (IsLaneShared) {
      for (auto Opnd : MI.implicit_operands())
        assert(!Opnd.isReg() || !TRI.isVectorRegister(MRI, Opnd.getReg()));
    }
#endif

    if (IsLaneShared)
      // Create and add laneshared info to the LS map
      addLaneSharedIdxInfo(&MI, UsedRegs);
    else
      addPrivateObjectIdxInfo(&MI, UsedRegs);
  }
}
#endif /* LLPC_BUILD_NPI */
