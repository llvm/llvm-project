//=====-- GCNSubtarget.h - Define GCN Subtarget for AMDGPU ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==-----------------------------------------------------------------------===//
//
/// \file
/// AMD GCN specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_GCNSUBTARGET_H
#define LLVM_LIB_TARGET_AMDGPU_GCNSUBTARGET_H

#include "AMDGPUCallLowering.h"
#include "AMDGPURegisterBankInfo.h"
#include "AMDGPUSubtarget.h"
#include "SIFrameLowering.h"
#include "SIISelLowering.h"
#include "SIInstrInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/Support/ErrorHandling.h"

#define GET_SUBTARGETINFO_HEADER
#include "AMDGPUGenSubtargetInfo.inc"

namespace llvm {

class GCNTargetMachine;

class GCNSubtarget final : public AMDGPUGenSubtargetInfo,
                           public AMDGPUSubtarget {
public:
  using AMDGPUSubtarget::getMaxWavesPerEU;

  // Following 2 enums are documented at:
  //   - https://llvm.org/docs/AMDGPUUsage.html#trap-handler-abi
  enum class TrapHandlerAbi {
    NONE   = 0x00,
    AMDHSA = 0x01,
  };

  enum class TrapID {
    LLVMAMDHSATrap      = 0x02,
    LLVMAMDHSADebugTrap = 0x03,
  };

private:
  /// SelectionDAGISel related APIs.
  std::unique_ptr<const SelectionDAGTargetInfo> TSInfo;

  /// GlobalISel related APIs.
  std::unique_ptr<AMDGPUCallLowering> CallLoweringInfo;
  std::unique_ptr<InlineAsmLowering> InlineAsmLoweringInfo;
  std::unique_ptr<InstructionSelector> InstSelector;
  std::unique_ptr<LegalizerInfo> Legalizer;
  std::unique_ptr<AMDGPURegisterBankInfo> RegBankInfo;

protected:
  // Basic subtarget description.
  AMDGPU::IsaInfo::AMDGPUTargetID TargetID;
  unsigned Gen = INVALID;
  InstrItineraryData InstrItins;
  int LDSBankCount = 0;
  unsigned MaxPrivateElementSize = 0;

  // Dynamically set bits that enable features.
  bool DynamicVGPR = false;
  bool DynamicVGPRBlockSize32 = false;
  bool ScalarizeGlobal = false;

  /// The maximum number of instructions that may be placed within an S_CLAUSE,
  /// which is one greater than the maximum argument to S_CLAUSE. A value of 0
  /// indicates a lack of S_CLAUSE support.
  unsigned MaxHardClauseLength = 0;

#define GET_SUBTARGETINFO_MACRO(ATTRIBUTE, DEFAULT, GETTER)                    \
  bool ATTRIBUTE = DEFAULT;
#include "AMDGPUGenSubtargetInfo.inc"

private:
  SIInstrInfo InstrInfo;
  SITargetLowering TLInfo;
  SIFrameLowering FrameLowering;

public:
  GCNSubtarget(const Triple &TT, StringRef GPU, StringRef FS,
               const GCNTargetMachine &TM);
  ~GCNSubtarget() override;

  GCNSubtarget &initializeSubtargetDependencies(const Triple &TT, StringRef GPU,
                                                StringRef FS);

  /// Diagnose inconsistent subtarget features before attempting to codegen
  /// function \p F.
  void checkSubtargetFeatures(const Function &F) const;

  const SIInstrInfo *getInstrInfo() const override { return &InstrInfo; }

  const SIFrameLowering *getFrameLowering() const override {
    return &FrameLowering;
  }

  const SITargetLowering *getTargetLowering() const override { return &TLInfo; }

  const SIRegisterInfo *getRegisterInfo() const override {
    return &InstrInfo.getRegisterInfo();
  }

  const SelectionDAGTargetInfo *getSelectionDAGInfo() const override;

  const CallLowering *getCallLowering() const override {
    return CallLoweringInfo.get();
  }

  const InlineAsmLowering *getInlineAsmLowering() const override {
    return InlineAsmLoweringInfo.get();
  }

  InstructionSelector *getInstructionSelector() const override {
    return InstSelector.get();
  }

  const LegalizerInfo *getLegalizerInfo() const override {
    return Legalizer.get();
  }

  const AMDGPURegisterBankInfo *getRegBankInfo() const override {
    return RegBankInfo.get();
  }

  const AMDGPU::IsaInfo::AMDGPUTargetID &getTargetID() const {
    return TargetID;
  }

  const InstrItineraryData *getInstrItineraryData() const override {
    return &InstrItins;
  }

  void ParseSubtargetFeatures(StringRef CPU, StringRef TuneCPU, StringRef FS);

  Generation getGeneration() const { return (Generation)Gen; }

  bool isGFX11Plus() const { return getGeneration() >= GFX11; }

#define GET_SUBTARGETINFO_MACRO(ATTRIBUTE, DEFAULT, GETTER)                    \
  bool GETTER() const override { return ATTRIBUTE; }
#include "AMDGPUGenSubtargetInfo.inc"

  unsigned getMaxWaveScratchSize() const {
    // See COMPUTE_TMPRING_SIZE.WAVESIZE.
    if (getGeneration() >= GFX12) {
      // 18-bit field in units of 64-dword.
      return (64 * 4) * ((1 << 18) - 1);
    }
    if (getGeneration() == GFX11) {
      // 15-bit field in units of 64-dword.
      return (64 * 4) * ((1 << 15) - 1);
    }
    // 13-bit field in units of 256-dword.
    return (256 * 4) * ((1 << 13) - 1);
  }

  /// Return the number of high bits known to be zero for a frame index.
  unsigned getKnownHighZeroBitsForFrameIndex() const {
    return llvm::countl_zero(getMaxWaveScratchSize()) + getWavefrontSizeLog2();
  }

  int getLDSBankCount() const { return LDSBankCount; }

  unsigned getMaxPrivateElementSize(bool ForBufferRSrc = false) const {
    return (ForBufferRSrc || !hasFlatScratchEnabled()) ? MaxPrivateElementSize
                                                       : 16;
  }

  unsigned getConstantBusLimit(unsigned Opcode) const;

  /// Returns if the result of this instruction with a 16-bit result returned in
  /// a 32-bit register implicitly zeroes the high 16-bits, rather than preserve
  /// the original value.
  bool zeroesHigh16BitsOfDest(unsigned Opcode) const;

  bool supportsWGP() const {
    if (HasGFX1250Insts)
      return false;
    return getGeneration() >= GFX10;
  }

  bool hasHWFP64() const { return HasFP64; }

  bool hasAddr64() const {
    return (getGeneration() < AMDGPUSubtarget::VOLCANIC_ISLANDS);
  }

  bool hasFlat() const {
    return (getGeneration() > AMDGPUSubtarget::SOUTHERN_ISLANDS);
  }

  // Return true if the target only has the reverse operand versions of VALU
  // shift instructions (e.g. v_lshrrev_b32, and no v_lshr_b32).
  bool hasOnlyRevVALUShifts() const {
    return getGeneration() >= VOLCANIC_ISLANDS;
  }

  bool hasFractBug() const { return getGeneration() == SOUTHERN_ISLANDS; }

  bool hasMed3_16() const { return getGeneration() >= AMDGPUSubtarget::GFX9; }

  bool hasMin3Max3_16() const {
    return getGeneration() >= AMDGPUSubtarget::GFX9;
  }

  bool hasSwap() const { return HasGFX9Insts; }

  bool hasScalarPackInsts() const { return HasGFX9Insts; }

  bool hasScalarMulHiInsts() const { return HasGFX9Insts; }

  bool hasScalarSubwordLoads() const { return getGeneration() >= GFX12; }

  TrapHandlerAbi getTrapHandlerAbi() const {
    return isAmdHsaOS() ? TrapHandlerAbi::AMDHSA : TrapHandlerAbi::NONE;
  }

  bool supportsGetDoorbellID() const {
    // The S_GETREG DOORBELL_ID is supported by all GFX9 onward targets.
    return getGeneration() >= GFX9;
  }

  /// True if the offset field of DS instructions works as expected. On SI, the
  /// offset uses a 16-bit adder and does not always wrap properly.
  bool hasUsableDSOffset() const { return getGeneration() >= SEA_ISLANDS; }

  bool unsafeDSOffsetFoldingEnabled() const {
    return EnableUnsafeDSOffsetFolding;
  }

  /// Condition output from div_scale is usable.
  bool hasUsableDivScaleConditionOutput() const {
    return getGeneration() != SOUTHERN_ISLANDS;
  }

  /// Extra wait hazard is needed in some cases before
  /// s_cbranch_vccnz/s_cbranch_vccz.
  bool hasReadVCCZBug() const { return getGeneration() <= SEA_ISLANDS; }

  /// Writes to VCC_LO/VCC_HI update the VCCZ flag.
  bool partialVCCWritesUpdateVCCZ() const { return getGeneration() >= GFX10; }

  /// A read of an SGPR by SMRD instruction requires 4 wait states when the SGPR
  /// was written by a VALU instruction.
  bool hasSMRDReadVALUDefHazard() const {
    return getGeneration() == SOUTHERN_ISLANDS;
  }

  /// A read of an SGPR by a VMEM instruction requires 5 wait states when the
  /// SGPR was written by a VALU Instruction.
  bool hasVMEMReadSGPRVALUDefHazard() const {
    return getGeneration() >= VOLCANIC_ISLANDS;
  }

  bool hasRFEHazards() const { return getGeneration() >= VOLCANIC_ISLANDS; }

  /// Number of hazard wait states for s_setreg_b32/s_setreg_imm32_b32.
  unsigned getSetRegWaitStates() const {
    return getGeneration() <= SEA_ISLANDS ? 1 : 2;
  }

  /// Return the amount of LDS that can be used that will not restrict the
  /// occupancy lower than WaveCount.
  unsigned getMaxLocalMemSizeWithWaveCount(unsigned WaveCount,
                                           const Function &) const;

  bool supportsMinMaxDenormModes() const {
    return getGeneration() >= AMDGPUSubtarget::GFX9;
  }

  /// \returns If target supports S_DENORM_MODE.
  bool hasDenormModeInst() const {
    return getGeneration() >= AMDGPUSubtarget::GFX10;
  }

  /// \returns If target supports ds_read/write_b128 and user enables generation
  /// of ds_read/write_b128.
  bool useDS128() const { return HasCIInsts && EnableDS128; }

  /// \return If target supports ds_read/write_b96/128.
  bool hasDS96AndDS128() const { return HasCIInsts; }

  /// Have v_trunc_f64, v_ceil_f64, v_rndne_f64
  bool haveRoundOpsF64() const { return HasCIInsts; }

  /// \returns If MUBUF instructions always perform range checking, even for
  /// buffer resources used for private memory access.
  bool privateMemoryResourceIsRangeChecked() const {
    return getGeneration() < AMDGPUSubtarget::GFX9;
  }

  /// \returns If target requires PRT Struct NULL support (zero result registers
  /// for sparse texture support).
  bool usePRTStrictNull() const { return EnablePRTStrictNull; }

  bool hasUnalignedBufferAccessEnabled() const {
    return HasUnalignedBufferAccess && HasUnalignedAccessMode;
  }

  bool hasUnalignedDSAccessEnabled() const {
    return HasUnalignedDSAccess && HasUnalignedAccessMode;
  }

  bool hasUnalignedScratchAccessEnabled() const {
    return HasUnalignedScratchAccess && HasUnalignedAccessMode;
  }

  bool isXNACKEnabled() const { return TargetID.isXnackOnOrAny(); }

  bool isTgSplitEnabled() const { return EnableTgSplit; }

  bool isCuModeEnabled() const { return EnableCuMode; }

  bool isPreciseMemoryEnabled() const { return EnablePreciseMemory; }

  bool hasFlatScrRegister() const { return hasFlatAddressSpace(); }

  // Check if target supports ST addressing mode with FLAT scratch instructions.
  // The ST addressing mode means no registers are used, either VGPR or SGPR,
  // but only immediate offset is swizzled and added to the FLAT scratch base.
  bool hasFlatScratchSTMode() const {
    return hasFlatScratchInsts() && (hasGFX10_3Insts() || hasGFX940Insts());
  }

  bool hasFlatScratchSVSMode() const { return HasGFX940Insts || HasGFX11Insts; }

  bool hasFlatScratchEnabled() const {
    return hasArchitectedFlatScratch() ||
           (EnableFlatScratch && hasFlatScratchInsts());
  }

  bool hasGlobalAddTidInsts() const { return HasGFX10_BEncoding; }

  bool hasAtomicCSub() const { return HasGFX10_BEncoding; }

  bool hasMTBUFInsts() const { return !hasGFX1250Insts(); }

  bool hasFormattedMUBUFInsts() const { return !hasGFX1250Insts(); }

  bool hasExportInsts() const {
    return !hasGFX940Insts() && !hasGFX1250Insts();
  }

  bool hasVINTERPEncoding() const {
    return HasGFX11Insts && !hasGFX1250Insts();
  }

  // DS_ADD_F64/DS_ADD_RTN_F64
  bool hasLdsAtomicAddF64() const {
    return hasGFX90AInsts() || hasGFX1250Insts();
  }

  bool hasMultiDwordFlatScratchAddressing() const {
    return getGeneration() >= GFX9;
  }

  bool hasFlatLgkmVMemCountInOrder() const { return getGeneration() > GFX9; }

  bool hasD16LoadStore() const { return getGeneration() >= GFX9; }

  bool d16PreservesUnusedBits() const {
    return hasD16LoadStore() && !TargetID.isSramEccOnOrAny();
  }

  bool hasD16Images() const { return getGeneration() >= VOLCANIC_ISLANDS; }

  /// Return if most LDS instructions have an m0 use that require m0 to be
  /// initialized.
  bool ldsRequiresM0Init() const { return getGeneration() < GFX9; }

  // True if the hardware rewinds and replays GWS operations if a wave is
  // preempted.
  //
  // If this is false, a GWS operation requires testing if a nack set the
  // MEM_VIOL bit, and repeating if so.
  bool hasGWSAutoReplay() const { return getGeneration() >= GFX9; }

  /// \returns if target has ds_gws_sema_release_all instruction.
  bool hasGWSSemaReleaseAll() const { return HasCIInsts; }

  bool hasScalarAddSub64() const { return getGeneration() >= GFX12; }

  bool hasScalarSMulU64() const { return getGeneration() >= GFX12; }

  // Covers VS/PS/CS graphics shaders
  bool isMesaGfxShader(const Function &F) const {
    return isMesa3DOS() && AMDGPU::isShader(F.getCallingConv());
  }

  bool hasMad64_32() const { return getGeneration() >= SEA_ISLANDS; }

  bool hasAtomicFaddInsts() const {
    return HasAtomicFaddRtnInsts || HasAtomicFaddNoRtnInsts;
  }

  bool vmemWriteNeedsExpWaitcnt() const {
    return getGeneration() < SEA_ISLANDS;
  }

  bool hasInstPrefetch() const {
    return getGeneration() == GFX10 || getGeneration() == GFX11;
  }

  bool hasPrefetch() const { return HasGFX12Insts; }

  // Has s_cmpk_* instructions.
  bool hasSCmpK() const { return getGeneration() < GFX12; }

  // Scratch is allocated in 256 dword per wave blocks for the entire
  // wavefront. When viewed from the perspective of an arbitrary workitem, this
  // is 4-byte aligned.
  //
  // Only 4-byte alignment is really needed to access anything. Transformations
  // on the pointer value itself may rely on the alignment / known low bits of
  // the pointer. Set this to something above the minimum to avoid needing
  // dynamic realignment in common cases.
  Align getStackAlignment() const { return Align(16); }

  bool enableMachineScheduler() const override { return true; }

  bool useAA() const override;

  bool enableSubRegLiveness() const override { return true; }

  void setScalarizeGlobalBehavior(bool b) { ScalarizeGlobal = b; }
  bool getScalarizeGlobalBehavior() const { return ScalarizeGlobal; }

  // static wrappers
  static bool hasHalfRate64Ops(const TargetSubtargetInfo &STI);

  // XXX - Why is this here if it isn't in the default pass set?
  bool enableEarlyIfConversion() const override { return true; }

  void overrideSchedPolicy(MachineSchedPolicy &Policy,
                           const SchedRegion &Region) const override;

  void overridePostRASchedPolicy(MachineSchedPolicy &Policy,
                                 const SchedRegion &Region) const override;

  void mirFileLoaded(MachineFunction &MF) const override;

  unsigned getMaxNumUserSGPRs() const {
    return AMDGPU::getMaxNumUserSGPRs(*this);
  }

  bool useVGPRIndexMode() const;

  bool hasScalarCompareEq64() const {
    return getGeneration() >= VOLCANIC_ISLANDS;
  }

  bool hasLDSFPAtomicAddF32() const { return HasGFX8Insts; }
  bool hasLDSFPAtomicAddF64() const {
    return HasGFX90AInsts || HasGFX1250Insts;
  }

  /// \returns true if the subtarget has the v_permlanex16_b32 instruction.
  bool hasPermLaneX16() const { return getGeneration() >= GFX10; }

  /// \returns true if the subtarget has the v_permlane64_b32 instruction.
  bool hasPermLane64() const { return getGeneration() >= GFX11; }

  bool hasDPPBroadcasts() const { return HasDPP && getGeneration() < GFX10; }

  bool hasDPPWavefrontShifts() const {
    return HasDPP && getGeneration() < GFX10;
  }

  // Has V_PK_MOV_B32 opcode
  bool hasPkMovB32() const { return HasGFX90AInsts; }

  bool hasFmaakFmamkF32Insts() const {
    return getGeneration() >= GFX10 || hasGFX940Insts();
  }

  bool hasFmaakFmamkF64Insts() const { return hasGFX1250Insts(); }

  bool hasNonNSAEncoding() const { return getGeneration() < GFX12; }

  unsigned getNSAMaxSize(bool HasSampler = false) const {
    return AMDGPU::getNSAMaxSize(*this, HasSampler);
  }

  bool hasMadF16() const;

  bool hasMovB64() const { return HasGFX940Insts || HasGFX1250Insts; }

  // Scalar and global loads support scale_offset bit.
  bool hasScaleOffset() const { return HasGFX1250Insts; }

  // FLAT GLOBAL VOffset is signed
  bool hasSignedGVSOffset() const { return HasGFX1250Insts; }

  bool loadStoreOptEnabled() const { return EnableLoadStoreOpt; }

  bool hasUserSGPRInit16BugInWave32() const {
    return HasUserSGPRInit16Bug && isWave32();
  }

  bool has12DWordStoreHazard() const {
    return getGeneration() != AMDGPUSubtarget::SOUTHERN_ISLANDS;
  }

  // \returns true if the subtarget supports DWORDX3 load/store instructions.
  bool hasDwordx3LoadStores() const { return HasCIInsts; }

  bool hasReadM0MovRelInterpHazard() const {
    return getGeneration() == AMDGPUSubtarget::GFX9;
  }

  bool hasReadM0SendMsgHazard() const {
    return getGeneration() >= AMDGPUSubtarget::VOLCANIC_ISLANDS &&
           getGeneration() <= AMDGPUSubtarget::GFX9;
  }

  bool hasReadM0LdsDmaHazard() const {
    return getGeneration() == AMDGPUSubtarget::GFX9;
  }

  bool hasReadM0LdsDirectHazard() const {
    return getGeneration() == AMDGPUSubtarget::GFX9;
  }

  bool hasLDSMisalignedBugInWGPMode() const {
    return HasLDSMisalignedBug && !EnableCuMode;
  }

  // Shift amount of a 64 bit shift cannot be a highest allocated register
  // if also at the end of the allocation block.
  bool hasShift64HighRegBug() const {
    return HasGFX90AInsts && !HasGFX940Insts;
  }

  // Has one cycle hazard on transcendental instruction feeding a
  // non transcendental VALU.
  bool hasTransForwardingHazard() const { return HasGFX940Insts; }

  // Has one cycle hazard on a VALU instruction partially writing dst with
  // a shift of result bits feeding another VALU instruction.
  bool hasDstSelForwardingHazard() const { return HasGFX940Insts; }

  // Cannot use op_sel with v_dot instructions.
  bool hasDOTOpSelHazard() const { return HasGFX940Insts || HasGFX11Insts; }

  // Does not have HW interlocs for VALU writing and then reading SGPRs.
  bool hasVDecCoExecHazard() const { return HasGFX940Insts; }

  bool hasHardClauses() const { return MaxHardClauseLength > 0; }

  bool hasFPAtomicToDenormModeHazard() const {
    return getGeneration() == GFX10;
  }

  bool hasVOP3DPP() const { return getGeneration() >= GFX11; }

  bool hasLdsDirect() const { return getGeneration() >= GFX11; }

  bool hasLdsWaitVMSRC() const { return getGeneration() >= GFX12; }

  bool hasVALUPartialForwardingHazard() const {
    return getGeneration() == GFX11;
  }

  bool hasCvtScaleForwardingHazard() const { return HasGFX950Insts; }

  bool requiresCodeObjectV6() const { return RequiresCOV6; }

  bool useVGPRBlockOpsForCSR() const { return UseBlockVGPROpsForCSR; }

  bool hasVALUMaskWriteHazard() const { return getGeneration() == GFX11; }

  bool hasVALUReadSGPRHazard() const {
    return HasGFX12Insts && !HasGFX1250Insts;
  }

  bool setRegModeNeedsVNOPs() const {
    return HasGFX1250Insts && getGeneration() == GFX12;
  }

  /// Return if operations acting on VGPR tuples require even alignment.
  bool needsAlignedVGPRs() const { return RequiresAlignVGPR; }

  /// Return true if the target has the S_PACK_HL_B32_B16 instruction.
  bool hasSPackHL() const { return HasGFX11Insts; }

  /// Return true if the target's EXP instruction has the COMPR flag, which
  /// affects the meaning of the EN (enable) bits.
  bool hasCompressedExport() const { return !HasGFX11Insts; }

  /// Return true if the target's EXP instruction supports the NULL export
  /// target.
  bool hasNullExportTarget() const { return !HasGFX11Insts; }

  bool hasFlatScratchSVSSwizzleBug() const { return getGeneration() == GFX11; }

  /// Return true if the target has the S_DELAY_ALU instruction.
  bool hasDelayAlu() const { return HasGFX11Insts; }

  /// Returns true if the target supports
  /// global_load_lds_dwordx3/global_load_lds_dwordx4 or
  /// buffer_load_dwordx3/buffer_load_dwordx4 with the lds bit.
  bool hasLDSLoadB96_B128() const { return hasGFX950Insts(); }

  /// \returns true if the target uses LOADcnt/SAMPLEcnt/BVHcnt, DScnt/KMcnt
  /// and STOREcnt rather than VMcnt, LGKMcnt and VScnt respectively.
  bool hasExtendedWaitCounts() const { return getGeneration() >= GFX12; }

  /// \returns true if inline constants are not supported for F16 pseudo
  /// scalar transcendentals.
  bool hasNoF16PseudoScalarTransInlineConstants() const {
    return getGeneration() == GFX12;
  }

  /// \returns true if the target has packed f32 instructions that only read 32
  /// bits from a scalar operand (SGPR or literal) and replicates the bits to
  /// both channels.
  bool hasPKF32InstsReplicatingLower32BitsOfScalarInput() const {
    return getGeneration() == GFX12 && HasGFX1250Insts;
  }

  bool hasAddPC64Inst() const { return HasGFX1250Insts; }

  /// \returns true if the target supports expert scheduling mode 2 which relies
  /// on the compiler to insert waits to avoid hazards between VMEM and VALU
  /// instructions in some instances.
  bool hasExpertSchedulingMode() const { return getGeneration() >= GFX12; }

  /// \returns The maximum number of instructions that can be enclosed in an
  /// S_CLAUSE on the given subtarget, or 0 for targets that do not support that
  /// instruction.
  unsigned maxHardClauseLength() const { return MaxHardClauseLength; }

  /// Return the maximum number of waves per SIMD for kernels using \p SGPRs
  /// SGPRs
  unsigned getOccupancyWithNumSGPRs(unsigned SGPRs) const;

  /// Return the maximum number of waves per SIMD for kernels using \p VGPRs
  /// VGPRs
  unsigned getOccupancyWithNumVGPRs(unsigned VGPRs,
                                    unsigned DynamicVGPRBlockSize) const;

  /// Subtarget's minimum/maximum occupancy, in number of waves per EU, that can
  /// be achieved when the only function running on a CU is \p F, each workgroup
  /// uses \p LDSSize bytes of LDS, and each wave uses \p NumSGPRs SGPRs and \p
  /// NumVGPRs VGPRs. The flat workgroup sizes associated to the function are a
  /// range, so this returns a range as well.
  ///
  /// Note that occupancy can be affected by the scratch allocation as well, but
  /// we do not have enough information to compute it.
  std::pair<unsigned, unsigned> computeOccupancy(const Function &F,
                                                 unsigned LDSSize = 0,
                                                 unsigned NumSGPRs = 0,
                                                 unsigned NumVGPRs = 0) const;

  /// \returns true if the flat_scratch register should be initialized with the
  /// pointer to the wave's scratch memory rather than a size and offset.
  bool flatScratchIsPointer() const {
    return getGeneration() >= AMDGPUSubtarget::GFX9;
  }

  /// \returns true if the machine has merged shaders in which s0-s7 are
  /// reserved by the hardware and user SGPRs start at s8
  bool hasMergedShaders() const { return getGeneration() >= GFX9; }

  // \returns true if the target supports the pre-NGG legacy geometry path.
  bool hasLegacyGeometry() const { return getGeneration() < GFX11; }

  // \returns true if the target has split barriers feature
  bool hasSplitBarriers() const { return getGeneration() >= GFX12; }

  // \returns true if the target has DX10_CLAMP kernel descriptor mode bit
  bool hasDX10ClampMode() const { return getGeneration() < GFX12; }

  // \returns true if the target has IEEE kernel descriptor mode bit
  bool hasIEEEMode() const { return getGeneration() < GFX12; }

  // \returns true if the target has WG_RR_MODE kernel descriptor mode bit
  bool hasRrWGMode() const { return getGeneration() >= GFX12; }

  /// \returns true if VADDR and SADDR fields in VSCRATCH can use negative
  /// values.
  bool hasSignedScratchOffsets() const { return getGeneration() >= GFX12; }

  bool hasINVWBL2WaitCntRequirement() const { return HasGFX1250Insts; }

  bool hasVOPD3() const { return HasGFX1250Insts; }

  // \returns true if the target has V_MUL_U64/V_MUL_I64 instructions.
  bool hasVectorMulU64() const { return HasGFX1250Insts; }

  // \returns true if the target has V_MAD_NC_U64_U32/V_MAD_NC_I64_I32
  // instructions.
  bool hasMadU64U32NoCarry() const { return HasGFX1250Insts; }

  // \returns true if the target has V_{MIN|MAX}_{I|U}64 instructions.
  bool hasIntMinMax64() const { return HasGFX1250Insts; }

  // \returns true if the target has V_PK_{MIN|MAX}3_{I|U}16 instructions.
  bool hasPkMinMax3Insts() const { return HasGFX1250Insts; }

  // \returns ture if target has S_GET_SHADER_CYCLES_U64 instruction.
  bool hasSGetShaderCyclesInst() const { return HasGFX1250Insts; }

  // \returns true if S_GETPC_B64 zero-extends the result from 48 bits instead
  // of sign-extending. Note that GFX1250 has not only fixed the bug but also
  // extended VA to 57 bits.
  bool hasGetPCZeroExtension() const {
    return HasGFX12Insts && !HasGFX1250Insts;
  }

  // \returns true if the target needs to create a prolog for backward
  // compatibility when preloading kernel arguments.
  bool needsKernArgPreloadProlog() const {
    return hasKernargPreload() && !HasGFX1250Insts;
  }

  bool hasCondSubInsts() const { return HasGFX12Insts; }

  bool hasSubClampInsts() const { return hasGFX10_3Insts(); }

  /// \returns SGPR allocation granularity supported by the subtarget.
  unsigned getSGPRAllocGranule() const {
    return AMDGPU::IsaInfo::getSGPRAllocGranule(this);
  }

  /// \returns SGPR encoding granularity supported by the subtarget.
  unsigned getSGPREncodingGranule() const {
    return AMDGPU::IsaInfo::getSGPREncodingGranule(this);
  }

  /// \returns Total number of SGPRs supported by the subtarget.
  unsigned getTotalNumSGPRs() const {
    return AMDGPU::IsaInfo::getTotalNumSGPRs(this);
  }

  /// \returns Addressable number of SGPRs supported by the subtarget.
  unsigned getAddressableNumSGPRs() const {
    return AMDGPU::IsaInfo::getAddressableNumSGPRs(this);
  }

  /// \returns Minimum number of SGPRs that meets the given number of waves per
  /// execution unit requirement supported by the subtarget.
  unsigned getMinNumSGPRs(unsigned WavesPerEU) const {
    return AMDGPU::IsaInfo::getMinNumSGPRs(this, WavesPerEU);
  }

  /// \returns Maximum number of SGPRs that meets the given number of waves per
  /// execution unit requirement supported by the subtarget.
  unsigned getMaxNumSGPRs(unsigned WavesPerEU, bool Addressable) const {
    return AMDGPU::IsaInfo::getMaxNumSGPRs(this, WavesPerEU, Addressable);
  }

  /// \returns Reserved number of SGPRs. This is common
  /// utility function called by MachineFunction and
  /// Function variants of getReservedNumSGPRs.
  unsigned getBaseReservedNumSGPRs(const bool HasFlatScratch) const;
  /// \returns Reserved number of SGPRs for given machine function \p MF.
  unsigned getReservedNumSGPRs(const MachineFunction &MF) const;

  /// \returns Reserved number of SGPRs for given function \p F.
  unsigned getReservedNumSGPRs(const Function &F) const;

  /// \returns Maximum number of preloaded SGPRs for the subtarget.
  unsigned getMaxNumPreloadedSGPRs() const;

  /// \returns max num SGPRs. This is the common utility
  /// function called by MachineFunction and Function
  /// variants of getMaxNumSGPRs.
  unsigned getBaseMaxNumSGPRs(const Function &F,
                              std::pair<unsigned, unsigned> WavesPerEU,
                              unsigned PreloadedSGPRs,
                              unsigned ReservedNumSGPRs) const;

  /// \returns Maximum number of SGPRs that meets number of waves per execution
  /// unit requirement for function \p MF, or number of SGPRs explicitly
  /// requested using "amdgpu-num-sgpr" attribute attached to function \p MF.
  ///
  /// \returns Value that meets number of waves per execution unit requirement
  /// if explicitly requested value cannot be converted to integer, violates
  /// subtarget's specifications, or does not meet number of waves per execution
  /// unit requirement.
  unsigned getMaxNumSGPRs(const MachineFunction &MF) const;

  /// \returns Maximum number of SGPRs that meets number of waves per execution
  /// unit requirement for function \p F, or number of SGPRs explicitly
  /// requested using "amdgpu-num-sgpr" attribute attached to function \p F.
  ///
  /// \returns Value that meets number of waves per execution unit requirement
  /// if explicitly requested value cannot be converted to integer, violates
  /// subtarget's specifications, or does not meet number of waves per execution
  /// unit requirement.
  unsigned getMaxNumSGPRs(const Function &F) const;

  /// \returns VGPR allocation granularity supported by the subtarget.
  unsigned getVGPRAllocGranule(unsigned DynamicVGPRBlockSize) const {
    return AMDGPU::IsaInfo::getVGPRAllocGranule(this, DynamicVGPRBlockSize);
  }

  /// \returns VGPR encoding granularity supported by the subtarget.
  unsigned getVGPREncodingGranule() const {
    return AMDGPU::IsaInfo::getVGPREncodingGranule(this);
  }

  /// \returns Total number of VGPRs supported by the subtarget.
  unsigned getTotalNumVGPRs() const {
    return AMDGPU::IsaInfo::getTotalNumVGPRs(this);
  }

  /// \returns Addressable number of architectural VGPRs supported by the
  /// subtarget.
  unsigned getAddressableNumArchVGPRs() const {
    return AMDGPU::IsaInfo::getAddressableNumArchVGPRs(this);
  }

  /// \returns Addressable number of VGPRs supported by the subtarget.
  unsigned getAddressableNumVGPRs(unsigned DynamicVGPRBlockSize) const {
    return AMDGPU::IsaInfo::getAddressableNumVGPRs(this, DynamicVGPRBlockSize);
  }

  /// \returns the minimum number of VGPRs that will prevent achieving more than
  /// the specified number of waves \p WavesPerEU.
  unsigned getMinNumVGPRs(unsigned WavesPerEU,
                          unsigned DynamicVGPRBlockSize) const {
    return AMDGPU::IsaInfo::getMinNumVGPRs(this, WavesPerEU,
                                           DynamicVGPRBlockSize);
  }

  /// \returns the maximum number of VGPRs that can be used and still achieved
  /// at least the specified number of waves \p WavesPerEU.
  unsigned getMaxNumVGPRs(unsigned WavesPerEU,
                          unsigned DynamicVGPRBlockSize) const {
    return AMDGPU::IsaInfo::getMaxNumVGPRs(this, WavesPerEU,
                                           DynamicVGPRBlockSize);
  }

  /// \returns max num VGPRs. This is the common utility function
  /// called by MachineFunction and Function variants of getMaxNumVGPRs.
  unsigned
  getBaseMaxNumVGPRs(const Function &F,
                     std::pair<unsigned, unsigned> NumVGPRBounds) const;

  /// \returns Maximum number of VGPRs that meets number of waves per execution
  /// unit requirement for function \p F, or number of VGPRs explicitly
  /// requested using "amdgpu-num-vgpr" attribute attached to function \p F.
  ///
  /// \returns Value that meets number of waves per execution unit requirement
  /// if explicitly requested value cannot be converted to integer, violates
  /// subtarget's specifications, or does not meet number of waves per execution
  /// unit requirement.
  unsigned getMaxNumVGPRs(const Function &F) const;

  unsigned getMaxNumAGPRs(const Function &F) const { return getMaxNumVGPRs(F); }

  /// Return a pair of maximum numbers of VGPRs and AGPRs that meet the number
  /// of waves per execution unit required for the function \p MF.
  std::pair<unsigned, unsigned> getMaxNumVectorRegs(const Function &F) const;

  /// \returns Maximum number of VGPRs that meets number of waves per execution
  /// unit requirement for function \p MF, or number of VGPRs explicitly
  /// requested using "amdgpu-num-vgpr" attribute attached to function \p MF.
  ///
  /// \returns Value that meets number of waves per execution unit requirement
  /// if explicitly requested value cannot be converted to integer, violates
  /// subtarget's specifications, or does not meet number of waves per execution
  /// unit requirement.
  unsigned getMaxNumVGPRs(const MachineFunction &MF) const;

  bool supportsWave32() const { return getGeneration() >= GFX10; }

  bool supportsWave64() const { return !hasGFX1250Insts(); }

  bool isWave32() const { return getWavefrontSize() == 32; }

  bool isWave64() const { return getWavefrontSize() == 64; }

  /// Returns if the wavesize of this subtarget is known reliable. This is false
  /// only for the a default target-cpu that does not have an explicit
  /// +wavefrontsize target feature.
  bool isWaveSizeKnown() const {
    return hasFeature(AMDGPU::FeatureWavefrontSize32) ||
           hasFeature(AMDGPU::FeatureWavefrontSize64);
  }

  const TargetRegisterClass *getBoolRC() const {
    return getRegisterInfo()->getBoolRC();
  }

  /// \returns Maximum number of work groups per compute unit supported by the
  /// subtarget and limited by given \p FlatWorkGroupSize.
  unsigned getMaxWorkGroupsPerCU(unsigned FlatWorkGroupSize) const override {
    return AMDGPU::IsaInfo::getMaxWorkGroupsPerCU(this, FlatWorkGroupSize);
  }

  /// \returns Minimum flat work group size supported by the subtarget.
  unsigned getMinFlatWorkGroupSize() const override {
    return AMDGPU::IsaInfo::getMinFlatWorkGroupSize(this);
  }

  /// \returns Maximum flat work group size supported by the subtarget.
  unsigned getMaxFlatWorkGroupSize() const override {
    return AMDGPU::IsaInfo::getMaxFlatWorkGroupSize(this);
  }

  /// \returns Number of waves per execution unit required to support the given
  /// \p FlatWorkGroupSize.
  unsigned
  getWavesPerEUForWorkGroup(unsigned FlatWorkGroupSize) const override {
    return AMDGPU::IsaInfo::getWavesPerEUForWorkGroup(this, FlatWorkGroupSize);
  }

  /// \returns Minimum number of waves per execution unit supported by the
  /// subtarget.
  unsigned getMinWavesPerEU() const override {
    return AMDGPU::IsaInfo::getMinWavesPerEU(this);
  }

  void adjustSchedDependency(SUnit *Def, int DefOpIdx, SUnit *Use, int UseOpIdx,
                             SDep &Dep,
                             const TargetSchedModel *SchedModel) const override;

  // \returns true if it's beneficial on this subtarget for the scheduler to
  // cluster stores as well as loads.
  bool shouldClusterStores() const { return getGeneration() >= GFX11; }

  // \returns the number of address arguments from which to enable MIMG NSA
  // on supported architectures.
  unsigned getNSAThreshold(const MachineFunction &MF) const;

  // \returns true if the subtarget has a hazard requiring an "s_nop 0"
  // instruction before "s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)".
  bool requiresNopBeforeDeallocVGPRs() const { return !HasGFX1250Insts; }

  // \returns true if the subtarget needs S_WAIT_ALU 0 before S_GETREG_B32 on
  // STATUS, STATE_PRIV, EXCP_FLAG_PRIV, or EXCP_FLAG_USER.
  bool requiresWaitIdleBeforeGetReg() const { return HasGFX1250Insts; }

  bool isDynamicVGPREnabled() const { return DynamicVGPR; }
  unsigned getDynamicVGPRBlockSize() const {
    return DynamicVGPRBlockSize32 ? 32 : 16;
  }

  bool requiresDisjointEarlyClobberAndUndef() const override {
    // AMDGPU doesn't care if early-clobber and undef operands are allocated
    // to the same register.
    return false;
  }

  // DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64 shall not be claused with anything
  // and surronded by S_WAIT_ALU(0xFFE3).
  bool hasDsAtomicAsyncBarrierArriveB64PipeBug() const {
    return getGeneration() == GFX12;
  }

  // Requires s_wait_alu(0) after s102/s103 write and src_flat_scratch_base
  // read.
  bool hasScratchBaseForwardingHazard() const {
    return HasGFX1250Insts && getGeneration() == GFX12;
  }

  // src_flat_scratch_hi cannot be used as a source in SALU producing a 64-bit
  // result.
  bool hasFlatScratchHiInB64InstHazard() const {
    return HasGFX1250Insts && getGeneration() == GFX12;
  }

  /// \returns true if the subtarget requires a wait for xcnt before VMEM
  /// accesses that must never be repeated in the event of a page fault/re-try.
  /// Atomic stores/rmw and all volatile accesses fall under this criteria.
  bool requiresWaitXCntForSingleAccessInstructions() const {
    return HasGFX1250Insts;
  }

  /// \returns the number of significant bits in the immediate field of the
  /// S_NOP instruction.
  unsigned getSNopBits() const {
    if (getGeneration() >= AMDGPUSubtarget::GFX12)
      return 7;
    if (getGeneration() >= AMDGPUSubtarget::VOLCANIC_ISLANDS)
      return 4;
    return 3;
  }

  bool supportsBPermute() const {
    return getGeneration() >= AMDGPUSubtarget::VOLCANIC_ISLANDS;
  }

  bool supportsWaveWideBPermute() const {
    return (getGeneration() <= AMDGPUSubtarget::GFX9 ||
            getGeneration() == AMDGPUSubtarget::GFX12) ||
           isWave32();
  }

  /// Return true if real (non-fake) variants of True16 instructions using
  /// 16-bit registers should be code-generated. Fake True16 instructions are
  /// identical to non-fake ones except that they take 32-bit registers as
  /// operands and always use their low halves.
  // TODO: Remove and use hasTrue16BitInsts() instead once True16 is fully
  // supported and the support for fake True16 instructions is removed.
  bool useRealTrue16Insts() const {
    return hasTrue16BitInsts() && EnableRealTrue16Insts;
  }
};

class GCNUserSGPRUsageInfo {
public:
  bool hasImplicitBufferPtr() const { return ImplicitBufferPtr; }

  bool hasPrivateSegmentBuffer() const { return PrivateSegmentBuffer; }

  bool hasDispatchPtr() const { return DispatchPtr; }

  bool hasQueuePtr() const { return QueuePtr; }

  bool hasKernargSegmentPtr() const { return KernargSegmentPtr; }

  bool hasDispatchID() const { return DispatchID; }

  bool hasFlatScratchInit() const { return FlatScratchInit; }

  bool hasPrivateSegmentSize() const { return PrivateSegmentSize; }

  unsigned getNumKernargPreloadSGPRs() const { return NumKernargPreloadSGPRs; }

  unsigned getNumUsedUserSGPRs() const { return NumUsedUserSGPRs; }

  unsigned getNumFreeUserSGPRs();

  void allocKernargPreloadSGPRs(unsigned NumSGPRs);

  enum UserSGPRID : unsigned {
    ImplicitBufferPtrID = 0,
    PrivateSegmentBufferID = 1,
    DispatchPtrID = 2,
    QueuePtrID = 3,
    KernargSegmentPtrID = 4,
    DispatchIdID = 5,
    FlatScratchInitID = 6,
    PrivateSegmentSizeID = 7
  };

  // Returns the size in number of SGPRs for preload user SGPR field.
  static unsigned getNumUserSGPRForField(UserSGPRID ID) {
    switch (ID) {
    case ImplicitBufferPtrID:
      return 2;
    case PrivateSegmentBufferID:
      return 4;
    case DispatchPtrID:
      return 2;
    case QueuePtrID:
      return 2;
    case KernargSegmentPtrID:
      return 2;
    case DispatchIdID:
      return 2;
    case FlatScratchInitID:
      return 2;
    case PrivateSegmentSizeID:
      return 1;
    }
    llvm_unreachable("Unknown UserSGPRID.");
  }

  GCNUserSGPRUsageInfo(const Function &F, const GCNSubtarget &ST);

private:
  const GCNSubtarget &ST;

  // Private memory buffer
  // Compute directly in sgpr[0:1]
  // Other shaders indirect 64-bits at sgpr[0:1]
  bool ImplicitBufferPtr = false;

  bool PrivateSegmentBuffer = false;

  bool DispatchPtr = false;

  bool QueuePtr = false;

  bool KernargSegmentPtr = false;

  bool DispatchID = false;

  bool FlatScratchInit = false;

  bool PrivateSegmentSize = false;

  unsigned NumKernargPreloadSGPRs = 0;

  unsigned NumUsedUserSGPRs = 0;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_GCNSUBTARGET_H
