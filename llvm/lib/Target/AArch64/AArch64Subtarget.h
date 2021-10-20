//===--- AArch64Subtarget.h - Define Subtarget for the AArch64 -*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the AArch64 specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AARCH64_AARCH64SUBTARGET_H
#define LLVM_LIB_TARGET_AARCH64_AARCH64SUBTARGET_H

#include "AArch64FrameLowering.h"
#include "AArch64ISelLowering.h"
#include "AArch64InstrInfo.h"
#include "AArch64RegisterInfo.h"
#include "AArch64SelectionDAGInfo.h"
#include "llvm/CodeGen/GlobalISel/CallLowering.h"
#include "llvm/CodeGen/GlobalISel/InlineAsmLowering.h"
#include "llvm/CodeGen/GlobalISel/InstructionSelector.h"
#include "llvm/CodeGen/GlobalISel/LegalizerInfo.h"
#include "llvm/CodeGen/GlobalISel/RegisterBankInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DataLayout.h"
#include <string>

#define GET_SUBTARGETINFO_HEADER
#include "AArch64GenSubtargetInfo.inc"

namespace llvm {
class GlobalValue;
class StringRef;
class Triple;

class AArch64Subtarget final : public AArch64GenSubtargetInfo {
public:
  enum ARMProcFamilyEnum : uint8_t {
    Others,
    A64FX,
    AppleA7,
    AppleA10,
    AppleA11,
    AppleA12,
    AppleA13,
    AppleA14,
    Carmel,
    CortexA35,
    CortexA53,
    CortexA55,
    CortexA510,
    CortexA57,
    CortexA65,
    CortexA72,
    CortexA73,
    CortexA75,
    CortexA76,
    CortexA77,
    CortexA78,
    CortexA78C,
    CortexR82,
    CortexX1,
    ExynosM3,
    Falkor,
    Kryo,
    NeoverseE1,
    NeoverseN1,
    NeoverseN2,
    NeoverseV1,
    Saphira,
    ThunderX2T99,
    ThunderX,
    ThunderXT81,
    ThunderXT83,
    ThunderXT88,
    ThunderX3T110,
    TSV110
  };

protected:
  /// ARMProcFamily - ARM processor family: Cortex-A53, Cortex-A57, and others.
  ARMProcFamilyEnum ARMProcFamily = Others;

  bool HasV8_1aOps = false;
  bool HasV8_2aOps = false;
  bool HasV8_3aOps = false;
  bool HasV8_4aOps = false;
  bool HasV8_5aOps = false;
  bool HasV8_6aOps = false;
  bool HasV8_7aOps = false;
  bool HasV9_0aOps = false;
  bool HasV9_1aOps = false;
  bool HasV9_2aOps = false;

  bool HasV8_0rOps = false;
  bool HasCONTEXTIDREL2 = false;

  bool HasFPARMv8 = false;
  bool HasNEON = false;
  bool HasCrypto = false;
  bool HasDotProd = false;
  bool HasCRC = false;
  bool HasLSE = false;
  bool HasLSE2 = false;
  bool HasRAS = false;
  bool HasRDM = false;
  bool HasPerfMon = false;
  bool HasFullFP16 = false;
  bool HasFP16FML = false;
  bool HasSPE = false;

  // ARMv8.1 extensions
  bool HasVH = false;
  bool HasPAN = false;
  bool HasLOR = false;

  // ARMv8.2 extensions
  bool HasPsUAO = false;
  bool HasPAN_RWV = false;
  bool HasCCPP = false;

  // SVE extensions
  bool HasSVE = false;
  bool UseExperimentalZeroingPseudos = false;
  bool UseScalarIncVL = false;

  // Armv8.2 Crypto extensions
  bool HasSM4 = false;
  bool HasSHA3 = false;
  bool HasSHA2 = false;
  bool HasAES = false;

  // ARMv8.3 extensions
  bool HasPAuth = false;
  bool HasJS = false;
  bool HasCCIDX = false;
  bool HasComplxNum = false;

  // ARMv8.4 extensions
  bool HasNV = false;
  bool HasMPAM = false;
  bool HasDIT = false;
  bool HasTRACEV8_4 = false;
  bool HasAM = false;
  bool HasSEL2 = false;
  bool HasTLB_RMI = false;
  bool HasFlagM = false;
  bool HasRCPC_IMMO = false;

  bool HasLSLFast = false;
  bool HasRCPC = false;
  bool HasAggressiveFMA = false;

  // Armv8.5-A Extensions
  bool HasAlternativeNZCV = false;
  bool HasFRInt3264 = false;
  bool HasSpecRestrict = false;
  bool HasSSBS = false;
  bool HasSB = false;
  bool HasPredRes = false;
  bool HasCCDP = false;
  bool HasBTI = false;
  bool HasRandGen = false;
  bool HasMTE = false;
  bool HasTME = false;

  // Armv8.6-A Extensions
  bool HasBF16 = false;
  bool HasMatMulInt8 = false;
  bool HasMatMulFP32 = false;
  bool HasMatMulFP64 = false;
  bool HasAMVS = false;
  bool HasFineGrainedTraps = false;
  bool HasEnhancedCounterVirtualization = false;

  // Armv8.7-A Extensions
  bool HasXS = false;
  bool HasWFxT = false;
  bool HasHCX = false;
  bool HasLS64 = false;

  // Arm SVE2 extensions
  bool HasSVE2 = false;
  bool HasSVE2AES = false;
  bool HasSVE2SM4 = false;
  bool HasSVE2SHA3 = false;
  bool HasSVE2BitPerm = false;

  // Armv9-A Extensions
  bool HasRME = false;

  // Arm Scalable Matrix Extension (SME)
  bool HasSME = false;
  bool HasSMEF64 = false;
  bool HasSMEI64 = false;
  bool HasStreamingSVE = false;

  // Future architecture extensions.
  bool HasETE = false;
  bool HasTRBE = false;
  bool HasBRBE = false;
  bool HasPAUTH = false;
  bool HasSPE_EEF = false;

  // HasZeroCycleRegMove - Has zero-cycle register mov instructions.
  bool HasZeroCycleRegMove = false;

  // HasZeroCycleZeroing - Has zero-cycle zeroing instructions.
  bool HasZeroCycleZeroing = false;
  bool HasZeroCycleZeroingGP = false;
  bool HasZeroCycleZeroingFPWorkaround = false;

  // It is generally beneficial to rewrite "fmov s0, wzr" to "movi d0, #0".
  // as movi is more efficient across all cores. Newer cores can eliminate
  // fmovs early and there is no difference with movi, but this not true for
  // all implementations.
  bool HasZeroCycleZeroingFP = true;

  // StrictAlign - Disallow unaligned memory accesses.
  bool StrictAlign = false;

  // NegativeImmediates - transform instructions with negative immediates
  bool NegativeImmediates = true;

  // Enable 64-bit vectorization in SLP.
  unsigned MinVectorRegisterBitWidth = 64;

  bool OutlineAtomics = false;
  bool PredictableSelectIsExpensive = false;
  bool BalanceFPOps = false;
  bool CustomAsCheapAsMove = false;
  bool ExynosAsCheapAsMove = false;
  bool UsePostRAScheduler = false;
  bool Misaligned128StoreIsSlow = false;
  bool Paired128IsSlow = false;
  bool STRQroIsSlow = false;
  bool UseAlternateSExtLoadCVTF32Pattern = false;
  bool HasArithmeticBccFusion = false;
  bool HasArithmeticCbzFusion = false;
  bool HasCmpBccFusion = false;
  bool HasFuseAddress = false;
  bool HasFuseAES = false;
  bool HasFuseArithmeticLogic = false;
  bool HasFuseCCSelect = false;
  bool HasFuseCryptoEOR = false;
  bool HasFuseLiterals = false;
  bool DisableLatencySchedHeuristic = false;
  bool UseRSqrt = false;
  bool Force32BitJumpTables = false;
  bool UseEL1ForTP = false;
  bool UseEL2ForTP = false;
  bool UseEL3ForTP = false;
  bool AllowTaggedGlobals = false;
  bool HardenSlsRetBr = false;
  bool HardenSlsBlr = false;
  bool HardenSlsNoComdat = false;
  uint8_t MaxInterleaveFactor = 2;
  uint8_t VectorInsertExtractBaseCost = 3;
  uint16_t CacheLineSize = 0;
  uint16_t PrefetchDistance = 0;
  uint16_t MinPrefetchStride = 1;
  unsigned MaxPrefetchIterationsAhead = UINT_MAX;
  unsigned PrefFunctionLogAlignment = 0;
  unsigned PrefLoopLogAlignment = 0;
  unsigned MaxJumpTableSize = 0;
  unsigned WideningBaseCost = 0;

  // ReserveXRegister[i] - X#i is not available as a general purpose register.
  BitVector ReserveXRegister;

  // CustomCallUsedXRegister[i] - X#i call saved.
  BitVector CustomCallSavedXRegs;

  bool IsLittle;

  unsigned MinSVEVectorSizeInBits;
  unsigned MaxSVEVectorSizeInBits;

  /// TargetTriple - What processor and OS we're targeting.
  Triple TargetTriple;

  AArch64FrameLowering FrameLowering;
  AArch64InstrInfo InstrInfo;
  AArch64SelectionDAGInfo TSInfo;
  AArch64TargetLowering TLInfo;

  /// GlobalISel related APIs.
  std::unique_ptr<CallLowering> CallLoweringInfo;
  std::unique_ptr<InlineAsmLowering> InlineAsmLoweringInfo;
  std::unique_ptr<InstructionSelector> InstSelector;
  std::unique_ptr<LegalizerInfo> Legalizer;
  std::unique_ptr<RegisterBankInfo> RegBankInfo;

private:
  /// initializeSubtargetDependencies - Initializes using CPUString and the
  /// passed in feature string so that we can use initializer lists for
  /// subtarget initialization.
  AArch64Subtarget &initializeSubtargetDependencies(StringRef FS,
                                                    StringRef CPUString,
                                                    StringRef TuneCPUString);

  /// Initialize properties based on the selected processor family.
  void initializeProperties();

public:
  /// This constructor initializes the data members to match that
  /// of the specified triple.
  AArch64Subtarget(const Triple &TT, const std::string &CPU,
                   const std::string &TuneCPU, const std::string &FS,
                   const TargetMachine &TM, bool LittleEndian,
                   unsigned MinSVEVectorSizeInBitsOverride = 0,
                   unsigned MaxSVEVectorSizeInBitsOverride = 0);

  const AArch64SelectionDAGInfo *getSelectionDAGInfo() const override {
    return &TSInfo;
  }
  const AArch64FrameLowering *getFrameLowering() const override {
    return &FrameLowering;
  }
  const AArch64TargetLowering *getTargetLowering() const override {
    return &TLInfo;
  }
  const AArch64InstrInfo *getInstrInfo() const override { return &InstrInfo; }
  const AArch64RegisterInfo *getRegisterInfo() const override {
    return &getInstrInfo()->getRegisterInfo();
  }
  const CallLowering *getCallLowering() const override;
  const InlineAsmLowering *getInlineAsmLowering() const override;
  InstructionSelector *getInstructionSelector() const override;
  const LegalizerInfo *getLegalizerInfo() const override;
  const RegisterBankInfo *getRegBankInfo() const override;
  const Triple &getTargetTriple() const { return TargetTriple; }
  bool enableMachineScheduler() const override { return true; }
  bool enablePostRAScheduler() const override {
    return UsePostRAScheduler;
  }

  /// Returns ARM processor family.
  /// Avoid this function! CPU specifics should be kept local to this class
  /// and preferably modeled with SubtargetFeatures or properties in
  /// initializeProperties().
  ARMProcFamilyEnum getProcFamily() const {
    return ARMProcFamily;
  }

  bool hasV8_1aOps() const { return HasV8_1aOps; }
  bool hasV8_2aOps() const { return HasV8_2aOps; }
  bool hasV8_3aOps() const { return HasV8_3aOps; }
  bool hasV8_4aOps() const { return HasV8_4aOps; }
  bool hasV8_5aOps() const { return HasV8_5aOps; }
  bool hasV9_0aOps() const { return HasV9_0aOps; }
  bool hasV9_1aOps() const { return HasV9_1aOps; }
  bool hasV9_2aOps() const { return HasV9_2aOps; }
  bool hasV8_0rOps() const { return HasV8_0rOps; }

  bool hasZeroCycleRegMove() const { return HasZeroCycleRegMove; }

  bool hasZeroCycleZeroingGP() const { return HasZeroCycleZeroingGP; }

  bool hasZeroCycleZeroingFP() const { return HasZeroCycleZeroingFP; }

  bool hasZeroCycleZeroingFPWorkaround() const {
    return HasZeroCycleZeroingFPWorkaround;
  }

  bool requiresStrictAlign() const { return StrictAlign; }

  bool isXRaySupported() const override { return true; }

  unsigned getMinVectorRegisterBitWidth() const {
    return MinVectorRegisterBitWidth;
  }

  bool isXRegisterReserved(size_t i) const { return ReserveXRegister[i]; }
  unsigned getNumXRegisterReserved() const { return ReserveXRegister.count(); }
  bool isXRegCustomCalleeSaved(size_t i) const {
    return CustomCallSavedXRegs[i];
  }
  bool hasCustomCallingConv() const { return CustomCallSavedXRegs.any(); }
  bool hasFPARMv8() const { return HasFPARMv8; }
  bool hasNEON() const { return HasNEON; }
  bool hasCrypto() const { return HasCrypto; }
  bool hasDotProd() const { return HasDotProd; }
  bool hasCRC() const { return HasCRC; }
  bool hasLSE() const { return HasLSE; }
  bool hasLSE2() const { return HasLSE2; }
  bool hasRAS() const { return HasRAS; }
  bool hasRDM() const { return HasRDM; }
  bool hasSM4() const { return HasSM4; }
  bool hasSHA3() const { return HasSHA3; }
  bool hasSHA2() const { return HasSHA2; }
  bool hasAES() const { return HasAES; }
  bool hasCONTEXTIDREL2() const { return HasCONTEXTIDREL2; }
  bool balanceFPOps() const { return BalanceFPOps; }
  bool predictableSelectIsExpensive() const {
    return PredictableSelectIsExpensive;
  }
  bool hasCustomCheapAsMoveHandling() const { return CustomAsCheapAsMove; }
  bool hasExynosCheapAsMoveHandling() const { return ExynosAsCheapAsMove; }
  bool isMisaligned128StoreSlow() const { return Misaligned128StoreIsSlow; }
  bool isPaired128Slow() const { return Paired128IsSlow; }
  bool isSTRQroSlow() const { return STRQroIsSlow; }
  bool useAlternateSExtLoadCVTF32Pattern() const {
    return UseAlternateSExtLoadCVTF32Pattern;
  }
  bool hasArithmeticBccFusion() const { return HasArithmeticBccFusion; }
  bool hasArithmeticCbzFusion() const { return HasArithmeticCbzFusion; }
  bool hasCmpBccFusion() const { return HasCmpBccFusion; }
  bool hasFuseAddress() const { return HasFuseAddress; }
  bool hasFuseAES() const { return HasFuseAES; }
  bool hasFuseArithmeticLogic() const { return HasFuseArithmeticLogic; }
  bool hasFuseCCSelect() const { return HasFuseCCSelect; }
  bool hasFuseCryptoEOR() const { return HasFuseCryptoEOR; }
  bool hasFuseLiterals() const { return HasFuseLiterals; }

  /// Return true if the CPU supports any kind of instruction fusion.
  bool hasFusion() const {
    return hasArithmeticBccFusion() || hasArithmeticCbzFusion() ||
           hasFuseAES() || hasFuseArithmeticLogic() ||
           hasFuseCCSelect() || hasFuseLiterals();
  }

  bool hardenSlsRetBr() const { return HardenSlsRetBr; }
  bool hardenSlsBlr() const { return HardenSlsBlr; }
  bool hardenSlsNoComdat() const { return HardenSlsNoComdat; }

  bool useEL1ForTP() const { return UseEL1ForTP; }
  bool useEL2ForTP() const { return UseEL2ForTP; }
  bool useEL3ForTP() const { return UseEL3ForTP; }

  bool useRSqrt() const { return UseRSqrt; }
  bool force32BitJumpTables() const { return Force32BitJumpTables; }
  unsigned getMaxInterleaveFactor() const { return MaxInterleaveFactor; }
  unsigned getVectorInsertExtractBaseCost() const {
    return VectorInsertExtractBaseCost;
  }
  unsigned getCacheLineSize() const override { return CacheLineSize; }
  unsigned getPrefetchDistance() const override { return PrefetchDistance; }
  unsigned getMinPrefetchStride(unsigned NumMemAccesses,
                                unsigned NumStridedMemAccesses,
                                unsigned NumPrefetches,
                                bool HasCall) const override {
    return MinPrefetchStride;
  }
  unsigned getMaxPrefetchIterationsAhead() const override {
    return MaxPrefetchIterationsAhead;
  }
  unsigned getPrefFunctionLogAlignment() const {
    return PrefFunctionLogAlignment;
  }
  unsigned getPrefLoopLogAlignment() const { return PrefLoopLogAlignment; }

  unsigned getMaximumJumpTableSize() const { return MaxJumpTableSize; }

  unsigned getWideningBaseCost() const { return WideningBaseCost; }

  bool useExperimentalZeroingPseudos() const {
    return UseExperimentalZeroingPseudos;
  }

  bool useScalarIncVL() const { return UseScalarIncVL; }

  /// CPU has TBI (top byte of addresses is ignored during HW address
  /// translation) and OS enables it.
  bool supportsAddressTopByteIgnored() const;

  bool hasPerfMon() const { return HasPerfMon; }
  bool hasFullFP16() const { return HasFullFP16; }
  bool hasFP16FML() const { return HasFP16FML; }
  bool hasSPE() const { return HasSPE; }
  bool hasLSLFast() const { return HasLSLFast; }
  bool hasSVE() const { return HasSVE; }
  bool hasSVE2() const { return HasSVE2; }
  bool hasRCPC() const { return HasRCPC; }
  bool hasAggressiveFMA() const { return HasAggressiveFMA; }
  bool hasAlternativeNZCV() const { return HasAlternativeNZCV; }
  bool hasFRInt3264() const { return HasFRInt3264; }
  bool hasSpecRestrict() const { return HasSpecRestrict; }
  bool hasSSBS() const { return HasSSBS; }
  bool hasSB() const { return HasSB; }
  bool hasPredRes() const { return HasPredRes; }
  bool hasCCDP() const { return HasCCDP; }
  bool hasBTI() const { return HasBTI; }
  bool hasRandGen() const { return HasRandGen; }
  bool hasMTE() const { return HasMTE; }
  bool hasTME() const { return HasTME; }
  bool hasPAUTH() const { return HasPAUTH; }
  // Arm SVE2 extensions
  bool hasSVE2AES() const { return HasSVE2AES; }
  bool hasSVE2SM4() const { return HasSVE2SM4; }
  bool hasSVE2SHA3() const { return HasSVE2SHA3; }
  bool hasSVE2BitPerm() const { return HasSVE2BitPerm; }
  bool hasMatMulInt8() const { return HasMatMulInt8; }
  bool hasMatMulFP32() const { return HasMatMulFP32; }
  bool hasMatMulFP64() const { return HasMatMulFP64; }

  // Armv8.6-A Extensions
  bool hasBF16() const { return HasBF16; }
  bool hasFineGrainedTraps() const { return HasFineGrainedTraps; }
  bool hasEnhancedCounterVirtualization() const {
    return HasEnhancedCounterVirtualization;
  }

  // Arm Scalable Matrix Extension (SME)
  bool hasSME() const { return HasSME; }
  bool hasSMEF64() const { return HasSMEF64; }
  bool hasSMEI64() const { return HasSMEI64; }
  bool hasStreamingSVE() const { return HasStreamingSVE; }

  bool isLittleEndian() const { return IsLittle; }

  bool isTargetDarwin() const { return TargetTriple.isOSDarwin(); }
  bool isTargetIOS() const { return TargetTriple.isiOS(); }
  bool isTargetLinux() const { return TargetTriple.isOSLinux(); }
  bool isTargetWindows() const { return TargetTriple.isOSWindows(); }
  bool isTargetAndroid() const { return TargetTriple.isAndroid(); }
  bool isTargetFuchsia() const { return TargetTriple.isOSFuchsia(); }

  bool isTargetCOFF() const { return TargetTriple.isOSBinFormatCOFF(); }
  bool isTargetELF() const { return TargetTriple.isOSBinFormatELF(); }
  bool isTargetMachO() const { return TargetTriple.isOSBinFormatMachO(); }

  bool isTargetILP32() const {
    return TargetTriple.isArch32Bit() ||
           TargetTriple.getEnvironment() == Triple::GNUILP32;
  }

  bool useAA() const override;

  bool outlineAtomics() const { return OutlineAtomics; }

  bool hasVH() const { return HasVH; }
  bool hasPAN() const { return HasPAN; }
  bool hasLOR() const { return HasLOR; }

  bool hasPsUAO() const { return HasPsUAO; }
  bool hasPAN_RWV() const { return HasPAN_RWV; }
  bool hasCCPP() const { return HasCCPP; }

  bool hasPAuth() const { return HasPAuth; }
  bool hasJS() const { return HasJS; }
  bool hasCCIDX() const { return HasCCIDX; }
  bool hasComplxNum() const { return HasComplxNum; }

  bool hasNV() const { return HasNV; }
  bool hasMPAM() const { return HasMPAM; }
  bool hasDIT() const { return HasDIT; }
  bool hasTRACEV8_4() const { return HasTRACEV8_4; }
  bool hasAM() const { return HasAM; }
  bool hasAMVS() const { return HasAMVS; }
  bool hasXS() const { return HasXS; }
  bool hasWFxT() const { return HasWFxT; }
  bool hasHCX() const { return HasHCX; }
  bool hasLS64() const { return HasLS64; }
  bool hasSEL2() const { return HasSEL2; }
  bool hasTLB_RMI() const { return HasTLB_RMI; }
  bool hasFlagM() const { return HasFlagM; }
  bool hasRCPC_IMMO() const { return HasRCPC_IMMO; }

  bool addrSinkUsingGEPs() const override {
    // Keeping GEPs inbounds is important for exploiting AArch64
    // addressing-modes in ILP32 mode.
    return useAA() || isTargetILP32();
  }

  bool useSmallAddressing() const {
    switch (TLInfo.getTargetMachine().getCodeModel()) {
      case CodeModel::Kernel:
        // Kernel is currently allowed only for Fuchsia targets,
        // where it is the same as Small for almost all purposes.
      case CodeModel::Small:
        return true;
      default:
        return false;
    }
  }

  /// ParseSubtargetFeatures - Parses features string setting specified
  /// subtarget options.  Definition of function is auto generated by tblgen.
  void ParseSubtargetFeatures(StringRef CPU, StringRef TuneCPU, StringRef FS);

  /// ClassifyGlobalReference - Find the target operand flags that describe
  /// how a global value should be referenced for the current subtarget.
  unsigned ClassifyGlobalReference(const GlobalValue *GV,
                                   const TargetMachine &TM) const;

  unsigned classifyGlobalFunctionReference(const GlobalValue *GV,
                                           const TargetMachine &TM) const;

  void overrideSchedPolicy(MachineSchedPolicy &Policy,
                           unsigned NumRegionInstrs) const override;

  bool enableEarlyIfConversion() const override;

  bool enableAdvancedRASplitCost() const override { return false; }

  std::unique_ptr<PBQPRAConstraint> getCustomPBQPConstraints() const override;

  bool isCallingConvWin64(CallingConv::ID CC) const {
    switch (CC) {
    case CallingConv::C:
    case CallingConv::Fast:
    case CallingConv::Swift:
      return isTargetWindows();
    case CallingConv::Win64:
      return true;
    default:
      return false;
    }
  }

  /// Return whether FrameLowering should always set the "extended frame
  /// present" bit in FP, or set it based on a symbol in the runtime.
  bool swiftAsyncContextIsDynamicallySet() const {
    // Older OS versions (particularly system unwinders) are confused by the
    // Swift extended frame, so when building code that might be run on them we
    // must dynamically query the concurrency library to determine whether
    // extended frames should be flagged as present.
    const Triple &TT = getTargetTriple();

    unsigned Major, Minor, Micro;
    TT.getOSVersion(Major, Minor, Micro);
    switch(TT.getOS()) {
    default:
      return false;
    case Triple::IOS:
    case Triple::TvOS:
      return Major < 15;
    case Triple::WatchOS:
      return Major < 8;
    case Triple::MacOSX:
    case Triple::Darwin:
      return Major < 12;
    }
  }

  void mirFileLoaded(MachineFunction &MF) const override;

  // Return the known range for the bit length of SVE data registers. A value
  // of 0 means nothing is known about that particular limit beyong what's
  // implied by the architecture.
  unsigned getMaxSVEVectorSizeInBits() const {
    assert(HasSVE && "Tried to get SVE vector length without SVE support!");
    return MaxSVEVectorSizeInBits;
  }

  unsigned getMinSVEVectorSizeInBits() const {
    assert(HasSVE && "Tried to get SVE vector length without SVE support!");
    return MinSVEVectorSizeInBits;
  }

  bool useSVEForFixedLengthVectors() const;
};
} // End llvm namespace

#endif
