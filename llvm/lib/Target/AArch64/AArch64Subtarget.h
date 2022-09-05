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
#include "llvm/CodeGen/RegisterBankInfo.h"
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
    Ampere1,
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
    CortexA710,
    CortexR82,
    CortexX1,
    CortexX1C,
    CortexX2,
    ExynosM3,
    Falkor,
    Kryo,
    NeoverseE1,
    NeoverseN1,
    NeoverseN2,
    Neoverse512TVB,
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

  // Enable 64-bit vectorization in SLP.
  unsigned MinVectorRegisterBitWidth = 64;

// Bool members corresponding to the SubtargetFeatures defined in tablegen
#define GET_SUBTARGETINFO_MACRO(ATTRIBUTE, DEFAULT, GETTER)                    \
  bool ATTRIBUTE = DEFAULT;
#include "AArch64GenSubtargetInfo.inc"

  uint8_t MaxInterleaveFactor = 2;
  uint8_t VectorInsertExtractBaseCost = 3;
  uint16_t CacheLineSize = 0;
  uint16_t PrefetchDistance = 0;
  uint16_t MinPrefetchStride = 1;
  unsigned MaxPrefetchIterationsAhead = UINT_MAX;
  unsigned PrefFunctionLogAlignment = 0;
  unsigned PrefLoopLogAlignment = 0;
  unsigned MaxBytesForLoopAlignment = 0;
  unsigned MaxJumpTableSize = 0;

  // ReserveXRegister[i] - X#i is not available as a general purpose register.
  BitVector ReserveXRegister;

  // CustomCallUsedXRegister[i] - X#i call saved.
  BitVector CustomCallSavedXRegs;

  bool IsLittle;

  unsigned MinSVEVectorSizeInBits;
  unsigned MaxSVEVectorSizeInBits;
  unsigned VScaleForTuning = 2;

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

// Getters for SubtargetFeatures defined in tablegen
#define GET_SUBTARGETINFO_MACRO(ATTRIBUTE, DEFAULT, GETTER)                    \
  bool GETTER() const { return ATTRIBUTE; }
#include "AArch64GenSubtargetInfo.inc"

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
  bool enablePostRAScheduler() const override { return usePostRAScheduler(); }

  /// Returns ARM processor family.
  /// Avoid this function! CPU specifics should be kept local to this class
  /// and preferably modeled with SubtargetFeatures or properties in
  /// initializeProperties().
  ARMProcFamilyEnum getProcFamily() const {
    return ARMProcFamily;
  }

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

  /// Return true if the CPU supports any kind of instruction fusion.
  bool hasFusion() const {
    return hasArithmeticBccFusion() || hasArithmeticCbzFusion() ||
           hasFuseAES() || hasFuseArithmeticLogic() || hasFuseCCSelect() ||
           hasFuseAdrpAdd() || hasFuseLiterals();
  }

  unsigned getMaxInterleaveFactor() const { return MaxInterleaveFactor; }
  unsigned getVectorInsertExtractBaseCost() const;
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

  unsigned getMaxBytesForLoopAlignment() const {
    return MaxBytesForLoopAlignment;
  }

  unsigned getMaximumJumpTableSize() const { return MaxJumpTableSize; }

  /// CPU has TBI (top byte of addresses is ignored during HW address
  /// translation) and OS enables it.
  bool supportsAddressTopByteIgnored() const;

  bool isLittleEndian() const { return IsLittle; }

  bool isTargetDarwin() const { return TargetTriple.isOSDarwin(); }
  bool isTargetIOS() const { return TargetTriple.isiOS(); }
  bool isTargetLinux() const { return TargetTriple.isOSLinux(); }
  bool isTargetWindows() const { return TargetTriple.isOSWindows(); }
  bool isTargetAndroid() const { return TargetTriple.isAndroid(); }
  bool isTargetFuchsia() const { return TargetTriple.isOSFuchsia(); }
  bool isWindowsArm64EC() const { return TargetTriple.isWindowsArm64EC(); }

  bool isTargetCOFF() const { return TargetTriple.isOSBinFormatCOFF(); }
  bool isTargetELF() const { return TargetTriple.isOSBinFormatELF(); }
  bool isTargetMachO() const { return TargetTriple.isOSBinFormatMachO(); }

  bool isTargetILP32() const {
    return TargetTriple.isArch32Bit() ||
           TargetTriple.getEnvironment() == Triple::GNUILP32;
  }

  bool useAA() const override;

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

    unsigned Major = TT.getOSVersion().getMajor();
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

  bool useSVEForFixedLengthVectors() const {
    // Prefer NEON unless larger SVE registers are available.
    return hasSVE() && getMinSVEVectorSizeInBits() >= 256;
  }

  unsigned getVScaleForTuning() const { return VScaleForTuning; }
};
} // End llvm namespace

#endif
