//=====-- AMDGPUSubtarget.h - Define Subtarget for AMDGPU ------*- C++ -*-====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//
//
/// \file
/// AMDGPU specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUSUBTARGET_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUSUBTARGET_H

#include "AMDGPU.h"
#include "AMDGPUCallLowering.h"
#include "R600FrameLowering.h"
#include "R600ISelLowering.h"
#include "R600InstrInfo.h"
#include "SIFrameLowering.h"
#include "SIISelLowering.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/Triple.h"
#include "llvm/CodeGen/GlobalISel/InstructionSelector.h"
#include "llvm/CodeGen/GlobalISel/LegalizerInfo.h"
#include "llvm/CodeGen/GlobalISel/RegisterBankInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/SelectionDAGTargetInfo.h"
#include "llvm/MC/MCInstrItineraries.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>
#include <cstdint>
#include <memory>
#include <utility>

#define GET_SUBTARGETINFO_HEADER
#include "AMDGPUGenSubtargetInfo.inc"

namespace llvm {

class StringRef;

class AMDGPUSubtarget : public AMDGPUGenSubtargetInfo {
public:
  enum Generation {
    R600 = 0,
    R700,
    EVERGREEN,
    NORTHERN_ISLANDS,
    SOUTHERN_ISLANDS,
    SEA_ISLANDS,
    VOLCANIC_ISLANDS,
    GFX9,
  };

  enum {
    ISAVersion0_0_0,
    ISAVersion6_0_0,
    ISAVersion6_0_1,
    ISAVersion7_0_0,
    ISAVersion7_0_1,
    ISAVersion7_0_2,
    ISAVersion7_0_3,
    ISAVersion7_0_4,
    ISAVersion8_0_1,
    ISAVersion8_0_2,
    ISAVersion8_0_3,
    ISAVersion8_1_0,
    ISAVersion9_0_0,
    ISAVersion9_0_2,
    ISAVersion9_0_4,
    ISAVersion9_0_6,
  };

  enum TrapHandlerAbi {
    TrapHandlerAbiNone = 0,
    TrapHandlerAbiHsa = 1
  };

  enum TrapID {
    TrapIDHardwareReserved = 0,
    TrapIDHSADebugTrap = 1,
    TrapIDLLVMTrap = 2,
    TrapIDLLVMDebugTrap = 3,
    TrapIDDebugBreakpoint = 7,
    TrapIDDebugReserved8 = 8,
    TrapIDDebugReservedFE = 0xfe,
    TrapIDDebugReservedFF = 0xff
  };

  enum TrapRegValues {
    LLVMTrapHandlerRegValue = 1
  };

protected:
  // Basic subtarget description.
  Triple TargetTriple;
  Generation Gen;
  unsigned IsaVersion;
  unsigned WavefrontSize;
  int LocalMemorySize;
  int LDSBankCount;
  unsigned MaxPrivateElementSize;

  // Possibly statically set by tablegen, but may want to be overridden.
  bool FastFMAF32;
  bool HalfRate64Ops;

  // Dynamially set bits that enable features.
  bool FP32Denormals;
  bool FP64FP16Denormals;
  bool FPExceptions;
  bool DX10Clamp;
  bool FlatForGlobal;
  bool AutoWaitcntBeforeBarrier;
  bool CodeObjectV3;
  bool UnalignedScratchAccess;
  bool UnalignedBufferAccess;
  bool HasApertureRegs;
  bool EnableXNACK;
  bool TrapHandler;
  bool DebuggerInsertNops;
  bool DebuggerReserveRegs;
  bool DebuggerEmitPrologue;

  // Used as options.
  bool EnableHugePrivateBuffer;
  bool EnableVGPRSpilling;
  bool EnablePromoteAlloca;
  bool EnableLoadStoreOpt;
  bool EnableUnsafeDSOffsetFolding;
  bool EnableSIScheduler;
  bool EnableDS128;
  bool DumpCode;

  // Subtarget statically properties set by tablegen
  bool FP64;
  bool FMA;
  bool MIMG_R128;
  bool IsGCN;
  bool GCN3Encoding;
  bool CIInsts;
  bool GFX9Insts;
  bool SGPRInitBug;
  bool HasSMemRealTime;
  bool Has16BitInsts;
  bool HasIntClamp;
  bool HasVOP3PInsts;
  bool HasMadMixInsts;
  bool HasFmaMixInsts;
  bool HasMovrel;
  bool HasVGPRIndexMode;
  bool HasScalarStores;
  bool HasScalarAtomics;
  bool HasInv2PiInlineImm;
  bool HasSDWA;
  bool HasSDWAOmod;
  bool HasSDWAScalar;
  bool HasSDWASdst;
  bool HasSDWAMac;
  bool HasSDWAOutModsVOPC;
  bool HasDPP;
  bool HasDLInsts;
  bool D16PreservesUnusedBits;
  bool FlatAddressSpace;
  bool FlatInstOffsets;
  bool FlatGlobalInsts;
  bool FlatScratchInsts;
  bool AddNoCarryInsts;
  bool HasUnpackedD16VMem;
  bool R600ALUInst;
  bool CaymanISA;
  bool CFALUBug;
  bool HasVertexCache;
  short TexVTXClauseSize;
  bool ScalarizeGlobal;

  // Dummy feature to use for assembler in tablegen.
  bool FeatureDisable;

  InstrItineraryData InstrItins;
  SelectionDAGTargetInfo TSInfo;
  AMDGPUAS AS;

public:
  AMDGPUSubtarget(const Triple &TT, StringRef GPU, StringRef FS,
                  const TargetMachine &TM);
  ~AMDGPUSubtarget() override;

  AMDGPUSubtarget &initializeSubtargetDependencies(const Triple &TT,
                                                   StringRef GPU, StringRef FS);

  const AMDGPUInstrInfo *getInstrInfo() const override = 0;
  const AMDGPUFrameLowering *getFrameLowering() const override = 0;
  const AMDGPUTargetLowering *getTargetLowering() const override = 0;
  const AMDGPURegisterInfo *getRegisterInfo() const override = 0;

  const InstrItineraryData *getInstrItineraryData() const override {
    return &InstrItins;
  }

  // Nothing implemented, just prevent crashes on use.
  const SelectionDAGTargetInfo *getSelectionDAGInfo() const override {
    return &TSInfo;
  }

  void ParseSubtargetFeatures(StringRef CPU, StringRef FS);

  bool isAmdHsaOS() const {
    return TargetTriple.getOS() == Triple::AMDHSA;
  }

  bool isMesa3DOS() const {
    return TargetTriple.getOS() == Triple::Mesa3D;
  }

  bool isAmdPalOS() const {
    return TargetTriple.getOS() == Triple::AMDPAL;
  }

  Generation getGeneration() const {
    return Gen;
  }

  unsigned getWavefrontSize() const {
    return WavefrontSize;
  }

  unsigned getWavefrontSizeLog2() const {
    return Log2_32(WavefrontSize);
  }

  int getLocalMemorySize() const {
    return LocalMemorySize;
  }

  int getLDSBankCount() const {
    return LDSBankCount;
  }

  unsigned getMaxPrivateElementSize() const {
    return MaxPrivateElementSize;
  }

  AMDGPUAS getAMDGPUAS() const {
    return AS;
  }

  bool has16BitInsts() const {
    return Has16BitInsts;
  }

  bool hasIntClamp() const {
    return HasIntClamp;
  }

  bool hasVOP3PInsts() const {
    return HasVOP3PInsts;
  }

  bool hasFP64() const {
    return FP64;
  }

  bool hasMIMG_R128() const {
    return MIMG_R128;
  }

  bool hasFastFMAF32() const {
    return FastFMAF32;
  }

  bool hasHalfRate64Ops() const {
    return HalfRate64Ops;
  }

  bool hasAddr64() const {
    return (getGeneration() < VOLCANIC_ISLANDS);
  }

  bool hasBFE() const {
    return (getGeneration() >= EVERGREEN);
  }

  bool hasBFI() const {
    return (getGeneration() >= EVERGREEN);
  }

  bool hasBFM() const {
    return hasBFE();
  }

  bool hasBCNT(unsigned Size) const {
    if (Size == 32)
      return (getGeneration() >= EVERGREEN);

    if (Size == 64)
      return (getGeneration() >= SOUTHERN_ISLANDS);

    return false;
  }

  bool hasMulU24() const {
    return (getGeneration() >= EVERGREEN);
  }

  bool hasMulI24() const {
    return (getGeneration() >= SOUTHERN_ISLANDS ||
            hasCaymanISA());
  }

  bool hasFFBL() const {
    return (getGeneration() >= EVERGREEN);
  }

  bool hasFFBH() const {
    return (getGeneration() >= EVERGREEN);
  }

  bool hasMed3_16() const {
    return getGeneration() >= GFX9;
  }

  bool hasMin3Max3_16() const {
    return getGeneration() >= GFX9;
  }

  bool hasMadMixInsts() const {
    return HasMadMixInsts;
  }

  bool hasFmaMixInsts() const {
    return HasFmaMixInsts;
  }

  bool hasCARRY() const {
    return (getGeneration() >= EVERGREEN);
  }

  bool hasBORROW() const {
    return (getGeneration() >= EVERGREEN);
  }

  bool hasCaymanISA() const {
    return CaymanISA;
  }

  bool hasFMA() const {
    return FMA;
  }

  TrapHandlerAbi getTrapHandlerAbi() const {
    return isAmdHsaOS() ? TrapHandlerAbiHsa : TrapHandlerAbiNone;
  }

  bool enableHugePrivateBuffer() const {
    return EnableHugePrivateBuffer;
  }

  bool isPromoteAllocaEnabled() const {
    return EnablePromoteAlloca;
  }

  bool unsafeDSOffsetFoldingEnabled() const {
    return EnableUnsafeDSOffsetFolding;
  }

  bool dumpCode() const {
    return DumpCode;
  }

  /// Return the amount of LDS that can be used that will not restrict the
  /// occupancy lower than WaveCount.
  unsigned getMaxLocalMemSizeWithWaveCount(unsigned WaveCount,
                                           const Function &) const;

  /// Inverse of getMaxLocalMemWithWaveCount. Return the maximum wavecount if
  /// the given LDS memory size is the only constraint.
  unsigned getOccupancyWithLocalMemSize(uint32_t Bytes, const Function &) const;

  unsigned getOccupancyWithLocalMemSize(const MachineFunction &MF) const {
    const auto *MFI = MF.getInfo<SIMachineFunctionInfo>();
    return getOccupancyWithLocalMemSize(MFI->getLDSSize(), MF.getFunction());
  }

  bool hasFP16Denormals() const {
    return FP64FP16Denormals;
  }

  bool hasFP32Denormals() const {
    return FP32Denormals;
  }

  bool hasFP64Denormals() const {
    return FP64FP16Denormals;
  }

  bool supportsMinMaxDenormModes() const {
    return getGeneration() >= AMDGPUSubtarget::GFX9;
  }

  bool hasFPExceptions() const {
    return FPExceptions;
  }

  bool enableDX10Clamp() const {
    return DX10Clamp;
  }

  bool enableIEEEBit(const MachineFunction &MF) const {
    return AMDGPU::isCompute(MF.getFunction().getCallingConv());
  }

  bool useFlatForGlobal() const {
    return FlatForGlobal;
  }

  /// \returns If target supports ds_read/write_b128 and user enables generation
  /// of ds_read/write_b128.
  bool useDS128() const {
    return CIInsts && EnableDS128;
  }

  /// \returns If MUBUF instructions always perform range checking, even for
  /// buffer resources used for private memory access.
  bool privateMemoryResourceIsRangeChecked() const {
    return getGeneration() < AMDGPUSubtarget::GFX9;
  }

  bool hasAutoWaitcntBeforeBarrier() const {
    return AutoWaitcntBeforeBarrier;
  }

  bool hasCodeObjectV3() const {
    return CodeObjectV3;
  }

  bool hasUnalignedBufferAccess() const {
    return UnalignedBufferAccess;
  }

  bool hasUnalignedScratchAccess() const {
    return UnalignedScratchAccess;
  }

  bool hasApertureRegs() const {
   return HasApertureRegs;
  }

  bool isTrapHandlerEnabled() const {
    return TrapHandler;
  }

  bool isXNACKEnabled() const {
    return EnableXNACK;
  }

  bool hasFlatAddressSpace() const {
    return FlatAddressSpace;
  }

  bool hasFlatInstOffsets() const {
    return FlatInstOffsets;
  }

  bool hasFlatGlobalInsts() const {
    return FlatGlobalInsts;
  }

  bool hasFlatScratchInsts() const {
    return FlatScratchInsts;
  }

  bool hasD16LoadStore() const {
    return getGeneration() >= GFX9;
  }

  /// Return if most LDS instructions have an m0 use that require m0 to be
  /// iniitalized.
  bool ldsRequiresM0Init() const {
    return getGeneration() < GFX9;
  }

  bool hasAddNoCarry() const {
    return AddNoCarryInsts;
  }

  bool hasUnpackedD16VMem() const {
    return HasUnpackedD16VMem;
  }

  bool isMesaKernel(const MachineFunction &MF) const {
    return isMesa3DOS() && !AMDGPU::isShader(MF.getFunction().getCallingConv());
  }

  // Covers VS/PS/CS graphics shaders
  bool isMesaGfxShader(const MachineFunction &MF) const {
    return isMesa3DOS() && AMDGPU::isShader(MF.getFunction().getCallingConv());
  }

  bool isAmdCodeObjectV2(const MachineFunction &MF) const {
    return isAmdHsaOS() || isMesaKernel(MF);
  }

  bool hasMad64_32() const {
    return getGeneration() >= SEA_ISLANDS;
  }

  bool hasFminFmaxLegacy() const {
    return getGeneration() < AMDGPUSubtarget::VOLCANIC_ISLANDS;
  }

  bool hasSDWA() const {
    return HasSDWA;
  }

  bool hasSDWAOmod() const {
    return HasSDWAOmod;
  }

  bool hasSDWAScalar() const {
    return HasSDWAScalar;
  }

  bool hasSDWASdst() const {
    return HasSDWASdst;
  }

  bool hasSDWAMac() const {
    return HasSDWAMac;
  }

  bool hasSDWAOutModsVOPC() const {
    return HasSDWAOutModsVOPC;
  }

  bool vmemWriteNeedsExpWaitcnt() const {
    return getGeneration() < SEA_ISLANDS;
  }

  bool hasDLInsts() const {
    return HasDLInsts;
  }

  bool d16PreservesUnusedBits() const {
    return D16PreservesUnusedBits;
  }

  /// Returns the offset in bytes from the start of the input buffer
  ///        of the first explicit kernel argument.
  unsigned getExplicitKernelArgOffset(const MachineFunction &MF) const {
    return isAmdCodeObjectV2(MF) ? 0 : 36;
  }

  unsigned getAlignmentForImplicitArgPtr() const {
    return isAmdHsaOS() ? 8 : 4;
  }

  /// \returns Number of bytes of arguments that are passed to a shader or
  /// kernel in addition to the explicit ones declared for the function.
  unsigned getImplicitArgNumBytes(const MachineFunction &MF) const {
    if (isMesaKernel(MF))
      return 16;
    return AMDGPU::getIntegerAttribute(
      MF.getFunction(), "amdgpu-implicitarg-num-bytes", 0);
  }

  // Scratch is allocated in 256 dword per wave blocks for the entire
  // wavefront. When viewed from the perspecive of an arbitrary workitem, this
  // is 4-byte aligned.
  //
  // Only 4-byte alignment is really needed to access anything. Transformations
  // on the pointer value itself may rely on the alignment / known low bits of
  // the pointer. Set this to something above the minimum to avoid needing
  // dynamic realignment in common cases.
  unsigned getStackAlignment() const {
    return 16;
  }

  bool enableMachineScheduler() const override {
    return true;
  }

  bool enableSubRegLiveness() const override {
    return true;
  }

  void setScalarizeGlobalBehavior(bool b) { ScalarizeGlobal = b;}
  bool getScalarizeGlobalBehavior() const { return ScalarizeGlobal;}

  /// \returns Number of execution units per compute unit supported by the
  /// subtarget.
  unsigned getEUsPerCU() const {
    return AMDGPU::IsaInfo::getEUsPerCU(getFeatureBits());
  }

  /// \returns Maximum number of work groups per compute unit supported by the
  /// subtarget and limited by given \p FlatWorkGroupSize.
  unsigned getMaxWorkGroupsPerCU(unsigned FlatWorkGroupSize) const {
    return AMDGPU::IsaInfo::getMaxWorkGroupsPerCU(getFeatureBits(),
                                                  FlatWorkGroupSize);
  }

  /// \returns Maximum number of waves per compute unit supported by the
  /// subtarget without any kind of limitation.
  unsigned getMaxWavesPerCU() const {
    return AMDGPU::IsaInfo::getMaxWavesPerCU(getFeatureBits());
  }

  /// \returns Maximum number of waves per compute unit supported by the
  /// subtarget and limited by given \p FlatWorkGroupSize.
  unsigned getMaxWavesPerCU(unsigned FlatWorkGroupSize) const {
    return AMDGPU::IsaInfo::getMaxWavesPerCU(getFeatureBits(),
                                             FlatWorkGroupSize);
  }

  /// \returns Minimum number of waves per execution unit supported by the
  /// subtarget.
  unsigned getMinWavesPerEU() const {
    return AMDGPU::IsaInfo::getMinWavesPerEU(getFeatureBits());
  }

  /// \returns Maximum number of waves per execution unit supported by the
  /// subtarget without any kind of limitation.
  unsigned getMaxWavesPerEU() const {
    return AMDGPU::IsaInfo::getMaxWavesPerEU(getFeatureBits());
  }

  /// \returns Maximum number of waves per execution unit supported by the
  /// subtarget and limited by given \p FlatWorkGroupSize.
  unsigned getMaxWavesPerEU(unsigned FlatWorkGroupSize) const {
    return AMDGPU::IsaInfo::getMaxWavesPerEU(getFeatureBits(),
                                             FlatWorkGroupSize);
  }

  /// \returns Minimum flat work group size supported by the subtarget.
  unsigned getMinFlatWorkGroupSize() const {
    return AMDGPU::IsaInfo::getMinFlatWorkGroupSize(getFeatureBits());
  }

  /// \returns Maximum flat work group size supported by the subtarget.
  unsigned getMaxFlatWorkGroupSize() const {
    return AMDGPU::IsaInfo::getMaxFlatWorkGroupSize(getFeatureBits());
  }

  /// \returns Number of waves per work group supported by the subtarget and
  /// limited by given \p FlatWorkGroupSize.
  unsigned getWavesPerWorkGroup(unsigned FlatWorkGroupSize) const {
    return AMDGPU::IsaInfo::getWavesPerWorkGroup(getFeatureBits(),
                                                 FlatWorkGroupSize);
  }

  /// \returns Default range flat work group size for a calling convention.
  std::pair<unsigned, unsigned> getDefaultFlatWorkGroupSize(CallingConv::ID CC) const;

  /// \returns Subtarget's default pair of minimum/maximum flat work group sizes
  /// for function \p F, or minimum/maximum flat work group sizes explicitly
  /// requested using "amdgpu-flat-work-group-size" attribute attached to
  /// function \p F.
  ///
  /// \returns Subtarget's default values if explicitly requested values cannot
  /// be converted to integer, or violate subtarget's specifications.
  std::pair<unsigned, unsigned> getFlatWorkGroupSizes(const Function &F) const;

  /// \returns Subtarget's default pair of minimum/maximum number of waves per
  /// execution unit for function \p F, or minimum/maximum number of waves per
  /// execution unit explicitly requested using "amdgpu-waves-per-eu" attribute
  /// attached to function \p F.
  ///
  /// \returns Subtarget's default values if explicitly requested values cannot
  /// be converted to integer, violate subtarget's specifications, or are not
  /// compatible with minimum/maximum number of waves limited by flat work group
  /// size, register usage, and/or lds usage.
  std::pair<unsigned, unsigned> getWavesPerEU(const Function &F) const;

  /// Creates value range metadata on an workitemid.* inrinsic call or load.
  bool makeLIDRangeMetadata(Instruction *I) const;
};

class R600Subtarget final : public AMDGPUSubtarget {
private:
  R600InstrInfo InstrInfo;
  R600FrameLowering FrameLowering;
  R600TargetLowering TLInfo;

public:
  R600Subtarget(const Triple &TT, StringRef CPU, StringRef FS,
                const TargetMachine &TM);

  const R600InstrInfo *getInstrInfo() const override {
    return &InstrInfo;
  }

  const R600FrameLowering *getFrameLowering() const override {
    return &FrameLowering;
  }

  const R600TargetLowering *getTargetLowering() const override {
    return &TLInfo;
  }

  const R600RegisterInfo *getRegisterInfo() const override {
    return &InstrInfo.getRegisterInfo();
  }

  bool hasCFAluBug() const {
    return CFALUBug;
  }

  bool hasVertexCache() const {
    return HasVertexCache;
  }

  short getTexVTXClauseSize() const {
    return TexVTXClauseSize;
  }
};

class SISubtarget final : public AMDGPUSubtarget {
private:
  SIInstrInfo InstrInfo;
  SIFrameLowering FrameLowering;
  SITargetLowering TLInfo;

  /// GlobalISel related APIs.
  std::unique_ptr<AMDGPUCallLowering> CallLoweringInfo;
  std::unique_ptr<InstructionSelector> InstSelector;
  std::unique_ptr<LegalizerInfo> Legalizer;
  std::unique_ptr<RegisterBankInfo> RegBankInfo;

public:
  SISubtarget(const Triple &TT, StringRef CPU, StringRef FS,
              const GCNTargetMachine &TM);

  const SIInstrInfo *getInstrInfo() const override {
    return &InstrInfo;
  }

  const SIFrameLowering *getFrameLowering() const override {
    return &FrameLowering;
  }

  const SITargetLowering *getTargetLowering() const override {
    return &TLInfo;
  }

  const CallLowering *getCallLowering() const override {
    return CallLoweringInfo.get();
  }

  const InstructionSelector *getInstructionSelector() const override {
    return InstSelector.get();
  }

  const LegalizerInfo *getLegalizerInfo() const override {
    return Legalizer.get();
  }

  const RegisterBankInfo *getRegBankInfo() const override {
    return RegBankInfo.get();
  }

  const SIRegisterInfo *getRegisterInfo() const override {
    return &InstrInfo.getRegisterInfo();
  }

  // XXX - Why is this here if it isn't in the default pass set?
  bool enableEarlyIfConversion() const override {
    return true;
  }

  void overrideSchedPolicy(MachineSchedPolicy &Policy,
                           unsigned NumRegionInstrs) const override;

  bool isVGPRSpillingEnabled(const Function& F) const;

  unsigned getMaxNumUserSGPRs() const {
    return 16;
  }

  bool hasSMemRealTime() const {
    return HasSMemRealTime;
  }

  bool hasMovrel() const {
    return HasMovrel;
  }

  bool hasVGPRIndexMode() const {
    return HasVGPRIndexMode;
  }

  bool useVGPRIndexMode(bool UserEnable) const {
    return !hasMovrel() || (UserEnable && hasVGPRIndexMode());
  }

  bool hasScalarCompareEq64() const {
    return getGeneration() >= VOLCANIC_ISLANDS;
  }

  bool hasScalarStores() const {
    return HasScalarStores;
  }

  bool hasScalarAtomics() const {
    return HasScalarAtomics;
  }

  bool hasInv2PiInlineImm() const {
    return HasInv2PiInlineImm;
  }

  bool hasDPP() const {
    return HasDPP;
  }

  bool enableSIScheduler() const {
    return EnableSIScheduler;
  }

  bool debuggerSupported() const {
    return debuggerInsertNops() && debuggerReserveRegs() &&
      debuggerEmitPrologue();
  }

  bool debuggerInsertNops() const {
    return DebuggerInsertNops;
  }

  bool debuggerReserveRegs() const {
    return DebuggerReserveRegs;
  }

  bool debuggerEmitPrologue() const {
    return DebuggerEmitPrologue;
  }

  bool loadStoreOptEnabled() const {
    return EnableLoadStoreOpt;
  }

  bool hasSGPRInitBug() const {
    return SGPRInitBug;
  }

  bool has12DWordStoreHazard() const {
    return getGeneration() != AMDGPUSubtarget::SOUTHERN_ISLANDS;
  }

  bool hasSMovFedHazard() const {
    return getGeneration() >= AMDGPUSubtarget::GFX9;
  }

  bool hasReadM0MovRelInterpHazard() const {
    return getGeneration() >= AMDGPUSubtarget::GFX9;
  }

  bool hasReadM0SendMsgHazard() const {
    return getGeneration() >= AMDGPUSubtarget::VOLCANIC_ISLANDS;
  }

  unsigned getKernArgSegmentSize(const MachineFunction &MF,
                                 unsigned ExplictArgBytes) const;

  /// Return the maximum number of waves per SIMD for kernels using \p SGPRs SGPRs
  unsigned getOccupancyWithNumSGPRs(unsigned SGPRs) const;

  /// Return the maximum number of waves per SIMD for kernels using \p VGPRs VGPRs
  unsigned getOccupancyWithNumVGPRs(unsigned VGPRs) const;

  /// \returns true if the flat_scratch register should be initialized with the
  /// pointer to the wave's scratch memory rather than a size and offset.
  bool flatScratchIsPointer() const {
    return getGeneration() >= GFX9;
  }

  /// \returns true if the machine has merged shaders in which s0-s7 are
  /// reserved by the hardware and user SGPRs start at s8
  bool hasMergedShaders() const {
    return getGeneration() >= GFX9;
  }

  /// \returns SGPR allocation granularity supported by the subtarget.
  unsigned getSGPRAllocGranule() const {
    return AMDGPU::IsaInfo::getSGPRAllocGranule(getFeatureBits());
  }

  /// \returns SGPR encoding granularity supported by the subtarget.
  unsigned getSGPREncodingGranule() const {
    return AMDGPU::IsaInfo::getSGPREncodingGranule(getFeatureBits());
  }

  /// \returns Total number of SGPRs supported by the subtarget.
  unsigned getTotalNumSGPRs() const {
    return AMDGPU::IsaInfo::getTotalNumSGPRs(getFeatureBits());
  }

  /// \returns Addressable number of SGPRs supported by the subtarget.
  unsigned getAddressableNumSGPRs() const {
    return AMDGPU::IsaInfo::getAddressableNumSGPRs(getFeatureBits());
  }

  /// \returns Minimum number of SGPRs that meets the given number of waves per
  /// execution unit requirement supported by the subtarget.
  unsigned getMinNumSGPRs(unsigned WavesPerEU) const {
    return AMDGPU::IsaInfo::getMinNumSGPRs(getFeatureBits(), WavesPerEU);
  }

  /// \returns Maximum number of SGPRs that meets the given number of waves per
  /// execution unit requirement supported by the subtarget.
  unsigned getMaxNumSGPRs(unsigned WavesPerEU, bool Addressable) const {
    return AMDGPU::IsaInfo::getMaxNumSGPRs(getFeatureBits(), WavesPerEU,
                                           Addressable);
  }

  /// \returns Reserved number of SGPRs for given function \p MF.
  unsigned getReservedNumSGPRs(const MachineFunction &MF) const;

  /// \returns Maximum number of SGPRs that meets number of waves per execution
  /// unit requirement for function \p MF, or number of SGPRs explicitly
  /// requested using "amdgpu-num-sgpr" attribute attached to function \p MF.
  ///
  /// \returns Value that meets number of waves per execution unit requirement
  /// if explicitly requested value cannot be converted to integer, violates
  /// subtarget's specifications, or does not meet number of waves per execution
  /// unit requirement.
  unsigned getMaxNumSGPRs(const MachineFunction &MF) const;

  /// \returns VGPR allocation granularity supported by the subtarget.
  unsigned getVGPRAllocGranule() const {
    return AMDGPU::IsaInfo::getVGPRAllocGranule(getFeatureBits());
  }

  /// \returns VGPR encoding granularity supported by the subtarget.
  unsigned getVGPREncodingGranule() const {
    return AMDGPU::IsaInfo::getVGPREncodingGranule(getFeatureBits());
  }

  /// \returns Total number of VGPRs supported by the subtarget.
  unsigned getTotalNumVGPRs() const {
    return AMDGPU::IsaInfo::getTotalNumVGPRs(getFeatureBits());
  }

  /// \returns Addressable number of VGPRs supported by the subtarget.
  unsigned getAddressableNumVGPRs() const {
    return AMDGPU::IsaInfo::getAddressableNumVGPRs(getFeatureBits());
  }

  /// \returns Minimum number of VGPRs that meets given number of waves per
  /// execution unit requirement supported by the subtarget.
  unsigned getMinNumVGPRs(unsigned WavesPerEU) const {
    return AMDGPU::IsaInfo::getMinNumVGPRs(getFeatureBits(), WavesPerEU);
  }

  /// \returns Maximum number of VGPRs that meets given number of waves per
  /// execution unit requirement supported by the subtarget.
  unsigned getMaxNumVGPRs(unsigned WavesPerEU) const {
    return AMDGPU::IsaInfo::getMaxNumVGPRs(getFeatureBits(), WavesPerEU);
  }

  /// \returns Reserved number of VGPRs for given function \p MF.
  unsigned getReservedNumVGPRs(const MachineFunction &MF) const {
    return debuggerReserveRegs() ? 4 : 0;
  }

  /// \returns Maximum number of VGPRs that meets number of waves per execution
  /// unit requirement for function \p MF, or number of VGPRs explicitly
  /// requested using "amdgpu-num-vgpr" attribute attached to function \p MF.
  ///
  /// \returns Value that meets number of waves per execution unit requirement
  /// if explicitly requested value cannot be converted to integer, violates
  /// subtarget's specifications, or does not meet number of waves per execution
  /// unit requirement.
  unsigned getMaxNumVGPRs(const MachineFunction &MF) const;

  void getPostRAMutations(
      std::vector<std::unique_ptr<ScheduleDAGMutation>> &Mutations)
      const override;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUSUBTARGET_H
