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
/// \brief AMDGPU specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUSUBTARGET_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUSUBTARGET_H

#include "AMDGPU.h"
#include "R600InstrInfo.h"
#include "R600ISelLowering.h"
#include "R600FrameLowering.h"
#include "SIInstrInfo.h"
#include "SIISelLowering.h"
#include "SIFrameLowering.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/CodeGen/GlobalISel/GISelAccessor.h"
#include "llvm/CodeGen/SelectionDAGTargetInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"

#define GET_SUBTARGETINFO_HEADER
#include "AMDGPUGenSubtargetInfo.inc"

namespace llvm {

class SIMachineFunctionInfo;
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
  };

  enum {
    ISAVersion0_0_0,
    ISAVersion7_0_0,
    ISAVersion7_0_1,
    ISAVersion8_0_0,
    ISAVersion8_0_1,
    ISAVersion8_0_3
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
  bool FP64Denormals;
  bool FPExceptions;
  bool FlatForGlobal;
  bool UnalignedBufferAccess;
  bool EnableXNACK;
  bool DebuggerInsertNops;
  bool DebuggerReserveRegs;
  bool DebuggerEmitPrologue;

  // Used as options.
  bool EnableVGPRSpilling;
  bool EnablePromoteAlloca;
  bool EnableLoadStoreOpt;
  bool EnableUnsafeDSOffsetFolding;
  bool EnableSIScheduler;
  bool DumpCode;

  // Subtarget statically properties set by tablegen
  bool FP64;
  bool IsGCN;
  bool GCN1Encoding;
  bool GCN3Encoding;
  bool CIInsts;
  bool SGPRInitBug;
  bool HasSMemRealTime;
  bool Has16BitInsts;
  bool FlatAddressSpace;
  bool R600ALUInst;
  bool CaymanISA;
  bool CFALUBug;
  bool HasVertexCache;
  short TexVTXClauseSize;

  // Dummy feature to use for assembler in tablegen.
  bool FeatureDisable;

  InstrItineraryData InstrItins;
  SelectionDAGTargetInfo TSInfo;

public:
  AMDGPUSubtarget(const Triple &TT, StringRef GPU, StringRef FS,
                  const TargetMachine &TM);
  virtual ~AMDGPUSubtarget();
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

  Generation getGeneration() const {
    return Gen;
  }

  unsigned getWavefrontSize() const {
    return WavefrontSize;
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

  bool hasHWFP64() const {
    return FP64;
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

  bool hasCARRY() const {
    return (getGeneration() >= EVERGREEN);
  }

  bool hasBORROW() const {
    return (getGeneration() >= EVERGREEN);
  }

  bool hasCaymanISA() const {
    return CaymanISA;
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
  unsigned getMaxLocalMemSizeWithWaveCount(unsigned WaveCount) const;

  /// Inverse of getMaxLocalMemWithWaveCount. Return the maximum wavecount if
  /// the given LDS memory size is the only constraint.
  unsigned getOccupancyWithLocalMemSize(uint32_t Bytes) const;


  bool hasFP32Denormals() const {
    return FP32Denormals;
  }

  bool hasFP64Denormals() const {
    return FP64Denormals;
  }

  bool hasFPExceptions() const {
    return FPExceptions;
  }

  bool useFlatForGlobal() const {
    return FlatForGlobal;
  }

  bool hasUnalignedBufferAccess() const {
    return UnalignedBufferAccess;
  }

  bool isXNACKEnabled() const {
    return EnableXNACK;
  }

  bool isAmdCodeObjectV2() const {
    return isAmdHsaOS() || isMesa3DOS();
  }

  /// \brief Returns the offset in bytes from the start of the input buffer
  ///        of the first explicit kernel argument.
  unsigned getExplicitKernelArgOffset() const {
    return isAmdCodeObjectV2() ? 0 : 36;
  }

  unsigned getAlignmentForImplicitArgPtr() const {
    return isAmdHsaOS() ? 8 : 4;
  }

  unsigned getStackAlignment() const {
    // Scratch is allocated in 256 dword per wave blocks.
    return 4 * 256 / getWavefrontSize();
  }

  bool enableMachineScheduler() const override {
    return true;
  }

  bool enableSubRegLiveness() const override {
    return true;
  }

  /// \returns Number of execution units per compute unit supported by the
  /// subtarget.
  unsigned getEUsPerCU() const {
    return 4;
  }

  /// \returns Maximum number of work groups per compute unit supported by the
  /// subtarget and limited by given flat work group size.
  unsigned getMaxWorkGroupsPerCU(unsigned FlatWorkGroupSize) const {
    if (getGeneration() < AMDGPUSubtarget::SOUTHERN_ISLANDS)
      return 8;
    return getWavesPerWorkGroup(FlatWorkGroupSize) == 1 ? 40 : 16;
  }

  /// \returns Maximum number of waves per compute unit supported by the
  /// subtarget without any kind of limitation.
  unsigned getMaxWavesPerCU() const {
    return getMaxWavesPerEU() * getEUsPerCU();
  }

  /// \returns Maximum number of waves per compute unit supported by the
  /// subtarget and limited by given flat work group size.
  unsigned getMaxWavesPerCU(unsigned FlatWorkGroupSize) const {
    return getWavesPerWorkGroup(FlatWorkGroupSize);
  }

  /// \returns Minimum number of waves per execution unit supported by the
  /// subtarget.
  unsigned getMinWavesPerEU() const {
    return 1;
  }

  /// \returns Maximum number of waves per execution unit supported by the
  /// subtarget without any kind of limitation.
  unsigned getMaxWavesPerEU() const {
    if (getGeneration() < AMDGPUSubtarget::SOUTHERN_ISLANDS)
      return 8;
    // FIXME: Need to take scratch memory into account.
    return 10;
  }

  /// \returns Maximum number of waves per execution unit supported by the
  /// subtarget and limited by given flat work group size.
  unsigned getMaxWavesPerEU(unsigned FlatWorkGroupSize) const {
    return alignTo(getMaxWavesPerCU(FlatWorkGroupSize), getEUsPerCU()) /
      getEUsPerCU();
  }

  /// \returns Minimum flat work group size supported by the subtarget.
  unsigned getMinFlatWorkGroupSize() const {
    return 1;
  }

  /// \returns Maximum flat work group size supported by the subtarget.
  unsigned getMaxFlatWorkGroupSize() const {
    return 2048;
  }

  /// \returns Number of waves per work group given the flat work group size.
  unsigned getWavesPerWorkGroup(unsigned FlatWorkGroupSize) const {
    return alignTo(FlatWorkGroupSize, getWavefrontSize()) / getWavefrontSize();
  }

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
public:
  enum {
    // The closed Vulkan driver sets 96, which limits the wave count to 8 but
    // doesn't spill SGPRs as much as when 80 is set.
    FIXED_SGPR_COUNT_FOR_INIT_BUG = 96
  };

private:
  SIInstrInfo InstrInfo;
  SIFrameLowering FrameLowering;
  SITargetLowering TLInfo;
  std::unique_ptr<GISelAccessor> GISel;

public:
  SISubtarget(const Triple &TT, StringRef CPU, StringRef FS,
              const TargetMachine &TM);

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
    assert(GISel && "Access to GlobalISel APIs not set");
    return GISel->getCallLowering();
  }

  const SIRegisterInfo *getRegisterInfo() const override {
    return &InstrInfo.getRegisterInfo();
  }

  void setGISelAccessor(GISelAccessor &GISel) {
    this->GISel.reset(&GISel);
  }

  void overrideSchedPolicy(MachineSchedPolicy &Policy,
                           unsigned NumRegionInstrs) const override;

  bool isVGPRSpillingEnabled(const Function& F) const;

  unsigned getMaxNumUserSGPRs() const {
    return 16;
  }

  bool hasFlatAddressSpace() const {
    return FlatAddressSpace;
  }

  bool hasSMemRealTime() const {
    return HasSMemRealTime;
  }

  bool has16BitInsts() const {
    return Has16BitInsts;
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

  /// Return the maximum number of waves per SIMD for kernels using \p SGPRs SGPRs
  unsigned getOccupancyWithNumSGPRs(unsigned SGPRs) const;

  /// Return the maximum number of waves per SIMD for kernels using \p VGPRs VGPRs
  unsigned getOccupancyWithNumVGPRs(unsigned VGPRs) const;
};

} // End namespace llvm

#endif
