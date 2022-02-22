//==- SIMachineFunctionInfo.h - SIMachineFunctionInfo interface --*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_SIMACHINEFUNCTIONINFO_H
#define LLVM_LIB_TARGET_AMDGPU_SIMACHINEFUNCTIONINFO_H

#include "AMDGPUArgumentUsageInfo.h"
#include "AMDGPUMachineFunction.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/CodeGen/MIRYamlMapping.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

class MachineFrameInfo;
class MachineFunction;
class SIMachineFunctionInfo;
class SIRegisterInfo;
class TargetRegisterClass;

class AMDGPUPseudoSourceValue : public PseudoSourceValue {
public:
  enum AMDGPUPSVKind : unsigned {
    PSVBuffer = PseudoSourceValue::TargetCustom,
    PSVImage,
    GWSResource
  };

protected:
  AMDGPUPseudoSourceValue(unsigned Kind, const TargetInstrInfo &TII)
      : PseudoSourceValue(Kind, TII) {}

public:
  bool isConstant(const MachineFrameInfo *) const override {
    // This should probably be true for most images, but we will start by being
    // conservative.
    return false;
  }

  bool isAliased(const MachineFrameInfo *) const override {
    return true;
  }

  bool mayAlias(const MachineFrameInfo *) const override {
    return true;
  }
};

class AMDGPUBufferPseudoSourceValue final : public AMDGPUPseudoSourceValue {
public:
  explicit AMDGPUBufferPseudoSourceValue(const TargetInstrInfo &TII)
      : AMDGPUPseudoSourceValue(PSVBuffer, TII) {}

  static bool classof(const PseudoSourceValue *V) {
    return V->kind() == PSVBuffer;
  }

  void printCustom(raw_ostream &OS) const override { OS << "BufferResource"; }
};

class AMDGPUImagePseudoSourceValue final : public AMDGPUPseudoSourceValue {
public:
  // TODO: Is the img rsrc useful?
  explicit AMDGPUImagePseudoSourceValue(const TargetInstrInfo &TII)
      : AMDGPUPseudoSourceValue(PSVImage, TII) {}

  static bool classof(const PseudoSourceValue *V) {
    return V->kind() == PSVImage;
  }

  void printCustom(raw_ostream &OS) const override { OS << "ImageResource"; }
};

class AMDGPUGWSResourcePseudoSourceValue final : public AMDGPUPseudoSourceValue {
public:
  explicit AMDGPUGWSResourcePseudoSourceValue(const TargetInstrInfo &TII)
      : AMDGPUPseudoSourceValue(GWSResource, TII) {}

  static bool classof(const PseudoSourceValue *V) {
    return V->kind() == GWSResource;
  }

  // These are inaccessible memory from IR.
  bool isAliased(const MachineFrameInfo *) const override {
    return false;
  }

  // These are inaccessible memory from IR.
  bool mayAlias(const MachineFrameInfo *) const override {
    return false;
  }

  void printCustom(raw_ostream &OS) const override {
    OS << "GWSResource";
  }
};

namespace yaml {

struct SIArgument {
  bool IsRegister;
  union {
    StringValue RegisterName;
    unsigned StackOffset;
  };
  Optional<unsigned> Mask;

  // Default constructor, which creates a stack argument.
  SIArgument() : IsRegister(false), StackOffset(0) {}
  SIArgument(const SIArgument &Other) {
    IsRegister = Other.IsRegister;
    if (IsRegister) {
      ::new ((void *)std::addressof(RegisterName))
          StringValue(Other.RegisterName);
    } else
      StackOffset = Other.StackOffset;
    Mask = Other.Mask;
  }
  SIArgument &operator=(const SIArgument &Other) {
    IsRegister = Other.IsRegister;
    if (IsRegister) {
      ::new ((void *)std::addressof(RegisterName))
          StringValue(Other.RegisterName);
    } else
      StackOffset = Other.StackOffset;
    Mask = Other.Mask;
    return *this;
  }
  ~SIArgument() {
    if (IsRegister)
      RegisterName.~StringValue();
  }

  // Helper to create a register or stack argument.
  static inline SIArgument createArgument(bool IsReg) {
    if (IsReg)
      return SIArgument(IsReg);
    return SIArgument();
  }

private:
  // Construct a register argument.
  SIArgument(bool) : IsRegister(true), RegisterName() {}
};

template <> struct MappingTraits<SIArgument> {
  static void mapping(IO &YamlIO, SIArgument &A) {
    if (YamlIO.outputting()) {
      if (A.IsRegister)
        YamlIO.mapRequired("reg", A.RegisterName);
      else
        YamlIO.mapRequired("offset", A.StackOffset);
    } else {
      auto Keys = YamlIO.keys();
      if (is_contained(Keys, "reg")) {
        A = SIArgument::createArgument(true);
        YamlIO.mapRequired("reg", A.RegisterName);
      } else if (is_contained(Keys, "offset"))
        YamlIO.mapRequired("offset", A.StackOffset);
      else
        YamlIO.setError("missing required key 'reg' or 'offset'");
    }
    YamlIO.mapOptional("mask", A.Mask);
  }
  static const bool flow = true;
};

struct SIArgumentInfo {
  Optional<SIArgument> PrivateSegmentBuffer;
  Optional<SIArgument> DispatchPtr;
  Optional<SIArgument> QueuePtr;
  Optional<SIArgument> KernargSegmentPtr;
  Optional<SIArgument> DispatchID;
  Optional<SIArgument> FlatScratchInit;
  Optional<SIArgument> PrivateSegmentSize;

  Optional<SIArgument> WorkGroupIDX;
  Optional<SIArgument> WorkGroupIDY;
  Optional<SIArgument> WorkGroupIDZ;
  Optional<SIArgument> WorkGroupInfo;
  Optional<SIArgument> PrivateSegmentWaveByteOffset;

  Optional<SIArgument> ImplicitArgPtr;
  Optional<SIArgument> ImplicitBufferPtr;

  Optional<SIArgument> WorkItemIDX;
  Optional<SIArgument> WorkItemIDY;
  Optional<SIArgument> WorkItemIDZ;
};

template <> struct MappingTraits<SIArgumentInfo> {
  static void mapping(IO &YamlIO, SIArgumentInfo &AI) {
    YamlIO.mapOptional("privateSegmentBuffer", AI.PrivateSegmentBuffer);
    YamlIO.mapOptional("dispatchPtr", AI.DispatchPtr);
    YamlIO.mapOptional("queuePtr", AI.QueuePtr);
    YamlIO.mapOptional("kernargSegmentPtr", AI.KernargSegmentPtr);
    YamlIO.mapOptional("dispatchID", AI.DispatchID);
    YamlIO.mapOptional("flatScratchInit", AI.FlatScratchInit);
    YamlIO.mapOptional("privateSegmentSize", AI.PrivateSegmentSize);

    YamlIO.mapOptional("workGroupIDX", AI.WorkGroupIDX);
    YamlIO.mapOptional("workGroupIDY", AI.WorkGroupIDY);
    YamlIO.mapOptional("workGroupIDZ", AI.WorkGroupIDZ);
    YamlIO.mapOptional("workGroupInfo", AI.WorkGroupInfo);
    YamlIO.mapOptional("privateSegmentWaveByteOffset",
                       AI.PrivateSegmentWaveByteOffset);

    YamlIO.mapOptional("implicitArgPtr", AI.ImplicitArgPtr);
    YamlIO.mapOptional("implicitBufferPtr", AI.ImplicitBufferPtr);

    YamlIO.mapOptional("workItemIDX", AI.WorkItemIDX);
    YamlIO.mapOptional("workItemIDY", AI.WorkItemIDY);
    YamlIO.mapOptional("workItemIDZ", AI.WorkItemIDZ);
  }
};

// Default to default mode for default calling convention.
struct SIMode {
  bool IEEE = true;
  bool DX10Clamp = true;
  bool FP32InputDenormals = true;
  bool FP32OutputDenormals = true;
  bool FP64FP16InputDenormals = true;
  bool FP64FP16OutputDenormals = true;

  SIMode() = default;

  SIMode(const AMDGPU::SIModeRegisterDefaults &Mode) {
    IEEE = Mode.IEEE;
    DX10Clamp = Mode.DX10Clamp;
    FP32InputDenormals = Mode.FP32InputDenormals;
    FP32OutputDenormals = Mode.FP32OutputDenormals;
    FP64FP16InputDenormals = Mode.FP64FP16InputDenormals;
    FP64FP16OutputDenormals = Mode.FP64FP16OutputDenormals;
  }

  bool operator ==(const SIMode Other) const {
    return IEEE == Other.IEEE &&
           DX10Clamp == Other.DX10Clamp &&
           FP32InputDenormals == Other.FP32InputDenormals &&
           FP32OutputDenormals == Other.FP32OutputDenormals &&
           FP64FP16InputDenormals == Other.FP64FP16InputDenormals &&
           FP64FP16OutputDenormals == Other.FP64FP16OutputDenormals;
  }
};

template <> struct MappingTraits<SIMode> {
  static void mapping(IO &YamlIO, SIMode &Mode) {
    YamlIO.mapOptional("ieee", Mode.IEEE, true);
    YamlIO.mapOptional("dx10-clamp", Mode.DX10Clamp, true);
    YamlIO.mapOptional("fp32-input-denormals", Mode.FP32InputDenormals, true);
    YamlIO.mapOptional("fp32-output-denormals", Mode.FP32OutputDenormals, true);
    YamlIO.mapOptional("fp64-fp16-input-denormals", Mode.FP64FP16InputDenormals, true);
    YamlIO.mapOptional("fp64-fp16-output-denormals", Mode.FP64FP16OutputDenormals, true);
  }
};

struct SIMachineFunctionInfo final : public yaml::MachineFunctionInfo {
  uint64_t ExplicitKernArgSize = 0;
  unsigned MaxKernArgAlign = 0;
  unsigned LDSSize = 0;
  Align DynLDSAlign;
  bool IsEntryFunction = false;
  bool NoSignedZerosFPMath = false;
  bool MemoryBound = false;
  bool WaveLimiter = false;
  bool HasSpilledSGPRs = false;
  bool HasSpilledVGPRs = false;
  uint32_t HighBitsOf32BitAddress = 0;

  // TODO: 10 may be a better default since it's the maximum.
  unsigned Occupancy = 0;

  StringValue ScratchRSrcReg = "$private_rsrc_reg";
  StringValue FrameOffsetReg = "$fp_reg";
  StringValue StackPtrOffsetReg = "$sp_reg";

  Optional<SIArgumentInfo> ArgInfo;
  SIMode Mode;
  Optional<FrameIndex> ScavengeFI;

  SIMachineFunctionInfo() = default;
  SIMachineFunctionInfo(const llvm::SIMachineFunctionInfo &,
                        const TargetRegisterInfo &TRI,
                        const llvm::MachineFunction &MF);

  void mappingImpl(yaml::IO &YamlIO) override;
  ~SIMachineFunctionInfo() = default;
};

template <> struct MappingTraits<SIMachineFunctionInfo> {
  static void mapping(IO &YamlIO, SIMachineFunctionInfo &MFI) {
    YamlIO.mapOptional("explicitKernArgSize", MFI.ExplicitKernArgSize,
                       UINT64_C(0));
    YamlIO.mapOptional("maxKernArgAlign", MFI.MaxKernArgAlign, 0u);
    YamlIO.mapOptional("ldsSize", MFI.LDSSize, 0u);
    YamlIO.mapOptional("dynLDSAlign", MFI.DynLDSAlign, Align());
    YamlIO.mapOptional("isEntryFunction", MFI.IsEntryFunction, false);
    YamlIO.mapOptional("noSignedZerosFPMath", MFI.NoSignedZerosFPMath, false);
    YamlIO.mapOptional("memoryBound", MFI.MemoryBound, false);
    YamlIO.mapOptional("waveLimiter", MFI.WaveLimiter, false);
    YamlIO.mapOptional("hasSpilledSGPRs", MFI.HasSpilledSGPRs, false);
    YamlIO.mapOptional("hasSpilledVGPRs", MFI.HasSpilledVGPRs, false);
    YamlIO.mapOptional("scratchRSrcReg", MFI.ScratchRSrcReg,
                       StringValue("$private_rsrc_reg"));
    YamlIO.mapOptional("frameOffsetReg", MFI.FrameOffsetReg,
                       StringValue("$fp_reg"));
    YamlIO.mapOptional("stackPtrOffsetReg", MFI.StackPtrOffsetReg,
                       StringValue("$sp_reg"));
    YamlIO.mapOptional("argumentInfo", MFI.ArgInfo);
    YamlIO.mapOptional("mode", MFI.Mode, SIMode());
    YamlIO.mapOptional("highBitsOf32BitAddress",
                       MFI.HighBitsOf32BitAddress, 0u);
    YamlIO.mapOptional("occupancy", MFI.Occupancy, 0);
    YamlIO.mapOptional("scavengeFI", MFI.ScavengeFI);
  }
};

} // end namespace yaml

/// This class keeps track of the SPI_SP_INPUT_ADDR config register, which
/// tells the hardware which interpolation parameters to load.
class SIMachineFunctionInfo final : public AMDGPUMachineFunction {
  friend class GCNTargetMachine;

  Register TIDReg = AMDGPU::NoRegister;

  // Registers that may be reserved for spilling purposes. These may be the same
  // as the input registers.
  Register ScratchRSrcReg = AMDGPU::PRIVATE_RSRC_REG;

  // This is the the unswizzled offset from the current dispatch's scratch wave
  // base to the beginning of the current function's frame.
  Register FrameOffsetReg = AMDGPU::FP_REG;

  // This is an ABI register used in the non-entry calling convention to
  // communicate the unswizzled offset from the current dispatch's scratch wave
  // base to the beginning of the new function's frame.
  Register StackPtrOffsetReg = AMDGPU::SP_REG;

  AMDGPUFunctionArgInfo ArgInfo;

  // Graphics info.
  unsigned PSInputAddr = 0;
  unsigned PSInputEnable = 0;

  /// Number of bytes of arguments this function has on the stack. If the callee
  /// is expected to restore the argument stack this should be a multiple of 16,
  /// all usable during a tail call.
  ///
  /// The alternative would forbid tail call optimisation in some cases: if we
  /// want to transfer control from a function with 8-bytes of stack-argument
  /// space to a function with 16-bytes then misalignment of this value would
  /// make a stack adjustment necessary, which could not be undone by the
  /// callee.
  unsigned BytesInStackArgArea = 0;

  bool ReturnsVoid = true;

  // A pair of default/requested minimum/maximum flat work group sizes.
  // Minimum - first, maximum - second.
  std::pair<unsigned, unsigned> FlatWorkGroupSizes = {0, 0};

  // A pair of default/requested minimum/maximum number of waves per execution
  // unit. Minimum - first, maximum - second.
  std::pair<unsigned, unsigned> WavesPerEU = {0, 0};

  std::unique_ptr<const AMDGPUBufferPseudoSourceValue> BufferPSV;
  std::unique_ptr<const AMDGPUImagePseudoSourceValue> ImagePSV;
  std::unique_ptr<const AMDGPUGWSResourcePseudoSourceValue> GWSResourcePSV;

private:
  unsigned LDSWaveSpillSize = 0;
  unsigned NumUserSGPRs = 0;
  unsigned NumSystemSGPRs = 0;

  bool HasSpilledSGPRs = false;
  bool HasSpilledVGPRs = false;
  bool HasNonSpillStackObjects = false;
  bool IsStackRealigned = false;

  unsigned NumSpilledSGPRs = 0;
  unsigned NumSpilledVGPRs = 0;

  // Feature bits required for inputs passed in user SGPRs.
  bool PrivateSegmentBuffer : 1;
  bool DispatchPtr : 1;
  bool QueuePtr : 1;
  bool KernargSegmentPtr : 1;
  bool DispatchID : 1;
  bool FlatScratchInit : 1;

  // Feature bits required for inputs passed in system SGPRs.
  bool WorkGroupIDX : 1; // Always initialized.
  bool WorkGroupIDY : 1;
  bool WorkGroupIDZ : 1;
  bool WorkGroupInfo : 1;
  bool PrivateSegmentWaveByteOffset : 1;

  bool WorkItemIDX : 1; // Always initialized.
  bool WorkItemIDY : 1;
  bool WorkItemIDZ : 1;

  // Private memory buffer
  // Compute directly in sgpr[0:1]
  // Other shaders indirect 64-bits at sgpr[0:1]
  bool ImplicitBufferPtr : 1;

  // Pointer to where the ABI inserts special kernel arguments separate from the
  // user arguments. This is an offset from the KernargSegmentPtr.
  bool ImplicitArgPtr : 1;
  bool HostcallPtr : 1;

  bool MayNeedAGPRs : 1;

  // The hard-wired high half of the address of the global information table
  // for AMDPAL OS type. 0xffffffff represents no hard-wired high half, since
  // current hardware only allows a 16 bit value.
  unsigned GITPtrHigh;

  unsigned HighBitsOf32BitAddress;
  unsigned GDSSize;

  // Current recorded maximum possible occupancy.
  unsigned Occupancy;

  mutable Optional<bool> UsesAGPRs;

  MCPhysReg getNextUserSGPR() const;

  MCPhysReg getNextSystemSGPR() const;

public:
  struct SpilledReg {
    Register VGPR;
    int Lane = -1;

    SpilledReg() = default;
    SpilledReg(Register R, int L) : VGPR (R), Lane (L) {}

    bool hasLane() { return Lane != -1;}
    bool hasReg() { return VGPR != 0;}
  };

  struct SGPRSpillVGPR {
    // VGPR used for SGPR spills
    Register VGPR;

    // If the VGPR is is used for SGPR spills in a non-entrypoint function, the
    // stack slot used to save/restore it in the prolog/epilog.
    Optional<int> FI;

    SGPRSpillVGPR(Register V, Optional<int> F) : VGPR(V), FI(F) {}
  };

  struct VGPRSpillToAGPR {
    SmallVector<MCPhysReg, 32> Lanes;
    bool FullyAllocated = false;
    bool IsDead = false;
  };

  // Map WWM VGPR to a stack slot that is used to save/restore it in the
  // prolog/epilog.
  MapVector<Register, Optional<int>> WWMReservedRegs;

private:
  // Track VGPR + wave index for each subregister of the SGPR spilled to
  // frameindex key.
  DenseMap<int, std::vector<SpilledReg>> SGPRToVGPRSpills;
  unsigned NumVGPRSpillLanes = 0;
  SmallVector<SGPRSpillVGPR, 2> SpillVGPRs;

  DenseMap<int, VGPRSpillToAGPR> VGPRToAGPRSpills;

  // AGPRs used for VGPR spills.
  SmallVector<MCPhysReg, 32> SpillAGPR;

  // VGPRs used for AGPR spills.
  SmallVector<MCPhysReg, 32> SpillVGPR;

  // Emergency stack slot. Sometimes, we create this before finalizing the stack
  // frame, so save it here and add it to the RegScavenger later.
  Optional<int> ScavengeFI;

public: // FIXME
  /// If this is set, an SGPR used for save/restore of the register used for the
  /// frame pointer.
  Register SGPRForFPSaveRestoreCopy;
  Optional<int> FramePointerSaveIndex;

  /// If this is set, an SGPR used for save/restore of the register used for the
  /// base pointer.
  Register SGPRForBPSaveRestoreCopy;
  Optional<int> BasePointerSaveIndex;

  bool isCalleeSavedReg(const MCPhysReg *CSRegs, MCPhysReg Reg);

public:
  SIMachineFunctionInfo(const MachineFunction &MF);

  bool initializeBaseYamlFields(const yaml::SIMachineFunctionInfo &YamlMFI,
                                const MachineFunction &MF,
                                PerFunctionMIParsingState &PFS,
                                SMDiagnostic &Error, SMRange &SourceRange);

  void reserveWWMRegister(Register Reg, Optional<int> FI) {
    WWMReservedRegs.insert(std::make_pair(Reg, FI));
  }

  ArrayRef<SpilledReg> getSGPRToVGPRSpills(int FrameIndex) const {
    auto I = SGPRToVGPRSpills.find(FrameIndex);
    return (I == SGPRToVGPRSpills.end()) ?
      ArrayRef<SpilledReg>() : makeArrayRef(I->second);
  }

  ArrayRef<SGPRSpillVGPR> getSGPRSpillVGPRs() const { return SpillVGPRs; }

  void setSGPRSpillVGPRs(Register NewVGPR, Optional<int> newFI, int Index) {
    SpillVGPRs[Index].VGPR = NewVGPR;
    SpillVGPRs[Index].FI = newFI;
  }

  bool removeVGPRForSGPRSpill(Register ReservedVGPR, MachineFunction &MF);

  ArrayRef<MCPhysReg> getAGPRSpillVGPRs() const {
    return SpillAGPR;
  }

  ArrayRef<MCPhysReg> getVGPRSpillAGPRs() const {
    return SpillVGPR;
  }

  MCPhysReg getVGPRToAGPRSpill(int FrameIndex, unsigned Lane) const {
    auto I = VGPRToAGPRSpills.find(FrameIndex);
    return (I == VGPRToAGPRSpills.end()) ? (MCPhysReg)AMDGPU::NoRegister
                                         : I->second.Lanes[Lane];
  }

  void setVGPRToAGPRSpillDead(int FrameIndex) {
    auto I = VGPRToAGPRSpills.find(FrameIndex);
    if (I != VGPRToAGPRSpills.end())
      I->second.IsDead = true;
  }

  bool haveFreeLanesForSGPRSpill(const MachineFunction &MF,
                                 unsigned NumLane) const;
  bool allocateSGPRSpillToVGPR(MachineFunction &MF, int FI);
  bool allocateVGPRSpillToAGPR(MachineFunction &MF, int FI, bool isAGPRtoVGPR);

  /// If \p ResetSGPRSpillStackIDs is true, reset the stack ID from sgpr-spill
  /// to the default stack.
  bool removeDeadFrameIndices(MachineFrameInfo &MFI,
                              bool ResetSGPRSpillStackIDs);

  int getScavengeFI(MachineFrameInfo &MFI, const SIRegisterInfo &TRI);
  Optional<int> getOptionalScavengeFI() const { return ScavengeFI; }

  bool hasCalculatedTID() const { return TIDReg != 0; };
  Register getTIDReg() const { return TIDReg; };
  void setTIDReg(Register Reg) { TIDReg = Reg; }

  unsigned getBytesInStackArgArea() const {
    return BytesInStackArgArea;
  }

  void setBytesInStackArgArea(unsigned Bytes) {
    BytesInStackArgArea = Bytes;
  }

  // Add user SGPRs.
  Register addPrivateSegmentBuffer(const SIRegisterInfo &TRI);
  Register addDispatchPtr(const SIRegisterInfo &TRI);
  Register addQueuePtr(const SIRegisterInfo &TRI);
  Register addKernargSegmentPtr(const SIRegisterInfo &TRI);
  Register addDispatchID(const SIRegisterInfo &TRI);
  Register addFlatScratchInit(const SIRegisterInfo &TRI);
  Register addImplicitBufferPtr(const SIRegisterInfo &TRI);

  // Add system SGPRs.
  Register addWorkGroupIDX() {
    ArgInfo.WorkGroupIDX = ArgDescriptor::createRegister(getNextSystemSGPR());
    NumSystemSGPRs += 1;
    return ArgInfo.WorkGroupIDX.getRegister();
  }

  Register addWorkGroupIDY() {
    ArgInfo.WorkGroupIDY = ArgDescriptor::createRegister(getNextSystemSGPR());
    NumSystemSGPRs += 1;
    return ArgInfo.WorkGroupIDY.getRegister();
  }

  Register addWorkGroupIDZ() {
    ArgInfo.WorkGroupIDZ = ArgDescriptor::createRegister(getNextSystemSGPR());
    NumSystemSGPRs += 1;
    return ArgInfo.WorkGroupIDZ.getRegister();
  }

  Register addWorkGroupInfo() {
    ArgInfo.WorkGroupInfo = ArgDescriptor::createRegister(getNextSystemSGPR());
    NumSystemSGPRs += 1;
    return ArgInfo.WorkGroupInfo.getRegister();
  }

  // Add special VGPR inputs
  void setWorkItemIDX(ArgDescriptor Arg) {
    ArgInfo.WorkItemIDX = Arg;
  }

  void setWorkItemIDY(ArgDescriptor Arg) {
    ArgInfo.WorkItemIDY = Arg;
  }

  void setWorkItemIDZ(ArgDescriptor Arg) {
    ArgInfo.WorkItemIDZ = Arg;
  }

  Register addPrivateSegmentWaveByteOffset() {
    ArgInfo.PrivateSegmentWaveByteOffset
      = ArgDescriptor::createRegister(getNextSystemSGPR());
    NumSystemSGPRs += 1;
    return ArgInfo.PrivateSegmentWaveByteOffset.getRegister();
  }

  void setPrivateSegmentWaveByteOffset(Register Reg) {
    ArgInfo.PrivateSegmentWaveByteOffset = ArgDescriptor::createRegister(Reg);
  }

  bool hasPrivateSegmentBuffer() const {
    return PrivateSegmentBuffer;
  }

  bool hasDispatchPtr() const {
    return DispatchPtr;
  }

  bool hasQueuePtr() const {
    return QueuePtr;
  }

  bool hasKernargSegmentPtr() const {
    return KernargSegmentPtr;
  }

  bool hasDispatchID() const {
    return DispatchID;
  }

  bool hasFlatScratchInit() const {
    return FlatScratchInit;
  }

  bool hasWorkGroupIDX() const {
    return WorkGroupIDX;
  }

  bool hasWorkGroupIDY() const {
    return WorkGroupIDY;
  }

  bool hasWorkGroupIDZ() const {
    return WorkGroupIDZ;
  }

  bool hasWorkGroupInfo() const {
    return WorkGroupInfo;
  }

  bool hasPrivateSegmentWaveByteOffset() const {
    return PrivateSegmentWaveByteOffset;
  }

  bool hasWorkItemIDX() const {
    return WorkItemIDX;
  }

  bool hasWorkItemIDY() const {
    return WorkItemIDY;
  }

  bool hasWorkItemIDZ() const {
    return WorkItemIDZ;
  }

  bool hasImplicitArgPtr() const {
    return ImplicitArgPtr;
  }

  bool hasHostcallPtr() const {
    return HostcallPtr;
  }

  bool hasImplicitBufferPtr() const {
    return ImplicitBufferPtr;
  }

  AMDGPUFunctionArgInfo &getArgInfo() {
    return ArgInfo;
  }

  const AMDGPUFunctionArgInfo &getArgInfo() const {
    return ArgInfo;
  }

  std::tuple<const ArgDescriptor *, const TargetRegisterClass *, LLT>
  getPreloadedValue(AMDGPUFunctionArgInfo::PreloadedValue Value) const {
    return ArgInfo.getPreloadedValue(Value);
  }

  MCRegister getPreloadedReg(AMDGPUFunctionArgInfo::PreloadedValue Value) const {
    auto Arg = std::get<0>(ArgInfo.getPreloadedValue(Value));
    return Arg ? Arg->getRegister() : MCRegister();
  }

  unsigned getGITPtrHigh() const {
    return GITPtrHigh;
  }

  Register getGITPtrLoReg(const MachineFunction &MF) const;

  uint32_t get32BitAddressHighBits() const {
    return HighBitsOf32BitAddress;
  }

  unsigned getGDSSize() const {
    return GDSSize;
  }

  unsigned getNumUserSGPRs() const {
    return NumUserSGPRs;
  }

  unsigned getNumPreloadedSGPRs() const {
    return NumUserSGPRs + NumSystemSGPRs;
  }

  Register getPrivateSegmentWaveByteOffsetSystemSGPR() const {
    return ArgInfo.PrivateSegmentWaveByteOffset.getRegister();
  }

  /// Returns the physical register reserved for use as the resource
  /// descriptor for scratch accesses.
  Register getScratchRSrcReg() const {
    return ScratchRSrcReg;
  }

  void setScratchRSrcReg(Register Reg) {
    assert(Reg != 0 && "Should never be unset");
    ScratchRSrcReg = Reg;
  }

  Register getFrameOffsetReg() const {
    return FrameOffsetReg;
  }

  void setFrameOffsetReg(Register Reg) {
    assert(Reg != 0 && "Should never be unset");
    FrameOffsetReg = Reg;
  }

  void setStackPtrOffsetReg(Register Reg) {
    assert(Reg != 0 && "Should never be unset");
    StackPtrOffsetReg = Reg;
  }

  // Note the unset value for this is AMDGPU::SP_REG rather than
  // NoRegister. This is mostly a workaround for MIR tests where state that
  // can't be directly computed from the function is not preserved in serialized
  // MIR.
  Register getStackPtrOffsetReg() const {
    return StackPtrOffsetReg;
  }

  Register getQueuePtrUserSGPR() const {
    return ArgInfo.QueuePtr.getRegister();
  }

  Register getImplicitBufferPtrUserSGPR() const {
    return ArgInfo.ImplicitBufferPtr.getRegister();
  }

  bool hasSpilledSGPRs() const {
    return HasSpilledSGPRs;
  }

  void setHasSpilledSGPRs(bool Spill = true) {
    HasSpilledSGPRs = Spill;
  }

  bool hasSpilledVGPRs() const {
    return HasSpilledVGPRs;
  }

  void setHasSpilledVGPRs(bool Spill = true) {
    HasSpilledVGPRs = Spill;
  }

  bool hasNonSpillStackObjects() const {
    return HasNonSpillStackObjects;
  }

  void setHasNonSpillStackObjects(bool StackObject = true) {
    HasNonSpillStackObjects = StackObject;
  }

  bool isStackRealigned() const {
    return IsStackRealigned;
  }

  void setIsStackRealigned(bool Realigned = true) {
    IsStackRealigned = Realigned;
  }

  unsigned getNumSpilledSGPRs() const {
    return NumSpilledSGPRs;
  }

  unsigned getNumSpilledVGPRs() const {
    return NumSpilledVGPRs;
  }

  void addToSpilledSGPRs(unsigned num) {
    NumSpilledSGPRs += num;
  }

  void addToSpilledVGPRs(unsigned num) {
    NumSpilledVGPRs += num;
  }

  unsigned getPSInputAddr() const {
    return PSInputAddr;
  }

  unsigned getPSInputEnable() const {
    return PSInputEnable;
  }

  bool isPSInputAllocated(unsigned Index) const {
    return PSInputAddr & (1 << Index);
  }

  void markPSInputAllocated(unsigned Index) {
    PSInputAddr |= 1 << Index;
  }

  void markPSInputEnabled(unsigned Index) {
    PSInputEnable |= 1 << Index;
  }

  bool returnsVoid() const {
    return ReturnsVoid;
  }

  void setIfReturnsVoid(bool Value) {
    ReturnsVoid = Value;
  }

  /// \returns A pair of default/requested minimum/maximum flat work group sizes
  /// for this function.
  std::pair<unsigned, unsigned> getFlatWorkGroupSizes() const {
    return FlatWorkGroupSizes;
  }

  /// \returns Default/requested minimum flat work group size for this function.
  unsigned getMinFlatWorkGroupSize() const {
    return FlatWorkGroupSizes.first;
  }

  /// \returns Default/requested maximum flat work group size for this function.
  unsigned getMaxFlatWorkGroupSize() const {
    return FlatWorkGroupSizes.second;
  }

  /// \returns A pair of default/requested minimum/maximum number of waves per
  /// execution unit.
  std::pair<unsigned, unsigned> getWavesPerEU() const {
    return WavesPerEU;
  }

  /// \returns Default/requested minimum number of waves per execution unit.
  unsigned getMinWavesPerEU() const {
    return WavesPerEU.first;
  }

  /// \returns Default/requested maximum number of waves per execution unit.
  unsigned getMaxWavesPerEU() const {
    return WavesPerEU.second;
  }

  /// \returns SGPR used for \p Dim's work group ID.
  Register getWorkGroupIDSGPR(unsigned Dim) const {
    switch (Dim) {
    case 0:
      assert(hasWorkGroupIDX());
      return ArgInfo.WorkGroupIDX.getRegister();
    case 1:
      assert(hasWorkGroupIDY());
      return ArgInfo.WorkGroupIDY.getRegister();
    case 2:
      assert(hasWorkGroupIDZ());
      return ArgInfo.WorkGroupIDZ.getRegister();
    }
    llvm_unreachable("unexpected dimension");
  }

  unsigned getLDSWaveSpillSize() const {
    return LDSWaveSpillSize;
  }

  const AMDGPUBufferPseudoSourceValue *getBufferPSV(const SIInstrInfo &TII) {
    if (!BufferPSV)
      BufferPSV = std::make_unique<AMDGPUBufferPseudoSourceValue>(TII);

    return BufferPSV.get();
  }

  const AMDGPUImagePseudoSourceValue *getImagePSV(const SIInstrInfo &TII) {
    if (!ImagePSV)
      ImagePSV = std::make_unique<AMDGPUImagePseudoSourceValue>(TII);

    return ImagePSV.get();
  }

  const AMDGPUGWSResourcePseudoSourceValue *getGWSPSV(const SIInstrInfo &TII) {
    if (!GWSResourcePSV) {
      GWSResourcePSV =
          std::make_unique<AMDGPUGWSResourcePseudoSourceValue>(TII);
    }

    return GWSResourcePSV.get();
  }

  unsigned getOccupancy() const {
    return Occupancy;
  }

  unsigned getMinAllowedOccupancy() const {
    if (!isMemoryBound() && !needsWaveLimiter())
      return Occupancy;
    return (Occupancy < 4) ? Occupancy : 4;
  }

  void limitOccupancy(const MachineFunction &MF);

  void limitOccupancy(unsigned Limit) {
    if (Occupancy > Limit)
      Occupancy = Limit;
  }

  void increaseOccupancy(const MachineFunction &MF, unsigned Limit) {
    if (Occupancy < Limit)
      Occupancy = Limit;
    limitOccupancy(MF);
  }

  bool mayNeedAGPRs() const {
    return MayNeedAGPRs;
  }

  // \returns true if a function has a use of AGPRs via inline asm or
  // has a call which may use it.
  bool mayUseAGPRs(const MachineFunction &MF) const;

  // \returns true if a function needs or may need AGPRs.
  bool usesAGPRs(const MachineFunction &MF) const;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_SIMACHINEFUNCTIONINFO_H
