//===- SIMachineFunctionInfo.h - SIMachineFunctionInfo interface -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_SIMACHINEFUNCTIONINFO_H
#define LLVM_LIB_TARGET_AMDGPU_SIMACHINEFUNCTIONINFO_H

#include "AMDGPUMachineFunction.h"
#include "SIRegisterInfo.h"
#include <array>
#include <map>

namespace llvm {

class MachineRegisterInfo;

/// This class keeps track of the SPI_SP_INPUT_ADDR config register, which
/// tells the hardware which interpolation parameters to load.
class SIMachineFunctionInfo final : public AMDGPUMachineFunction {
  // FIXME: This should be removed and getPreloadedValue moved here.
  friend struct SIRegisterInfo;

  unsigned TIDReg;

  // Registers that may be reserved for spilling purposes. These may be the same
  // as the input registers.
  unsigned ScratchRSrcReg;
  unsigned ScratchWaveOffsetReg;

  // Input registers setup for the HSA ABI.
  // User SGPRs in allocation order.
  unsigned PrivateSegmentBufferUserSGPR;
  unsigned DispatchPtrUserSGPR;
  unsigned QueuePtrUserSGPR;
  unsigned KernargSegmentPtrUserSGPR;
  unsigned DispatchIDUserSGPR;
  unsigned FlatScratchInitUserSGPR;
  unsigned PrivateSegmentSizeUserSGPR;
  unsigned GridWorkGroupCountXUserSGPR;
  unsigned GridWorkGroupCountYUserSGPR;
  unsigned GridWorkGroupCountZUserSGPR;

  // System SGPRs in allocation order.
  unsigned WorkGroupIDXSystemSGPR;
  unsigned WorkGroupIDYSystemSGPR;
  unsigned WorkGroupIDZSystemSGPR;
  unsigned WorkGroupInfoSystemSGPR;
  unsigned PrivateSegmentWaveByteOffsetSystemSGPR;

  // Graphics info.
  unsigned PSInputAddr;
  bool ReturnsVoid;

  unsigned MaximumWorkGroupSize;

  // Number of reserved VGPRs for debugger usage.
  unsigned DebuggerReservedVGPRCount;
  // Stack object indices for work group IDs.
  std::array<int, 3> DebuggerWorkGroupIDStackObjectIndices;
  // Stack object indices for work item IDs.
  std::array<int, 3> DebuggerWorkItemIDStackObjectIndices;

public:
  // FIXME: Make private
  unsigned LDSWaveSpillSize;
  unsigned PSInputEna;
  std::map<unsigned, unsigned> LaneVGPRs;
  unsigned ScratchOffsetReg;
  unsigned NumUserSGPRs;
  unsigned NumSystemSGPRs;

private:
  bool HasSpilledSGPRs;
  bool HasSpilledVGPRs;
  bool HasNonSpillStackObjects;

  unsigned NumSpilledSGPRs;
  unsigned NumSpilledVGPRs;

  // Feature bits required for inputs passed in user SGPRs.
  bool PrivateSegmentBuffer : 1;
  bool DispatchPtr : 1;
  bool QueuePtr : 1;
  bool KernargSegmentPtr : 1;
  bool DispatchID : 1;
  bool FlatScratchInit : 1;
  bool GridWorkgroupCountX : 1;
  bool GridWorkgroupCountY : 1;
  bool GridWorkgroupCountZ : 1;

  // Feature bits required for inputs passed in system SGPRs.
  bool WorkGroupIDX : 1; // Always initialized.
  bool WorkGroupIDY : 1;
  bool WorkGroupIDZ : 1;
  bool WorkGroupInfo : 1;
  bool PrivateSegmentWaveByteOffset : 1;

  bool WorkItemIDX : 1; // Always initialized.
  bool WorkItemIDY : 1;
  bool WorkItemIDZ : 1;

  MCPhysReg getNextUserSGPR() const {
    assert(NumSystemSGPRs == 0 && "System SGPRs must be added after user SGPRs");
    return AMDGPU::SGPR0 + NumUserSGPRs;
  }

  MCPhysReg getNextSystemSGPR() const {
    return AMDGPU::SGPR0 + NumUserSGPRs + NumSystemSGPRs;
  }

public:
  struct SpilledReg {
    unsigned VGPR;
    int Lane;
    SpilledReg(unsigned R, int L) : VGPR (R), Lane (L) { }
    SpilledReg() : VGPR(AMDGPU::NoRegister), Lane(-1) { }
    bool hasLane() { return Lane != -1;}
    bool hasReg() { return VGPR != AMDGPU::NoRegister;}
  };

  // SIMachineFunctionInfo definition

  SIMachineFunctionInfo(const MachineFunction &MF);
  SpilledReg getSpilledReg(MachineFunction *MF, unsigned FrameIndex,
                           unsigned SubIdx);
  bool hasCalculatedTID() const { return TIDReg != AMDGPU::NoRegister; };
  unsigned getTIDReg() const { return TIDReg; };
  void setTIDReg(unsigned Reg) { TIDReg = Reg; }

  // Add user SGPRs.
  unsigned addPrivateSegmentBuffer(const SIRegisterInfo &TRI);
  unsigned addDispatchPtr(const SIRegisterInfo &TRI);
  unsigned addQueuePtr(const SIRegisterInfo &TRI);
  unsigned addKernargSegmentPtr(const SIRegisterInfo &TRI);
  unsigned addDispatchID(const SIRegisterInfo &TRI);
  unsigned addFlatScratchInit(const SIRegisterInfo &TRI);

  // Add system SGPRs.
  unsigned addWorkGroupIDX() {
    WorkGroupIDXSystemSGPR = getNextSystemSGPR();
    NumSystemSGPRs += 1;
    return WorkGroupIDXSystemSGPR;
  }

  unsigned addWorkGroupIDY() {
    WorkGroupIDYSystemSGPR = getNextSystemSGPR();
    NumSystemSGPRs += 1;
    return WorkGroupIDYSystemSGPR;
  }

  unsigned addWorkGroupIDZ() {
    WorkGroupIDZSystemSGPR = getNextSystemSGPR();
    NumSystemSGPRs += 1;
    return WorkGroupIDZSystemSGPR;
  }

  unsigned addWorkGroupInfo() {
    WorkGroupInfoSystemSGPR = getNextSystemSGPR();
    NumSystemSGPRs += 1;
    return WorkGroupInfoSystemSGPR;
  }

  unsigned addPrivateSegmentWaveByteOffset() {
    PrivateSegmentWaveByteOffsetSystemSGPR = getNextSystemSGPR();
    NumSystemSGPRs += 1;
    return PrivateSegmentWaveByteOffsetSystemSGPR;
  }

  void setPrivateSegmentWaveByteOffset(unsigned Reg) {
    PrivateSegmentWaveByteOffsetSystemSGPR = Reg;
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

  bool hasGridWorkgroupCountX() const {
    return GridWorkgroupCountX;
  }

  bool hasGridWorkgroupCountY() const {
    return GridWorkgroupCountY;
  }

  bool hasGridWorkgroupCountZ() const {
    return GridWorkgroupCountZ;
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

  unsigned getNumUserSGPRs() const {
    return NumUserSGPRs;
  }

  unsigned getNumPreloadedSGPRs() const {
    return NumUserSGPRs + NumSystemSGPRs;
  }

  unsigned getPrivateSegmentWaveByteOffsetSystemSGPR() const {
    return PrivateSegmentWaveByteOffsetSystemSGPR;
  }

  /// \brief Returns the physical register reserved for use as the resource
  /// descriptor for scratch accesses.
  unsigned getScratchRSrcReg() const {
    return ScratchRSrcReg;
  }

  void setScratchRSrcReg(unsigned Reg) {
    assert(Reg != AMDGPU::NoRegister && "Should never be unset");
    ScratchRSrcReg = Reg;
  }

  unsigned getScratchWaveOffsetReg() const {
    return ScratchWaveOffsetReg;
  }

  void setScratchWaveOffsetReg(unsigned Reg) {
    assert(Reg != AMDGPU::NoRegister && "Should never be unset");
    ScratchWaveOffsetReg = Reg;
  }

  unsigned getQueuePtrUserSGPR() const {
    return QueuePtrUserSGPR;
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

  bool isPSInputAllocated(unsigned Index) const {
    return PSInputAddr & (1 << Index);
  }

  void markPSInputAllocated(unsigned Index) {
    PSInputAddr |= 1 << Index;
  }

  bool returnsVoid() const {
    return ReturnsVoid;
  }

  void setIfReturnsVoid(bool Value) {
    ReturnsVoid = Value;
  }

  /// \returns Number of reserved VGPRs for debugger usage.
  unsigned getDebuggerReservedVGPRCount() const {
    return DebuggerReservedVGPRCount;
  }

  /// \returns Stack object index for \p Dim's work group ID.
  int getDebuggerWorkGroupIDStackObjectIndex(unsigned Dim) const {
    assert(Dim < 3);
    return DebuggerWorkGroupIDStackObjectIndices[Dim];
  }

  /// \brief Sets stack object index for \p Dim's work group ID to \p ObjectIdx.
  void setDebuggerWorkGroupIDStackObjectIndex(unsigned Dim, int ObjectIdx) {
    assert(Dim < 3);
    DebuggerWorkGroupIDStackObjectIndices[Dim] = ObjectIdx;
  }

  /// \returns Stack object index for \p Dim's work item ID.
  int getDebuggerWorkItemIDStackObjectIndex(unsigned Dim) const {
    assert(Dim < 3);
    return DebuggerWorkItemIDStackObjectIndices[Dim];
  }

  /// \brief Sets stack object index for \p Dim's work item ID to \p ObjectIdx.
  void setDebuggerWorkItemIDStackObjectIndex(unsigned Dim, int ObjectIdx) {
    assert(Dim < 3);
    DebuggerWorkItemIDStackObjectIndices[Dim] = ObjectIdx;
  }

  /// \returns SGPR used for \p Dim's work group ID.
  unsigned getWorkGroupIDSGPR(unsigned Dim) const {
    switch (Dim) {
    case 0:
      assert(hasWorkGroupIDX());
      return WorkGroupIDXSystemSGPR;
    case 1:
      assert(hasWorkGroupIDY());
      return WorkGroupIDYSystemSGPR;
    case 2:
      assert(hasWorkGroupIDZ());
      return WorkGroupIDZSystemSGPR;
    }
    llvm_unreachable("unexpected dimension");
  }

  /// \returns VGPR used for \p Dim' work item ID.
  unsigned getWorkItemIDVGPR(unsigned Dim) const {
    switch (Dim) {
    case 0:
      assert(hasWorkItemIDX());
      return AMDGPU::VGPR0;
    case 1:
      assert(hasWorkItemIDY());
      return AMDGPU::VGPR1;
    case 2:
      assert(hasWorkItemIDZ());
      return AMDGPU::VGPR2;
    }
    llvm_unreachable("unexpected dimension");
  }

  unsigned getMaximumWorkGroupSize(const MachineFunction &MF) const;
};

} // End namespace llvm

#endif
