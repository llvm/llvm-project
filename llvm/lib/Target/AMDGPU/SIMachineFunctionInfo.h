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


#ifndef LLVM_LIB_TARGET_R600_SIMACHINEFUNCTIONINFO_H
#define LLVM_LIB_TARGET_R600_SIMACHINEFUNCTIONINFO_H

#include "AMDGPUMachineFunction.h"
#include "SIRegisterInfo.h"
#include <map>

namespace llvm {

class MachineRegisterInfo;

/// This class keeps track of the SPI_SP_INPUT_ADDR config register, which
/// tells the hardware which interpolation parameters to load.
class SIMachineFunctionInfo : public AMDGPUMachineFunction {
  void anchor() override;

  unsigned TIDReg;
  unsigned ScratchRSrcReg;

public:
  // FIXME: Make private
  unsigned LDSWaveSpillSize;
  unsigned PSInputAddr;
  std::map<unsigned, unsigned> LaneVGPRs;
  unsigned ScratchOffsetReg;
  unsigned NumUserSGPRs;

private:
  bool HasSpilledSGPRs;
  bool HasSpilledVGPRs;

  // Feature bits required for inputs passed in user / system SGPRs.
  bool DispatchPtr : 1;
  bool QueuePtr : 1;
  bool DispatchID : 1;
  bool KernargSegmentPtr : 1;
  bool FlatScratchInit : 1;
  bool GridWorkgroupCountX : 1;
  bool GridWorkgroupCountY : 1;
  bool GridWorkgroupCountZ : 1;

  bool WorkGroupIDX : 1; // Always initialized.
  bool WorkGroupIDY : 1;
  bool WorkGroupIDZ : 1;
  bool WorkGroupInfo : 1;

  bool WorkItemIDX : 1; // Always initialized.
  bool WorkItemIDY : 1;
  bool WorkItemIDZ : 1;

public:
  struct SpilledReg {
    unsigned VGPR;
    int Lane;
    SpilledReg(unsigned R, int L) : VGPR (R), Lane (L) { }
    SpilledReg() : VGPR(0), Lane(-1) { }
    bool hasLane() { return Lane != -1;}
  };

  // SIMachineFunctionInfo definition

  SIMachineFunctionInfo(const MachineFunction &MF);
  SpilledReg getSpilledReg(MachineFunction *MF, unsigned FrameIndex,
                           unsigned SubIdx);
  bool hasCalculatedTID() const { return TIDReg != AMDGPU::NoRegister; };
  unsigned getTIDReg() const { return TIDReg; };
  void setTIDReg(unsigned Reg) { TIDReg = Reg; }

  bool hasDispatchPtr() const {
    return DispatchPtr;
  }

  bool hasQueuePtr() const {
    return QueuePtr;
  }

  bool hasDispatchID() const {
    return DispatchID;
  }

  bool hasKernargSegmentPtr() const {
    return KernargSegmentPtr;
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

  bool hasWorkItemIDX() const {
    return WorkItemIDX;
  }

  bool hasWorkItemIDY() const {
    return WorkItemIDY;
  }

  bool hasWorkItemIDZ() const {
    return WorkItemIDZ;
  }

  /// \brief Returns the physical register reserved for use as the resource
  /// descriptor for scratch accesses.
  unsigned getScratchRSrcReg() const {
    return ScratchRSrcReg;
  }

  void setScratchRSrcReg(const SIRegisterInfo *TRI);

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

  unsigned getMaximumWorkGroupSize(const MachineFunction &MF) const;
};

} // End namespace llvm


#endif
