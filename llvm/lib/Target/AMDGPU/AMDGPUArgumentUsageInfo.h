//==- AMDGPUArgumentrUsageInfo.h - Function Arg Usage Info -------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUARGUMENTUSAGEINFO_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUARGUMENTUSAGEINFO_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"

namespace llvm {

class Function;
class raw_ostream;
class GCNSubtarget;
class TargetMachine;
class TargetRegisterClass;
class TargetRegisterInfo;

struct ArgDescriptor {
private:
  friend struct AMDGPUFunctionArgInfo;
  friend class AMDGPUArgumentUsageInfo;

  union {
    Register Reg;
    unsigned StackOffset;
  };

  // Bitmask to locate argument within the register.
  unsigned Mask;

  bool IsStack : 1;
  bool IsSet : 1;

public:
  ArgDescriptor(unsigned Val = 0, unsigned Mask = ~0u,
                bool IsStack = false, bool IsSet = false)
    : Reg(Val), Mask(Mask), IsStack(IsStack), IsSet(IsSet) {}

  static ArgDescriptor createRegister(Register Reg, unsigned Mask = ~0u) {
    return ArgDescriptor(Reg, Mask, false, true);
  }

  static ArgDescriptor createStack(unsigned Offset, unsigned Mask = ~0u) {
    return ArgDescriptor(Offset, Mask, true, true);
  }

  static ArgDescriptor createArg(const ArgDescriptor &Arg, unsigned Mask) {
    return ArgDescriptor(Arg.Reg, Mask, Arg.IsStack, Arg.IsSet);
  }

  bool isSet() const {
    return IsSet;
  }

  explicit operator bool() const {
    return isSet();
  }

  bool isRegister() const {
    return !IsStack;
  }

  Register getRegister() const {
    assert(!IsStack);
    return Reg;
  }

  unsigned getStackOffset() const {
    assert(IsStack);
    return StackOffset;
  }

  unsigned getMask() const {
    return Mask;
  }

  bool isMasked() const {
    return Mask != ~0u;
  }

  void print(raw_ostream &OS, const TargetRegisterInfo *TRI = nullptr) const;
};

inline raw_ostream &operator<<(raw_ostream &OS, const ArgDescriptor &Arg) {
  Arg.print(OS);
  return OS;
}

struct AMDGPUFunctionArgInfo {
  enum PreloadedValue {
    // SGPRS:
    PRIVATE_SEGMENT_BUFFER = 0,
    DISPATCH_PTR        =  1,
    QUEUE_PTR           =  2,
    KERNARG_SEGMENT_PTR =  3,
    DISPATCH_ID         =  4,
    FLAT_SCRATCH_INIT   =  5,
    WORKGROUP_ID_X      = 10,
    WORKGROUP_ID_Y      = 11,
    WORKGROUP_ID_Z      = 12,
    PRIVATE_SEGMENT_WAVE_BYTE_OFFSET = 14,
    IMPLICIT_BUFFER_PTR = 15,
    IMPLICIT_ARG_PTR = 16,

    // VGPRS:
    WORKITEM_ID_X       = 17,
    WORKITEM_ID_Y       = 18,
    WORKITEM_ID_Z       = 19,
    FIRST_VGPR_VALUE    = WORKITEM_ID_X
  };

  // Kernel input registers setup for the HSA ABI in allocation order.

  // User SGPRs in kernels
  // XXX - Can these require argument spills?
  ArgDescriptor PrivateSegmentBuffer;
  ArgDescriptor DispatchPtr;
  ArgDescriptor QueuePtr;
  ArgDescriptor KernargSegmentPtr;
  ArgDescriptor DispatchID;
  ArgDescriptor FlatScratchInit;
  ArgDescriptor PrivateSegmentSize;

  // System SGPRs in kernels.
  ArgDescriptor WorkGroupIDX;
  ArgDescriptor WorkGroupIDY;
  ArgDescriptor WorkGroupIDZ;
  ArgDescriptor WorkGroupInfo;
  ArgDescriptor PrivateSegmentWaveByteOffset;

  // Pointer with offset from kernargsegmentptr to where special ABI arguments
  // are passed to callable functions.
  ArgDescriptor ImplicitArgPtr;

  // Input registers for non-HSA ABI
  ArgDescriptor ImplicitBufferPtr = 0;

  // VGPRs inputs. These are always v0, v1 and v2 for entry functions.
  ArgDescriptor WorkItemIDX;
  ArgDescriptor WorkItemIDY;
  ArgDescriptor WorkItemIDZ;

  std::pair<const ArgDescriptor *, const TargetRegisterClass *>
  getPreloadedValue(PreloadedValue Value) const;
};

class AMDGPUArgumentUsageInfo : public ImmutablePass {
private:
  static const AMDGPUFunctionArgInfo ExternFunctionInfo;
  DenseMap<const Function *, AMDGPUFunctionArgInfo> ArgInfoMap;

public:
  static char ID;

  AMDGPUArgumentUsageInfo() : ImmutablePass(ID) { }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  bool doInitialization(Module &M) override;
  bool doFinalization(Module &M) override;

  void print(raw_ostream &OS, const Module *M = nullptr) const override;

  void setFuncArgInfo(const Function &F, const AMDGPUFunctionArgInfo &ArgInfo) {
    ArgInfoMap[&F] = ArgInfo;
  }

  const AMDGPUFunctionArgInfo &lookupFuncArgInfo(const Function &F) const {
    auto I = ArgInfoMap.find(&F);
    if (I == ArgInfoMap.end()) {
      assert(F.isDeclaration());
      return ExternFunctionInfo;
    }

    return I->second;
  }
};

} // end namespace llvm

#endif
