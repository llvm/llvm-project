//==- AMDGPUArgumentrUsageInfo.h - Function Arg Usage Info -------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUARGUMENTUSAGEINFO_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUARGUMENTUSAGEINFO_H

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"

namespace llvm {

class Function;
class LLT;
class raw_ostream;
class TargetRegisterClass;
class TargetRegisterInfo;

struct ArgDescriptor {
private:
  friend struct AMDGPUFunctionArgInfo;
  friend class AMDGPUArgumentUsageInfo;

  union {
    MCRegister Reg;
    unsigned StackOffset;
  };

  // Bitmask to locate argument within the register.
  unsigned Mask;

  bool IsStack : 1;
  bool IsSet : 1;

public:
  ArgDescriptor(unsigned Val = 0, unsigned Mask = ~0u, bool IsStack = false,
                bool IsSet = false)
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

  MCRegister getRegister() const {
    assert(!IsStack);
    return Reg;
  }

  unsigned getStackOffset() const {
    assert(IsStack);
    return StackOffset;
  }

  unsigned getMask() const {
    // None of the target SGPRs or VGPRs are expected to have a 'zero' mask.
    assert(Mask && "Invalid mask.");
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

namespace KernArgPreload {

enum HiddenArg {
  HIDDEN_BLOCK_COUNT_X,
  HIDDEN_BLOCK_COUNT_Y,
  HIDDEN_BLOCK_COUNT_Z,
  HIDDEN_GROUP_SIZE_X,
  HIDDEN_GROUP_SIZE_Y,
  HIDDEN_GROUP_SIZE_Z,
  HIDDEN_REMAINDER_X,
  HIDDEN_REMAINDER_Y,
  HIDDEN_REMAINDER_Z,
  END_HIDDEN_ARGS
};

// Stores information about a specific hidden argument.
struct HiddenArgInfo {
  // Offset in bytes from the location in the kernearg segment pointed to by
  // the implicitarg pointer.
  uint8_t Offset;
  // The size of the hidden argument in bytes.
  uint8_t Size;
  // The name of the hidden argument in the kernel signature.
  const char *Name;
};

struct HiddenArgUtils {
  static constexpr HiddenArgInfo HiddenArgs[END_HIDDEN_ARGS] = {
      {0, 4, "_hidden_block_count_x"}, {4, 4, "_hidden_block_count_y"},
      {8, 4, "_hidden_block_count_z"}, {12, 2, "_hidden_group_size_x"},
      {14, 2, "_hidden_group_size_y"}, {16, 2, "_hidden_group_size_z"},
      {18, 2, "_hidden_remainder_x"},  {20, 2, "_hidden_remainder_y"},
      {22, 2, "_hidden_remainder_z"}};

  static HiddenArg getHiddenArgFromOffset(unsigned Offset) {
    for (unsigned I = 0; I < END_HIDDEN_ARGS; ++I)
      if (HiddenArgs[I].Offset == Offset)
        return static_cast<HiddenArg>(I);

    return END_HIDDEN_ARGS;
  }

  static Type *getHiddenArgType(LLVMContext &Ctx, HiddenArg HA) {
    if (HA < END_HIDDEN_ARGS)
      return static_cast<Type *>(Type::getIntNTy(Ctx, HiddenArgs[HA].Size * 8));

    llvm_unreachable("Unexpected hidden argument.");
  }

  static const char *getHiddenArgName(HiddenArg HA) {
    if (HA < END_HIDDEN_ARGS) {
      return HiddenArgs[HA].Name;
    }
    llvm_unreachable("Unexpected hidden argument.");
  }
};

struct KernArgPreloadDescriptor {
  // Id of the original argument in the IR kernel function argument list.
  unsigned OrigArgIdx = 0;

  // If this IR argument was split into multiple parts, this is the index of the
  // part in the original argument.
  unsigned PartIdx = 0;

  // The registers that the argument is preloaded into. The argument may be
  // split accross multilpe registers.
  SmallVector<MCRegister, 2> Regs;
};

} // namespace KernArgPreload

struct AMDGPUFunctionArgInfo {
  // clang-format off
  enum PreloadedValue {
    // SGPRS:
    PRIVATE_SEGMENT_BUFFER = 0,
    DISPATCH_PTR        =  1,
    QUEUE_PTR           =  2,
    KERNARG_SEGMENT_PTR =  3,
    DISPATCH_ID         =  4,
    FLAT_SCRATCH_INIT   =  5,
    LDS_KERNEL_ID       =  6, // LLVM internal, not part of the ABI
    WORKGROUP_ID_X      = 10,
    WORKGROUP_ID_Y      = 11,
    WORKGROUP_ID_Z      = 12,
    PRIVATE_SEGMENT_WAVE_BYTE_OFFSET = 14,
    IMPLICIT_BUFFER_PTR = 15,
    IMPLICIT_ARG_PTR = 16,
    PRIVATE_SEGMENT_SIZE = 17,

    // VGPRS:
    WORKITEM_ID_X       = 18,
    WORKITEM_ID_Y       = 19,
    WORKITEM_ID_Z       = 20,
    FIRST_VGPR_VALUE    = WORKITEM_ID_X
  };
  // clang-format on

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
  ArgDescriptor LDSKernelId;

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
  ArgDescriptor ImplicitBufferPtr;

  // VGPRs inputs. For entry functions these are either v0, v1 and v2 or packed
  // into v0, 10 bits per dimension if packed-tid is set.
  ArgDescriptor WorkItemIDX;
  ArgDescriptor WorkItemIDY;
  ArgDescriptor WorkItemIDZ;

  // Map the index of preloaded kernel arguments to its descriptor.
  SmallDenseMap<int, KernArgPreload::KernArgPreloadDescriptor>
      PreloadKernArgs{};
  // Map hidden argument to the index of it's descriptor.
  SmallDenseMap<KernArgPreload::HiddenArg, int> PreloadHiddenArgsIndexMap{};
  // The first user SGPR allocated for kernarg preloading.
  Register FirstKernArgPreloadReg;

  std::tuple<const ArgDescriptor *, const TargetRegisterClass *, LLT>
  getPreloadedValue(PreloadedValue Value) const;

  static AMDGPUFunctionArgInfo fixedABILayout();

  // Returns preload argument descriptors for an IR argument index. Isel may
  // split IR arguments into multiple parts, the return vector holds all parts
  // associated with an IR argument in the kernel signature.
  SmallVector<const KernArgPreload::KernArgPreloadDescriptor *, 4>
  getPreloadDescriptorsForArgIdx(unsigned ArgIdx) const;

  // Returns the hidden arguments `KernArgPreloadDescriptor` if it is preloaded.
  std::optional<const KernArgPreload::KernArgPreloadDescriptor *>
  getHiddenArgPreloadDescriptor(KernArgPreload::HiddenArg HA) const;
};

class AMDGPUArgumentUsageInfo : public ImmutablePass {
private:
  DenseMap<const Function *, AMDGPUFunctionArgInfo> ArgInfoMap;

public:
  static char ID;

  static const AMDGPUFunctionArgInfo ExternFunctionInfo;
  static const AMDGPUFunctionArgInfo FixedABIFunctionInfo;

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

  const AMDGPUFunctionArgInfo &lookupFuncArgInfo(const Function &F) const;
};

} // end namespace llvm

#endif
