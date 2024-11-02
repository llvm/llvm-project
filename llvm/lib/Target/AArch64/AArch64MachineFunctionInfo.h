//=- AArch64MachineFunctionInfo.h - AArch64 machine function info -*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares AArch64-specific per-machine-function information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AARCH64_AARCH64MACHINEFUNCTIONINFO_H
#define LLVM_LIB_TARGET_AARCH64_AARCH64MACHINEFUNCTIONINFO_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MIRYamlMapping.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/IR/Function.h"
#include "llvm/MC/MCLinkerOptimizationHint.h"
#include <cassert>

namespace llvm {

namespace yaml {
struct AArch64FunctionInfo;
} // end namespace yaml

class MachineInstr;

/// AArch64FunctionInfo - This class is derived from MachineFunctionInfo and
/// contains private AArch64-specific information for each MachineFunction.
class AArch64FunctionInfo final : public MachineFunctionInfo {
  /// Backreference to the machine function.
  MachineFunction *MF;

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

  /// The number of bytes to restore to deallocate space for incoming
  /// arguments. Canonically 0 in the C calling convention, but non-zero when
  /// callee is expected to pop the args.
  unsigned ArgumentStackToRestore = 0;

  /// Space just below incoming stack pointer reserved for arguments being
  /// passed on the stack during a tail call. This will be the difference
  /// between the largest tail call argument space needed in this function and
  /// what's already available by reusing space of incoming arguments.
  unsigned TailCallReservedStack = 0;

  /// HasStackFrame - True if this function has a stack frame. Set by
  /// determineCalleeSaves().
  bool HasStackFrame = false;

  /// Amount of stack frame size, not including callee-saved registers.
  uint64_t LocalStackSize = 0;

  /// The start and end frame indices for the SVE callee saves.
  int MinSVECSFrameIndex = 0;
  int MaxSVECSFrameIndex = 0;

  /// Amount of stack frame size used for saving callee-saved registers.
  unsigned CalleeSavedStackSize = 0;
  unsigned SVECalleeSavedStackSize = 0;
  bool HasCalleeSavedStackSize = false;

  /// Number of TLS accesses using the special (combinable)
  /// _TLS_MODULE_BASE_ symbol.
  unsigned NumLocalDynamicTLSAccesses = 0;

  /// FrameIndex for start of varargs area for arguments passed on the
  /// stack.
  int VarArgsStackIndex = 0;

  /// Offset of start of varargs area for arguments passed on the stack.
  unsigned VarArgsStackOffset = 0;

  /// FrameIndex for start of varargs area for arguments passed in
  /// general purpose registers.
  int VarArgsGPRIndex = 0;

  /// Size of the varargs area for arguments passed in general purpose
  /// registers.
  unsigned VarArgsGPRSize = 0;

  /// FrameIndex for start of varargs area for arguments passed in
  /// floating-point registers.
  int VarArgsFPRIndex = 0;

  /// Size of the varargs area for arguments passed in floating-point
  /// registers.
  unsigned VarArgsFPRSize = 0;

  /// True if this function has a subset of CSRs that is handled explicitly via
  /// copies.
  bool IsSplitCSR = false;

  /// True when the stack gets realigned dynamically because the size of stack
  /// frame is unknown at compile time. e.g., in case of VLAs.
  bool StackRealigned = false;

  /// True when the callee-save stack area has unused gaps that may be used for
  /// other stack allocations.
  bool CalleeSaveStackHasFreeSpace = false;

  /// SRetReturnReg - sret lowering includes returning the value of the
  /// returned struct in a register. This field holds the virtual register into
  /// which the sret argument is passed.
  Register SRetReturnReg;

  /// SVE stack size (for predicates and data vectors) are maintained here
  /// rather than in FrameInfo, as the placement and Stack IDs are target
  /// specific.
  uint64_t StackSizeSVE = 0;

  /// HasCalculatedStackSizeSVE indicates whether StackSizeSVE is valid.
  bool HasCalculatedStackSizeSVE = false;

  /// Has a value when it is known whether or not the function uses a
  /// redzone, and no value otherwise.
  /// Initialized during frame lowering, unless the function has the noredzone
  /// attribute, in which case it is set to false at construction.
  Optional<bool> HasRedZone;

  /// ForwardedMustTailRegParms - A list of virtual and physical registers
  /// that must be forwarded to every musttail call.
  SmallVector<ForwardedRegister, 1> ForwardedMustTailRegParms;

  /// FrameIndex for the tagged base pointer.
  Optional<int> TaggedBasePointerIndex;

  /// Offset from SP-at-entry to the tagged base pointer.
  /// Tagged base pointer is set up to point to the first (lowest address)
  /// tagged stack slot.
  unsigned TaggedBasePointerOffset;

  /// OutliningStyle denotes, if a function was outined, how it was outlined,
  /// e.g. Tail Call, Thunk, or Function if none apply.
  Optional<std::string> OutliningStyle;

  // Offset from SP-after-callee-saved-spills (i.e. SP-at-entry minus
  // CalleeSavedStackSize) to the address of the frame record.
  int CalleeSaveBaseToFrameRecordOffset = 0;

  /// SignReturnAddress is true if PAC-RET is enabled for the function with
  /// defaults being sign non-leaf functions only, with the B key.
  bool SignReturnAddress = false;

  /// SignReturnAddressAll modifies the default PAC-RET mode to signing leaf
  /// functions as well.
  bool SignReturnAddressAll = false;

  /// SignWithBKey modifies the default PAC-RET mode to signing with the B key.
  bool SignWithBKey = false;

  /// BranchTargetEnforcement enables placing BTI instructions at potential
  /// indirect branch destinations.
  bool BranchTargetEnforcement = false;

  /// Whether this function has an extended frame record [Ctx, FP, LR]. If so,
  /// bit 60 of the in-memory FP will be 1 to enable other tools to detect the
  /// extended record.
  bool HasSwiftAsyncContext = false;

  /// The stack slot where the Swift asynchronous context is stored.
  int SwiftAsyncContextFrameIdx = std::numeric_limits<int>::max();

  bool IsMTETagged = false;

  /// The function has Scalable Vector or Scalable Predicate register argument
  /// or return type
  bool IsSVECC = false;

  /// The frame-index for the TPIDR2 object used for lazy saves.
  Register LazySaveTPIDR2Obj = 0;


  /// True if the function need unwind information.
  mutable Optional<bool> NeedsDwarfUnwindInfo;

  /// True if the function need asynchronous unwind information.
  mutable Optional<bool> NeedsAsyncDwarfUnwindInfo;

public:
  explicit AArch64FunctionInfo(MachineFunction &MF);

  MachineFunctionInfo *
  clone(BumpPtrAllocator &Allocator, MachineFunction &DestMF,
        const DenseMap<MachineBasicBlock *, MachineBasicBlock *> &Src2DstMBB)
      const override;

  bool isSVECC() const { return IsSVECC; };
  void setIsSVECC(bool s) { IsSVECC = s; };

  unsigned getLazySaveTPIDR2Obj() const { return LazySaveTPIDR2Obj; }
  void setLazySaveTPIDR2Obj(unsigned Reg) { LazySaveTPIDR2Obj = Reg; }

  void initializeBaseYamlFields(const yaml::AArch64FunctionInfo &YamlMFI);

  unsigned getBytesInStackArgArea() const { return BytesInStackArgArea; }
  void setBytesInStackArgArea(unsigned bytes) { BytesInStackArgArea = bytes; }

  unsigned getArgumentStackToRestore() const { return ArgumentStackToRestore; }
  void setArgumentStackToRestore(unsigned bytes) {
    ArgumentStackToRestore = bytes;
  }

  unsigned getTailCallReservedStack() const { return TailCallReservedStack; }
  void setTailCallReservedStack(unsigned bytes) {
    TailCallReservedStack = bytes;
  }

  bool hasCalculatedStackSizeSVE() const { return HasCalculatedStackSizeSVE; }

  void setStackSizeSVE(uint64_t S) {
    HasCalculatedStackSizeSVE = true;
    StackSizeSVE = S;
  }

  uint64_t getStackSizeSVE() const { return StackSizeSVE; }

  bool hasStackFrame() const { return HasStackFrame; }
  void setHasStackFrame(bool s) { HasStackFrame = s; }

  bool isStackRealigned() const { return StackRealigned; }
  void setStackRealigned(bool s) { StackRealigned = s; }

  bool hasCalleeSaveStackFreeSpace() const {
    return CalleeSaveStackHasFreeSpace;
  }
  void setCalleeSaveStackHasFreeSpace(bool s) {
    CalleeSaveStackHasFreeSpace = s;
  }
  bool isSplitCSR() const { return IsSplitCSR; }
  void setIsSplitCSR(bool s) { IsSplitCSR = s; }

  void setLocalStackSize(uint64_t Size) { LocalStackSize = Size; }
  uint64_t getLocalStackSize() const { return LocalStackSize; }

  void setOutliningStyle(std::string Style) { OutliningStyle = Style; }
  Optional<std::string> getOutliningStyle() const { return OutliningStyle; }

  void setCalleeSavedStackSize(unsigned Size) {
    CalleeSavedStackSize = Size;
    HasCalleeSavedStackSize = true;
  }

  // When CalleeSavedStackSize has not been set (for example when
  // some MachineIR pass is run in isolation), then recalculate
  // the CalleeSavedStackSize directly from the CalleeSavedInfo.
  // Note: This information can only be recalculated after PEI
  // has assigned offsets to the callee save objects.
  unsigned getCalleeSavedStackSize(const MachineFrameInfo &MFI) const {
    bool ValidateCalleeSavedStackSize = false;

#ifndef NDEBUG
    // Make sure the calculated size derived from the CalleeSavedInfo
    // equals the cached size that was calculated elsewhere (e.g. in
    // determineCalleeSaves).
    ValidateCalleeSavedStackSize = HasCalleeSavedStackSize;
#endif

    if (!HasCalleeSavedStackSize || ValidateCalleeSavedStackSize) {
      assert(MFI.isCalleeSavedInfoValid() && "CalleeSavedInfo not calculated");
      if (MFI.getCalleeSavedInfo().empty())
        return 0;

      int64_t MinOffset = std::numeric_limits<int64_t>::max();
      int64_t MaxOffset = std::numeric_limits<int64_t>::min();
      for (const auto &Info : MFI.getCalleeSavedInfo()) {
        int FrameIdx = Info.getFrameIdx();
        if (MFI.getStackID(FrameIdx) != TargetStackID::Default)
          continue;
        int64_t Offset = MFI.getObjectOffset(FrameIdx);
        int64_t ObjSize = MFI.getObjectSize(FrameIdx);
        MinOffset = std::min<int64_t>(Offset, MinOffset);
        MaxOffset = std::max<int64_t>(Offset + ObjSize, MaxOffset);
      }

      if (SwiftAsyncContextFrameIdx != std::numeric_limits<int>::max()) {
        int64_t Offset = MFI.getObjectOffset(getSwiftAsyncContextFrameIdx());
        int64_t ObjSize = MFI.getObjectSize(getSwiftAsyncContextFrameIdx());
        MinOffset = std::min<int64_t>(Offset, MinOffset);
        MaxOffset = std::max<int64_t>(Offset + ObjSize, MaxOffset);
      }

      unsigned Size = alignTo(MaxOffset - MinOffset, 16);
      assert((!HasCalleeSavedStackSize || getCalleeSavedStackSize() == Size) &&
             "Invalid size calculated for callee saves");
      return Size;
    }

    return getCalleeSavedStackSize();
  }

  unsigned getCalleeSavedStackSize() const {
    assert(HasCalleeSavedStackSize &&
           "CalleeSavedStackSize has not been calculated");
    return CalleeSavedStackSize;
  }

  // Saves the CalleeSavedStackSize for SVE vectors in 'scalable bytes'
  void setSVECalleeSavedStackSize(unsigned Size) {
    SVECalleeSavedStackSize = Size;
  }
  unsigned getSVECalleeSavedStackSize() const {
    return SVECalleeSavedStackSize;
  }

  void setMinMaxSVECSFrameIndex(int Min, int Max) {
    MinSVECSFrameIndex = Min;
    MaxSVECSFrameIndex = Max;
  }

  int getMinSVECSFrameIndex() const { return MinSVECSFrameIndex; }
  int getMaxSVECSFrameIndex() const { return MaxSVECSFrameIndex; }

  void incNumLocalDynamicTLSAccesses() { ++NumLocalDynamicTLSAccesses; }
  unsigned getNumLocalDynamicTLSAccesses() const {
    return NumLocalDynamicTLSAccesses;
  }

  Optional<bool> hasRedZone() const { return HasRedZone; }
  void setHasRedZone(bool s) { HasRedZone = s; }

  int getVarArgsStackIndex() const { return VarArgsStackIndex; }
  void setVarArgsStackIndex(int Index) { VarArgsStackIndex = Index; }

  unsigned getVarArgsStackOffset() const { return VarArgsStackOffset; }
  void setVarArgsStackOffset(unsigned Offset) { VarArgsStackOffset = Offset; }

  int getVarArgsGPRIndex() const { return VarArgsGPRIndex; }
  void setVarArgsGPRIndex(int Index) { VarArgsGPRIndex = Index; }

  unsigned getVarArgsGPRSize() const { return VarArgsGPRSize; }
  void setVarArgsGPRSize(unsigned Size) { VarArgsGPRSize = Size; }

  int getVarArgsFPRIndex() const { return VarArgsFPRIndex; }
  void setVarArgsFPRIndex(int Index) { VarArgsFPRIndex = Index; }

  unsigned getVarArgsFPRSize() const { return VarArgsFPRSize; }
  void setVarArgsFPRSize(unsigned Size) { VarArgsFPRSize = Size; }

  unsigned getSRetReturnReg() const { return SRetReturnReg; }
  void setSRetReturnReg(unsigned Reg) { SRetReturnReg = Reg; }

  unsigned getJumpTableEntrySize(int Idx) const {
    return JumpTableEntryInfo[Idx].first;
  }
  MCSymbol *getJumpTableEntryPCRelSymbol(int Idx) const {
    return JumpTableEntryInfo[Idx].second;
  }
  void setJumpTableEntryInfo(int Idx, unsigned Size, MCSymbol *PCRelSym) {
    if ((unsigned)Idx >= JumpTableEntryInfo.size())
      JumpTableEntryInfo.resize(Idx+1);
    JumpTableEntryInfo[Idx] = std::make_pair(Size, PCRelSym);
  }

  using SetOfInstructions = SmallPtrSet<const MachineInstr *, 16>;

  const SetOfInstructions &getLOHRelated() const { return LOHRelated; }

  // Shortcuts for LOH related types.
  class MILOHDirective {
    MCLOHType Kind;

    /// Arguments of this directive. Order matters.
    SmallVector<const MachineInstr *, 3> Args;

  public:
    using LOHArgs = ArrayRef<const MachineInstr *>;

    MILOHDirective(MCLOHType Kind, LOHArgs Args)
        : Kind(Kind), Args(Args.begin(), Args.end()) {
      assert(isValidMCLOHType(Kind) && "Invalid LOH directive type!");
    }

    MCLOHType getKind() const { return Kind; }
    LOHArgs getArgs() const { return Args; }
  };

  using MILOHArgs = MILOHDirective::LOHArgs;
  using MILOHContainer = SmallVector<MILOHDirective, 32>;

  const MILOHContainer &getLOHContainer() const { return LOHContainerSet; }

  /// Add a LOH directive of this @p Kind and this @p Args.
  void addLOHDirective(MCLOHType Kind, MILOHArgs Args) {
    LOHContainerSet.push_back(MILOHDirective(Kind, Args));
    LOHRelated.insert(Args.begin(), Args.end());
  }

  SmallVectorImpl<ForwardedRegister> &getForwardedMustTailRegParms() {
    return ForwardedMustTailRegParms;
  }

  Optional<int> getTaggedBasePointerIndex() const {
    return TaggedBasePointerIndex;
  }
  void setTaggedBasePointerIndex(int Index) { TaggedBasePointerIndex = Index; }

  unsigned getTaggedBasePointerOffset() const {
    return TaggedBasePointerOffset;
  }
  void setTaggedBasePointerOffset(unsigned Offset) {
    TaggedBasePointerOffset = Offset;
  }

  int getCalleeSaveBaseToFrameRecordOffset() const {
    return CalleeSaveBaseToFrameRecordOffset;
  }
  void setCalleeSaveBaseToFrameRecordOffset(int Offset) {
    CalleeSaveBaseToFrameRecordOffset = Offset;
  }

  bool shouldSignReturnAddress() const;
  bool shouldSignReturnAddress(bool SpillsLR) const;

  bool shouldSignWithBKey() const { return SignWithBKey; }
  bool isMTETagged() const { return IsMTETagged; }

  bool branchTargetEnforcement() const { return BranchTargetEnforcement; }

  void setHasSwiftAsyncContext(bool HasContext) {
    HasSwiftAsyncContext = HasContext;
  }
  bool hasSwiftAsyncContext() const { return HasSwiftAsyncContext; }

  void setSwiftAsyncContextFrameIdx(int FI) {
    SwiftAsyncContextFrameIdx = FI;
  }
  int getSwiftAsyncContextFrameIdx() const { return SwiftAsyncContextFrameIdx; }

  bool needsDwarfUnwindInfo() const;
  bool needsAsyncDwarfUnwindInfo() const;

private:
  // Hold the lists of LOHs.
  MILOHContainer LOHContainerSet;
  SetOfInstructions LOHRelated;

  SmallVector<std::pair<unsigned, MCSymbol *>, 2> JumpTableEntryInfo;
};

namespace yaml {
struct AArch64FunctionInfo final : public yaml::MachineFunctionInfo {
  Optional<bool> HasRedZone;

  AArch64FunctionInfo() = default;
  AArch64FunctionInfo(const llvm::AArch64FunctionInfo &MFI);

  void mappingImpl(yaml::IO &YamlIO) override;
  ~AArch64FunctionInfo() = default;
};

template <> struct MappingTraits<AArch64FunctionInfo> {
  static void mapping(IO &YamlIO, AArch64FunctionInfo &MFI) {
    YamlIO.mapOptional("hasRedZone", MFI.HasRedZone);
  }
};

} // end namespace yaml

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AARCH64_AARCH64MACHINEFUNCTIONINFO_H
