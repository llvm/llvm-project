//===-- CodeGen/MachineFrameInfo.h - Abstract Stack Frame Rep. --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The file defines the MachineFrameInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEFRAMEINFO_H
#define LLVM_CODEGEN_MACHINEFRAMEINFO_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DataTypes.h"
#include <cassert>
#include <vector>

namespace llvm {
class raw_ostream;
class DataLayout;
class TargetRegisterClass;
class Type;
class MachineFunction;
class MachineBasicBlock;
class TargetFrameLowering;
class BitVector;
class Value;
class AllocaInst;

/// The CalleeSavedInfo class tracks the information need to locate where a
/// callee saved register is in the current frame.
class CalleeSavedInfo {
  unsigned Reg;
  int FrameIdx;

public:
  explicit CalleeSavedInfo(unsigned R, int FI = 0)
  : Reg(R), FrameIdx(FI) {}

  // Accessors.
  unsigned getReg()                        const { return Reg; }
  int getFrameIdx()                        const { return FrameIdx; }
  void setFrameIdx(int FI)                       { FrameIdx = FI; }
};

/// The MachineFrameInfo class represents an abstract stack frame until
/// prolog/epilog code is inserted.  This class is key to allowing stack frame
/// representation optimizations, such as frame pointer elimination.  It also
/// allows more mundane (but still important) optimizations, such as reordering
/// of abstract objects on the stack frame.
///
/// To support this, the class assigns unique integer identifiers to stack
/// objects requested clients.  These identifiers are negative integers for
/// fixed stack objects (such as arguments passed on the stack) or nonnegative
/// for objects that may be reordered.  Instructions which refer to stack
/// objects use a special MO_FrameIndex operand to represent these frame
/// indexes.
///
/// Because this class keeps track of all references to the stack frame, it
/// knows when a variable sized object is allocated on the stack.  This is the
/// sole condition which prevents frame pointer elimination, which is an
/// important optimization on register-poor architectures.  Because original
/// variable sized alloca's in the source program are the only source of
/// variable sized stack objects, it is safe to decide whether there will be
/// any variable sized objects before all stack objects are known (for
/// example, register allocator spill code never needs variable sized
/// objects).
///
/// When prolog/epilog code emission is performed, the final stack frame is
/// built and the machine instructions are modified to refer to the actual
/// stack offsets of the object, eliminating all MO_FrameIndex operands from
/// the program.
///
/// @brief Abstract Stack Frame Information
class MachineFrameInfo {

  // StackObject - Represent a single object allocated on the stack.
  struct StackObject {
    // SPOffset - The offset of this object from the stack pointer on entry to
    // the function.  This field has no meaning for a variable sized element.
    int64_t SPOffset;

    // The size of this object on the stack. 0 means a variable sized object,
    // ~0ULL means a dead object.
    uint64_t Size;

    // Alignment - The required alignment of this stack slot.
    unsigned Alignment;

    // isImmutable - If true, the value of the stack object is set before
    // entering the function and is not modified inside the function. By
    // default, fixed objects are immutable unless marked otherwise.
    bool isImmutable;

    // isSpillSlot - If true the stack object is used as spill slot. It
    // cannot alias any other memory objects.
    bool isSpillSlot;

    // MayNeedSP - If true the stack object triggered the creation of the stack
    // protector. We should allocate this object right after the stack
    // protector.
    bool MayNeedSP;

    /// Alloca - If this stack object is originated from an Alloca instruction
    /// this value saves the original IR allocation. Can be NULL.
    const AllocaInst *Alloca;

    // PreAllocated - If true, the object was mapped into the local frame
    // block and doesn't need additional handling for allocation beyond that.
    bool PreAllocated;

    StackObject(uint64_t Sz, unsigned Al, int64_t SP, bool IM,
                bool isSS, bool NSP, const AllocaInst *Val)
      : SPOffset(SP), Size(Sz), Alignment(Al), isImmutable(IM),
        isSpillSlot(isSS), MayNeedSP(NSP), Alloca(Val), PreAllocated(false) {}
  };

  /// Objects - The list of stack objects allocated...
  ///
  std::vector<StackObject> Objects;

  /// NumFixedObjects - This contains the number of fixed objects contained on
  /// the stack.  Because fixed objects are stored at a negative index in the
  /// Objects list, this is also the index to the 0th object in the list.
  ///
  unsigned NumFixedObjects;

  /// HasVarSizedObjects - This boolean keeps track of whether any variable
  /// sized objects have been allocated yet.
  ///
  bool HasVarSizedObjects;

  /// FrameAddressTaken - This boolean keeps track of whether there is a call
  /// to builtin \@llvm.frameaddress.
  bool FrameAddressTaken;

  /// ReturnAddressTaken - This boolean keeps track of whether there is a call
  /// to builtin \@llvm.returnaddress.
  bool ReturnAddressTaken;

  /// StackSize - The prolog/epilog code inserter calculates the final stack
  /// offsets for all of the fixed size objects, updating the Objects list
  /// above.  It then updates StackSize to contain the number of bytes that need
  /// to be allocated on entry to the function.
  ///
  uint64_t StackSize;

  /// OffsetAdjustment - The amount that a frame offset needs to be adjusted to
  /// have the actual offset from the stack/frame pointer.  The exact usage of
  /// this is target-dependent, but it is typically used to adjust between
  /// SP-relative and FP-relative offsets.  E.G., if objects are accessed via
  /// SP then OffsetAdjustment is zero; if FP is used, OffsetAdjustment is set
  /// to the distance between the initial SP and the value in FP.  For many
  /// targets, this value is only used when generating debug info (via
  /// TargetRegisterInfo::getFrameIndexOffset); when generating code, the
  /// corresponding adjustments are performed directly.
  int OffsetAdjustment;

  /// MaxAlignment - The prolog/epilog code inserter may process objects
  /// that require greater alignment than the default alignment the target
  /// provides. To handle this, MaxAlignment is set to the maximum alignment
  /// needed by the objects on the current frame.  If this is greater than the
  /// native alignment maintained by the compiler, dynamic alignment code will
  /// be needed.
  ///
  unsigned MaxAlignment;

  /// AdjustsStack - Set to true if this function adjusts the stack -- e.g.,
  /// when calling another function. This is only valid during and after
  /// prolog/epilog code insertion.
  bool AdjustsStack;

  /// HasCalls - Set to true if this function has any function calls.
  bool HasCalls;

  /// StackProtectorIdx - The frame index for the stack protector.
  int StackProtectorIdx;

  /// FunctionContextIdx - The frame index for the function context. Used for
  /// SjLj exceptions.
  int FunctionContextIdx;

  /// MaxCallFrameSize - This contains the size of the largest call frame if the
  /// target uses frame setup/destroy pseudo instructions (as defined in the
  /// TargetFrameInfo class).  This information is important for frame pointer
  /// elimination.  If is only valid during and after prolog/epilog code
  /// insertion.
  ///
  unsigned MaxCallFrameSize;

  /// CSInfo - The prolog/epilog code inserter fills in this vector with each
  /// callee saved register saved in the frame.  Beyond its use by the prolog/
  /// epilog code inserter, this data used for debug info and exception
  /// handling.
  std::vector<CalleeSavedInfo> CSInfo;

  /// CSIValid - Has CSInfo been set yet?
  bool CSIValid;

  /// TargetFrameLowering - Target information about frame layout.
  ///
  const TargetFrameLowering &TFI;

  /// LocalFrameObjects - References to frame indices which are mapped
  /// into the local frame allocation block. <FrameIdx, LocalOffset>
  SmallVector<std::pair<int, int64_t>, 32> LocalFrameObjects;

  /// LocalFrameSize - Size of the pre-allocated local frame block.
  int64_t LocalFrameSize;

  /// Required alignment of the local object blob, which is the strictest
  /// alignment of any object in it.
  unsigned LocalFrameMaxAlign;

  /// Whether the local object blob needs to be allocated together. If not,
  /// PEI should ignore the isPreAllocated flags on the stack objects and
  /// just allocate them normally.
  bool UseLocalStackAllocationBlock;

public:
    explicit MachineFrameInfo(const TargetFrameLowering &tfi) : TFI(tfi) {
    StackSize = NumFixedObjects = OffsetAdjustment = MaxAlignment = 0;
    HasVarSizedObjects = false;
    FrameAddressTaken = false;
    ReturnAddressTaken = false;
    AdjustsStack = false;
    HasCalls = false;
    StackProtectorIdx = -1;
    FunctionContextIdx = -1;
    MaxCallFrameSize = 0;
    CSIValid = false;
    LocalFrameSize = 0;
    LocalFrameMaxAlign = 0;
    UseLocalStackAllocationBlock = false;
  }

  /// hasStackObjects - Return true if there are any stack objects in this
  /// function.
  ///
  bool hasStackObjects() const { return !Objects.empty(); }

  /// hasVarSizedObjects - This method may be called any time after instruction
  /// selection is complete to determine if the stack frame for this function
  /// contains any variable sized objects.
  ///
  bool hasVarSizedObjects() const { return HasVarSizedObjects; }

  /// getStackProtectorIndex/setStackProtectorIndex - Return the index for the
  /// stack protector object.
  ///
  int getStackProtectorIndex() const { return StackProtectorIdx; }
  void setStackProtectorIndex(int I) { StackProtectorIdx = I; }

  /// getFunctionContextIndex/setFunctionContextIndex - Return the index for the
  /// function context object. This object is used for SjLj exceptions.
  int getFunctionContextIndex() const { return FunctionContextIdx; }
  void setFunctionContextIndex(int I) { FunctionContextIdx = I; }

  /// isFrameAddressTaken - This method may be called any time after instruction
  /// selection is complete to determine if there is a call to
  /// \@llvm.frameaddress in this function.
  bool isFrameAddressTaken() const { return FrameAddressTaken; }
  void setFrameAddressIsTaken(bool T) { FrameAddressTaken = T; }

  /// isReturnAddressTaken - This method may be called any time after
  /// instruction selection is complete to determine if there is a call to
  /// \@llvm.returnaddress in this function.
  bool isReturnAddressTaken() const { return ReturnAddressTaken; }
  void setReturnAddressIsTaken(bool s) { ReturnAddressTaken = s; }

  /// getObjectIndexBegin - Return the minimum frame object index.
  ///
  int getObjectIndexBegin() const { return -NumFixedObjects; }

  /// getObjectIndexEnd - Return one past the maximum frame object index.
  ///
  int getObjectIndexEnd() const { return (int)Objects.size()-NumFixedObjects; }

  /// getNumFixedObjects - Return the number of fixed objects.
  unsigned getNumFixedObjects() const { return NumFixedObjects; }

  /// getNumObjects - Return the number of objects.
  ///
  unsigned getNumObjects() const { return Objects.size(); }

  /// mapLocalFrameObject - Map a frame index into the local object block
  void mapLocalFrameObject(int ObjectIndex, int64_t Offset) {
    LocalFrameObjects.push_back(std::pair<int, int64_t>(ObjectIndex, Offset));
    Objects[ObjectIndex + NumFixedObjects].PreAllocated = true;
  }

  /// getLocalFrameObjectMap - Get the local offset mapping for a for an object
  std::pair<int, int64_t> getLocalFrameObjectMap(int i) {
    assert (i >= 0 && (unsigned)i < LocalFrameObjects.size() &&
            "Invalid local object reference!");
    return LocalFrameObjects[i];
  }

  /// getLocalFrameObjectCount - Return the number of objects allocated into
  /// the local object block.
  int64_t getLocalFrameObjectCount() { return LocalFrameObjects.size(); }

  /// setLocalFrameSize - Set the size of the local object blob.
  void setLocalFrameSize(int64_t sz) { LocalFrameSize = sz; }

  /// getLocalFrameSize - Get the size of the local object blob.
  int64_t getLocalFrameSize() const { return LocalFrameSize; }

  /// setLocalFrameMaxAlign - Required alignment of the local object blob,
  /// which is the strictest alignment of any object in it.
  void setLocalFrameMaxAlign(unsigned Align) { LocalFrameMaxAlign = Align; }

  /// getLocalFrameMaxAlign - Return the required alignment of the local
  /// object blob.
  unsigned getLocalFrameMaxAlign() const { return LocalFrameMaxAlign; }

  /// getUseLocalStackAllocationBlock - Get whether the local allocation blob
  /// should be allocated together or let PEI allocate the locals in it
  /// directly.
  bool getUseLocalStackAllocationBlock() {return UseLocalStackAllocationBlock;}

  /// setUseLocalStackAllocationBlock - Set whether the local allocation blob
  /// should be allocated together or let PEI allocate the locals in it
  /// directly.
  void setUseLocalStackAllocationBlock(bool v) {
    UseLocalStackAllocationBlock = v;
  }

  /// isObjectPreAllocated - Return true if the object was pre-allocated into
  /// the local block.
  bool isObjectPreAllocated(int ObjectIdx) const {
    assert(unsigned(ObjectIdx+NumFixedObjects) < Objects.size() &&
           "Invalid Object Idx!");
    return Objects[ObjectIdx+NumFixedObjects].PreAllocated;
  }

  /// getObjectSize - Return the size of the specified object.
  ///
  int64_t getObjectSize(int ObjectIdx) const {
    assert(unsigned(ObjectIdx+NumFixedObjects) < Objects.size() &&
           "Invalid Object Idx!");
    return Objects[ObjectIdx+NumFixedObjects].Size;
  }

  /// setObjectSize - Change the size of the specified stack object.
  void setObjectSize(int ObjectIdx, int64_t Size) {
    assert(unsigned(ObjectIdx+NumFixedObjects) < Objects.size() &&
           "Invalid Object Idx!");
    Objects[ObjectIdx+NumFixedObjects].Size = Size;
  }

  /// getObjectAlignment - Return the alignment of the specified stack object.
  unsigned getObjectAlignment(int ObjectIdx) const {
    assert(unsigned(ObjectIdx+NumFixedObjects) < Objects.size() &&
           "Invalid Object Idx!");
    return Objects[ObjectIdx+NumFixedObjects].Alignment;
  }

  /// setObjectAlignment - Change the alignment of the specified stack object.
  void setObjectAlignment(int ObjectIdx, unsigned Align) {
    assert(unsigned(ObjectIdx+NumFixedObjects) < Objects.size() &&
           "Invalid Object Idx!");
    Objects[ObjectIdx+NumFixedObjects].Alignment = Align;
    ensureMaxAlignment(Align);
  }

  /// getObjectAllocation - Return the underlying Alloca of the specified
  /// stack object if it exists. Returns 0 if none exists.
  const AllocaInst* getObjectAllocation(int ObjectIdx) const {
    assert(unsigned(ObjectIdx+NumFixedObjects) < Objects.size() &&
           "Invalid Object Idx!");
    return Objects[ObjectIdx+NumFixedObjects].Alloca;
  }

  /// NeedsStackProtector - Returns true if the object may need stack
  /// protectors.
  bool MayNeedStackProtector(int ObjectIdx) const {
    assert(unsigned(ObjectIdx+NumFixedObjects) < Objects.size() &&
           "Invalid Object Idx!");
    return Objects[ObjectIdx+NumFixedObjects].MayNeedSP;
  }

  /// getObjectOffset - Return the assigned stack offset of the specified object
  /// from the incoming stack pointer.
  ///
  int64_t getObjectOffset(int ObjectIdx) const {
    assert(unsigned(ObjectIdx+NumFixedObjects) < Objects.size() &&
           "Invalid Object Idx!");
    assert(!isDeadObjectIndex(ObjectIdx) &&
           "Getting frame offset for a dead object?");
    return Objects[ObjectIdx+NumFixedObjects].SPOffset;
  }

  /// setObjectOffset - Set the stack frame offset of the specified object.  The
  /// offset is relative to the stack pointer on entry to the function.
  ///
  void setObjectOffset(int ObjectIdx, int64_t SPOffset) {
    assert(unsigned(ObjectIdx+NumFixedObjects) < Objects.size() &&
           "Invalid Object Idx!");
    assert(!isDeadObjectIndex(ObjectIdx) &&
           "Setting frame offset for a dead object?");
    Objects[ObjectIdx+NumFixedObjects].SPOffset = SPOffset;
  }

  /// getStackSize - Return the number of bytes that must be allocated to hold
  /// all of the fixed size frame objects.  This is only valid after
  /// Prolog/Epilog code insertion has finalized the stack frame layout.
  ///
  uint64_t getStackSize() const { return StackSize; }

  /// setStackSize - Set the size of the stack...
  ///
  void setStackSize(uint64_t Size) { StackSize = Size; }

  /// getOffsetAdjustment - Return the correction for frame offsets.
  ///
  int getOffsetAdjustment() const { return OffsetAdjustment; }

  /// setOffsetAdjustment - Set the correction for frame offsets.
  ///
  void setOffsetAdjustment(int Adj) { OffsetAdjustment = Adj; }

  /// getMaxAlignment - Return the alignment in bytes that this function must be
  /// aligned to, which is greater than the default stack alignment provided by
  /// the target.
  ///
  unsigned getMaxAlignment() const { return MaxAlignment; }

  /// ensureMaxAlignment - Make sure the function is at least Align bytes
  /// aligned.
  void ensureMaxAlignment(unsigned Align) {
    if (MaxAlignment < Align) MaxAlignment = Align;
  }

  /// AdjustsStack - Return true if this function adjusts the stack -- e.g.,
  /// when calling another function. This is only valid during and after
  /// prolog/epilog code insertion.
  bool adjustsStack() const { return AdjustsStack; }
  void setAdjustsStack(bool V) { AdjustsStack = V; }

  /// hasCalls - Return true if the current function has any function calls.
  bool hasCalls() const { return HasCalls; }
  void setHasCalls(bool V) { HasCalls = V; }

  /// getMaxCallFrameSize - Return the maximum size of a call frame that must be
  /// allocated for an outgoing function call.  This is only available if
  /// CallFrameSetup/Destroy pseudo instructions are used by the target, and
  /// then only during or after prolog/epilog code insertion.
  ///
  unsigned getMaxCallFrameSize() const { return MaxCallFrameSize; }
  void setMaxCallFrameSize(unsigned S) { MaxCallFrameSize = S; }

  /// CreateFixedObject - Create a new object at a fixed location on the stack.
  /// All fixed objects should be created before other objects are created for
  /// efficiency. By default, fixed objects are immutable. This returns an
  /// index with a negative value.
  ///
  int CreateFixedObject(uint64_t Size, int64_t SPOffset, bool Immutable);


  /// isFixedObjectIndex - Returns true if the specified index corresponds to a
  /// fixed stack object.
  bool isFixedObjectIndex(int ObjectIdx) const {
    return ObjectIdx < 0 && (ObjectIdx >= -(int)NumFixedObjects);
  }

  /// isImmutableObjectIndex - Returns true if the specified index corresponds
  /// to an immutable object.
  bool isImmutableObjectIndex(int ObjectIdx) const {
    assert(unsigned(ObjectIdx+NumFixedObjects) < Objects.size() &&
           "Invalid Object Idx!");
    return Objects[ObjectIdx+NumFixedObjects].isImmutable;
  }

  /// isSpillSlotObjectIndex - Returns true if the specified index corresponds
  /// to a spill slot..
  bool isSpillSlotObjectIndex(int ObjectIdx) const {
    assert(unsigned(ObjectIdx+NumFixedObjects) < Objects.size() &&
           "Invalid Object Idx!");
    return Objects[ObjectIdx+NumFixedObjects].isSpillSlot;
  }

  /// isDeadObjectIndex - Returns true if the specified index corresponds to
  /// a dead object.
  bool isDeadObjectIndex(int ObjectIdx) const {
    assert(unsigned(ObjectIdx+NumFixedObjects) < Objects.size() &&
           "Invalid Object Idx!");
    return Objects[ObjectIdx+NumFixedObjects].Size == ~0ULL;
  }

  /// CreateStackObject - Create a new statically sized stack object, returning
  /// a nonnegative identifier to represent it.
  ///
  int CreateStackObject(uint64_t Size, unsigned Alignment, bool isSS,
                        bool MayNeedSP = false, const AllocaInst *Alloca = 0) {
    assert(Size != 0 && "Cannot allocate zero size stack objects!");
    Objects.push_back(StackObject(Size, Alignment, 0, false, isSS, MayNeedSP,
                                  Alloca));
    int Index = (int)Objects.size() - NumFixedObjects - 1;
    assert(Index >= 0 && "Bad frame index!");
    ensureMaxAlignment(Alignment);
    return Index;
  }

  /// CreateSpillStackObject - Create a new statically sized stack object that
  /// represents a spill slot, returning a nonnegative identifier to represent
  /// it.
  ///
  int CreateSpillStackObject(uint64_t Size, unsigned Alignment) {
    CreateStackObject(Size, Alignment, true, false);
    int Index = (int)Objects.size() - NumFixedObjects - 1;
    ensureMaxAlignment(Alignment);
    return Index;
  }

  /// RemoveStackObject - Remove or mark dead a statically sized stack object.
  ///
  void RemoveStackObject(int ObjectIdx) {
    // Mark it dead.
    Objects[ObjectIdx+NumFixedObjects].Size = ~0ULL;
  }

  /// CreateVariableSizedObject - Notify the MachineFrameInfo object that a
  /// variable sized object has been created.  This must be created whenever a
  /// variable sized object is created, whether or not the index returned is
  /// actually used.
  ///
  int CreateVariableSizedObject(unsigned Alignment) {
    HasVarSizedObjects = true;
    Objects.push_back(StackObject(0, Alignment, 0, false, false, true, 0));
    ensureMaxAlignment(Alignment);
    return (int)Objects.size()-NumFixedObjects-1;
  }

  /// getCalleeSavedInfo - Returns a reference to call saved info vector for the
  /// current function.
  const std::vector<CalleeSavedInfo> &getCalleeSavedInfo() const {
    return CSInfo;
  }

  /// setCalleeSavedInfo - Used by prolog/epilog inserter to set the function's
  /// callee saved information.
  void setCalleeSavedInfo(const std::vector<CalleeSavedInfo> &CSI) {
    CSInfo = CSI;
  }

  /// isCalleeSavedInfoValid - Has the callee saved info been calculated yet?
  bool isCalleeSavedInfoValid() const { return CSIValid; }

  void setCalleeSavedInfoValid(bool v) { CSIValid = v; }

  /// getPristineRegs - Return a set of physical registers that are pristine on
  /// entry to the MBB.
  ///
  /// Pristine registers hold a value that is useless to the current function,
  /// but that must be preserved - they are callee saved registers that have not
  /// been saved yet.
  ///
  /// Before the PrologueEpilogueInserter has placed the CSR spill code, this
  /// method always returns an empty set.
  BitVector getPristineRegs(const MachineBasicBlock *MBB) const;

  /// print - Used by the MachineFunction printer to print information about
  /// stack objects. Implemented in MachineFunction.cpp
  ///
  void print(const MachineFunction &MF, raw_ostream &OS) const;

  /// dump - Print the function to stderr.
  void dump(const MachineFunction &MF) const;
};

} // End llvm namespace

#endif
