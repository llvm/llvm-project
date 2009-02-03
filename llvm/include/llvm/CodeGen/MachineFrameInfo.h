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

#include "llvm/Support/DataTypes.h"
#include <cassert>
#include <iosfwd>
#include <vector>

namespace llvm {
class TargetData;
class TargetRegisterClass;
class Type;
class MachineModuleInfo;
class MachineFunction;
class TargetFrameInfo;

/// The CalleeSavedInfo class tracks the information need to locate where a
/// callee saved register in the current frame.  
class CalleeSavedInfo {

private:
  unsigned Reg;
  const TargetRegisterClass *RegClass;
  int FrameIdx;
  
public:
  CalleeSavedInfo(unsigned R, const TargetRegisterClass *RC, int FI = 0)
  : Reg(R)
  , RegClass(RC)
  , FrameIdx(FI)
  {}
  
  // Accessors.
  unsigned getReg()                        const { return Reg; }
  const TargetRegisterClass *getRegClass() const { return RegClass; }
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
    // The size of this object on the stack. 0 means a variable sized object,
    // ~0ULL means a dead object.
    uint64_t Size;

    // Alignment - The required alignment of this stack slot.
    unsigned Alignment;

    // isImmutable - If true, the value of the stack object is set before
    // entering the function and is not modified inside the function. By
    // default, fixed objects are immutable unless marked otherwise.
    bool isImmutable;

    // SPOffset - The offset of this object from the stack pointer on entry to
    // the function.  This field has no meaning for a variable sized element.
    int64_t SPOffset;
    
    StackObject(uint64_t Sz, unsigned Al, int64_t SP = 0, bool IM = false)
      : Size(Sz), Alignment(Al), isImmutable(IM), SPOffset(SP) {}
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
  /// to builtin @llvm.frameaddress.
  bool FrameAddressTaken;

  /// StackSize - The prolog/epilog code inserter calculates the final stack
  /// offsets for all of the fixed size objects, updating the Objects list
  /// above.  It then updates StackSize to contain the number of bytes that need
  /// to be allocated on entry to the function.
  ///
  uint64_t StackSize;
  
  /// OffsetAdjustment - The amount that a frame offset needs to be adjusted to
  /// have the actual offset from the stack/frame pointer.  The calculation is 
  /// MFI->getObjectOffset(Index) + StackSize - TFI.getOffsetOfLocalArea() +
  /// OffsetAdjustment.  If OffsetAdjustment is zero (default) then offsets are
  /// away from TOS. If OffsetAdjustment == StackSize then offsets are toward
  /// TOS.
  int OffsetAdjustment;
  
  /// MaxAlignment - The prolog/epilog code inserter may process objects 
  /// that require greater alignment than the default alignment the target
  /// provides. To handle this, MaxAlignment is set to the maximum alignment 
  /// needed by the objects on the current frame.  If this is greater than the
  /// native alignment maintained by the compiler, dynamic alignment code will
  /// be needed.
  ///
  unsigned MaxAlignment;

  /// HasCalls - Set to true if this function has any function calls.  This is
  /// only valid during and after prolog/epilog code insertion.
  bool HasCalls;

  /// StackProtectorIdx - The frame index for the stack protector.
  int StackProtectorIdx;

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
  
  /// MMI - This field is set (via setMachineModuleInfo) by a module info
  /// consumer (ex. DwarfWriter) to indicate that frame layout information
  /// should be acquired.  Typically, it's the responsibility of the target's
  /// TargetRegisterInfo prologue/epilogue emitting code to inform
  /// MachineModuleInfo of frame layouts.
  MachineModuleInfo *MMI;
  
  /// TargetFrameInfo - Target information about frame layout.
  ///
  const TargetFrameInfo &TFI;
public:
  explicit MachineFrameInfo(const TargetFrameInfo &tfi) : TFI(tfi) {
    StackSize = NumFixedObjects = OffsetAdjustment = MaxAlignment = 0;
    HasVarSizedObjects = false;
    FrameAddressTaken = false;
    HasCalls = false;
    StackProtectorIdx = -1;
    MaxCallFrameSize = 0;
    MMI = 0;
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

  /// isFrameAddressTaken - This method may be called any time after instruction
  /// selection is complete to determine if there is a call to
  /// @llvm.frameaddress in this function.
  bool isFrameAddressTaken() const { return FrameAddressTaken; }
  void setFrameAddressIsTaken(bool T) { FrameAddressTaken = T; }

  /// getObjectIndexBegin - Return the minimum frame object index.
  ///
  int getObjectIndexBegin() const { return -NumFixedObjects; }

  /// getObjectIndexEnd - Return one past the maximum frame object index.
  ///
  int getObjectIndexEnd() const { return (int)Objects.size()-NumFixedObjects; }

  /// getNumFixedObjects() - Return the number of fixed objects.
  unsigned getNumFixedObjects() const { return NumFixedObjects; }

  /// getNumObjects() - Return the number of objects.
  ///
  unsigned getNumObjects() const { return Objects.size(); }

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
  
  /// setMaxAlignment - Set the preferred alignment.
  ///
  void setMaxAlignment(unsigned Align) { MaxAlignment = Align; }
  
  /// hasCalls - Return true if the current function has no function calls.
  /// This is only valid during or after prolog/epilog code emission.
  ///
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
  int CreateFixedObject(uint64_t Size, int64_t SPOffset,
                        bool Immutable = true);
  
  
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
  int CreateStackObject(uint64_t Size, unsigned Alignment) {
    assert(Size != 0 && "Cannot allocate zero size stack objects!");
    Objects.push_back(StackObject(Size, Alignment));
    return (int)Objects.size()-NumFixedObjects-1;
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
  int CreateVariableSizedObject() {
    HasVarSizedObjects = true;
    Objects.push_back(StackObject(0, 1));
    return (int)Objects.size()-NumFixedObjects-1;
  }
  
  /// getCalleeSavedInfo - Returns a reference to call saved info vector for the
  /// current function.
  const std::vector<CalleeSavedInfo> &getCalleeSavedInfo() const {
    return CSInfo;
  }

  /// setCalleeSavedInfo - Used by prolog/epilog inserter to set the function's
  /// callee saved information.
  void  setCalleeSavedInfo(const std::vector<CalleeSavedInfo> &CSI) {
    CSInfo = CSI;
  }

  /// getMachineModuleInfo - Used by a prologue/epilogue
  /// emitter (TargetRegisterInfo) to provide frame layout information. 
  MachineModuleInfo *getMachineModuleInfo() const { return MMI; }

  /// setMachineModuleInfo - Used by a meta info consumer (DwarfWriter) to
  /// indicate that frame layout information should be gathered.
  void setMachineModuleInfo(MachineModuleInfo *mmi) { MMI = mmi; }

  /// print - Used by the MachineFunction printer to print information about
  /// stack objects.  Implemented in MachineFunction.cpp
  ///
  void print(const MachineFunction &MF, std::ostream &OS) const;

  /// dump - Call print(MF, std::cerr) to be called from the debugger.
  void dump(const MachineFunction &MF) const;
};

} // End llvm namespace

#endif
