//===-- llvm/Target/TargetFrameInfo.h ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Interface to describe the layout of a stack frame on the target machine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETFRAMEINFO_H
#define LLVM_TARGET_TARGETFRAMEINFO_H

#include <utility>

namespace llvm {

/// Information about stack frame layout on the target.  It holds the direction
/// of stack growth, the known stack alignment on entry to each function, and
/// the offset to the locals area.
///
/// The offset to the local area is the offset from the stack pointer on
/// function entry to the first location where function data (local variables,
/// spill locations) can be stored.
class TargetFrameInfo {
public:
  enum StackDirection {
    StackGrowsUp,        // Adding to the stack increases the stack address
    StackGrowsDown       // Adding to the stack decreases the stack address
  };
private:
  StackDirection StackDir;
  unsigned StackAlignment;
  int LocalAreaOffset;
public:
  TargetFrameInfo(StackDirection D, unsigned StackAl, int LAO)
    : StackDir(D), StackAlignment(StackAl), LocalAreaOffset(LAO) {}

  virtual ~TargetFrameInfo();

  // These methods return information that describes the abstract stack layout
  // of the target machine.

  /// getStackGrowthDirection - Return the direction the stack grows
  ///
  StackDirection getStackGrowthDirection() const { return StackDir; }

  /// getStackAlignment - This method returns the number of bytes that the stack
  /// pointer must be aligned to.  Typically, this is the largest alignment for
  /// any data object in the target.
  ///
  unsigned getStackAlignment() const { return StackAlignment; }

  /// getOffsetOfLocalArea - This method returns the offset of the local area
  /// from the stack pointer on entrance to a function.
  ///
  int getOffsetOfLocalArea() const { return LocalAreaOffset; }

  /// getCalleeSavedSpillSlots - This method returns a pointer to an array of
  /// pairs, that contains an entry for each callee saved register that must be
  /// spilled to a particular stack location if it is spilled.
  ///
  /// Each entry in this array contains a <register,offset> pair, indicating the
  /// fixed offset from the incoming stack pointer that each register should be
  /// spilled at.  If a register is not listed here, the code generator is
  /// allowed to spill it anywhere it chooses.
  ///
  virtual const std::pair<unsigned, int> *
  getCalleeSavedSpillSlots(unsigned &NumEntries) const {
    NumEntries = 0;
    return 0;
  }
};

} // End llvm namespace

#endif
