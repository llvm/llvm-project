//===-- RegisterAliasingTracker.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines classes to keep track of register aliasing.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_EXEGESIS_ALIASINGTRACKER_H
#define LLVM_TOOLS_LLVM_EXEGESIS_ALIASINGTRACKER_H

#include <memory>
#include <unordered_map>

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/PackedVector.h"
#include "llvm/MC/MCRegisterInfo.h"

namespace llvm {
namespace exegesis {

// Returns the registers that are aliased by the ones set in SourceBits.
llvm::BitVector getAliasedBits(const llvm::MCRegisterInfo &RegInfo,
                               const llvm::BitVector &SourceBits);

// Keeps track of a mapping from one register (or a register class) to its
// aliased registers.
//
// e.g.
// RegisterAliasingTracker Tracker(RegInfo, llvm::X86::EAX);
// Tracker.sourceBits() == { llvm::X86::EAX }
// Tracker.aliasedBits() == { llvm::X86::AL, llvm::X86::AH, llvm::X86::AX,
//                            llvm::X86::EAX,llvm::X86::HAX, llvm::X86::RAX }
// Tracker.getOrigin(llvm::X86::AL) == llvm::X86::EAX;
// Tracker.getOrigin(llvm::X86::BX) == -1;
struct RegisterAliasingTracker {
  // Construct a tracker from an MCRegisterClass.
  RegisterAliasingTracker(const llvm::MCRegisterInfo &RegInfo,
                          const llvm::BitVector &ReservedReg,
                          const llvm::MCRegisterClass &RegClass);

  // Construct a tracker from an MCPhysReg.
  RegisterAliasingTracker(const llvm::MCRegisterInfo &RegInfo,
                          const llvm::MCPhysReg Register);

  const llvm::BitVector &sourceBits() const { return SourceBits; }

  // Retrieves all the touched registers as a BitVector.
  const llvm::BitVector &aliasedBits() const { return AliasedBits; }

  // Returns the origin of this register or -1.
  int getOrigin(llvm::MCPhysReg Aliased) const {
    if (!AliasedBits[Aliased])
      return -1;
    return Origins[Aliased];
  }

private:
  RegisterAliasingTracker(const llvm::MCRegisterInfo &RegInfo);
  RegisterAliasingTracker(const RegisterAliasingTracker &) = delete;

  void FillOriginAndAliasedBits(const llvm::MCRegisterInfo &RegInfo,
                                const llvm::BitVector &OriginalBits);

  llvm::BitVector SourceBits;
  llvm::BitVector AliasedBits;
  llvm::PackedVector<size_t, 10> Origins; // Max 1024 physical registers.
};

// A cache of existing trackers.
struct RegisterAliasingTrackerCache {
  // RegInfo must outlive the cache.
  RegisterAliasingTrackerCache(const llvm::MCRegisterInfo &RegInfo,
                               const llvm::BitVector &ReservedReg);

  // Convenient function to retrieve a BitVector of the right size.
  const llvm::BitVector &emptyRegisters() const { return EmptyRegisters; }

  // Convenient function to retrieve the registers the function body can't use.
  const llvm::BitVector &reservedRegisters() const { return ReservedReg; }

  // Convenient function to retrieve the underlying MCRegInfo.
  const llvm::MCRegisterInfo &regInfo() const { return RegInfo; }

  // Retrieves the RegisterAliasingTracker for this particular register.
  const RegisterAliasingTracker &getRegister(llvm::MCPhysReg Reg) const;

  // Retrieves the RegisterAliasingTracker for this particular register class.
  const RegisterAliasingTracker &getRegisterClass(unsigned RegClassIndex) const;

private:
  const llvm::MCRegisterInfo &RegInfo;
  const llvm::BitVector ReservedReg;
  const llvm::BitVector EmptyRegisters;
  mutable std::unordered_map<unsigned, std::unique_ptr<RegisterAliasingTracker>>
      Registers;
  mutable std::unordered_map<unsigned, std::unique_ptr<RegisterAliasingTracker>>
      RegisterClasses;
};

} // namespace exegesis
} // namespace llvm

#endif // LLVM_TOOLS_LLVM_EXEGESIS_ALIASINGTRACKER_H
