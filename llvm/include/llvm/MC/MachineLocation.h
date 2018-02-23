//===- llvm/MC/MachineLocation.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// The MachineLocation class is used to represent a simple location in a machine
// frame.  Locations will be one of two forms; a register or an address formed
// from a base address plus an offset.  Register indirection can be specified by
// explicitly passing an offset to the constructor.
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MACHINELOCATION_H
#define LLVM_MC_MACHINELOCATION_H

#include <cstdint>

namespace llvm {

class MachineLocation {
private:
  bool IsRegister = false;              // True if location is a register.
  unsigned Register = 0;                // gcc/gdb register number.
  int Offset = 0;                       // Displacement if not register.

public:
  enum : uint32_t {
    // The target register number for an abstract frame pointer. The value is
    // an arbitrary value that doesn't collide with any real target register.
    VirtualFP = ~0U
  };

  MachineLocation() = default;
  /// Create a direct register location.
  explicit MachineLocation(unsigned R) : IsRegister(true), Register(R) {}
  /// Create a register-indirect location with an offset.
  MachineLocation(unsigned R, int O) : Register(R), Offset(O) {}

  bool operator==(const MachineLocation &Other) const {
      return IsRegister == Other.IsRegister && Register == Other.Register &&
        Offset == Other.Offset;
  }

  // Accessors.
  /// \return true iff this is a register-indirect location.
  bool isIndirect()      const { return !IsRegister; }
  bool isReg()           const { return IsRegister; }
  unsigned getReg()      const { return Register; }
  int getOffset()        const { return Offset; }
  void setIsRegister(bool Is)  { IsRegister = Is; }
  void setRegister(unsigned R) { Register = R; }
  void setOffset(int O)        { Offset = O; }

  /// Make this location a direct register location.
  void set(unsigned R) {
    IsRegister = true;
    Register = R;
    Offset = 0;
  }

  /// Make this location a register-indirect+offset location.
  void set(unsigned R, int O) {
    IsRegister = false;
    Register = R;
    Offset = O;
  }
};

inline bool operator!=(const MachineLocation &LHS, const MachineLocation &RHS) {
  return !(LHS == RHS);
}

} // end namespace llvm

#endif // LLVM_MC_MACHINELOCATION_H
