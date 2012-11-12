//===-- llvm/MC/MachineLocation.h -------------------------------*- C++ -*-===//
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
// using an offset of zero.
//
// The MachineMove class is used to represent abstract move operations in the
// prolog/epilog of a compiled function.  A collection of these objects can be
// used by a debug consumer to track the location of values when unwinding stack
// frames.
//===----------------------------------------------------------------------===//


#ifndef LLVM_MC_MACHINELOCATION_H
#define LLVM_MC_MACHINELOCATION_H

namespace llvm {
  class MCSymbol;

class MachineLocation {
private:
  bool IsRegister;                      // True if location is a register.
  unsigned Register;                    // gcc/gdb register number.
  int Offset;                           // Displacement if not register.
public:
  enum {
    // The target register number for an abstract frame pointer. The value is
    // an arbitrary value that doesn't collide with any real target register.
    VirtualFP = ~0U
  };
  MachineLocation()
    : IsRegister(false), Register(0), Offset(0) {}
  explicit MachineLocation(unsigned R)
    : IsRegister(true), Register(R), Offset(0) {}
  MachineLocation(unsigned R, int O)
    : IsRegister(false), Register(R), Offset(O) {}

  bool operator==(const MachineLocation &Other) const {
      return IsRegister == Other.IsRegister && Register == Other.Register &&
        Offset == Other.Offset;
  }

  // Accessors
  bool isReg()           const { return IsRegister; }
  unsigned getReg()      const { return Register; }
  int getOffset()        const { return Offset; }
  void setIsRegister(bool Is)  { IsRegister = Is; }
  void setRegister(unsigned R) { Register = R; }
  void setOffset(int O)        { Offset = O; }
  void set(unsigned R) {
    IsRegister = true;
    Register = R;
    Offset = 0;
  }
  void set(unsigned R, int O) {
    IsRegister = false;
    Register = R;
    Offset = O;
  }

#ifndef NDEBUG
  void dump();
#endif
};

/// MachineMove - This class represents the save or restore of a callee saved
/// register that exception or debug info needs to know about.
class MachineMove {
private:
  /// Label - Symbol for post-instruction address when result of move takes
  /// effect.
  MCSymbol *Label;

  // Move to & from location.
  MachineLocation Destination, Source;
public:
  MachineMove() : Label(0) {}

  MachineMove(MCSymbol *label, const MachineLocation &D,
              const MachineLocation &S)
  : Label(label), Destination(D), Source(S) {}

  // Accessors
  MCSymbol *getLabel()                    const { return Label; }
  const MachineLocation &getDestination() const { return Destination; }
  const MachineLocation &getSource()      const { return Source; }
};

} // End llvm namespace

#endif
