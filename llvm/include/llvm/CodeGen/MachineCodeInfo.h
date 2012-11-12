//===-- MachineCodeInfo.h - Class used to report JIT info -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines MachineCodeInfo, a class used by the JIT ExecutionEngine
// to report information about the generated machine code.
//
// See JIT::runJITOnFunction for usage.
//
//===----------------------------------------------------------------------===//

#ifndef EE_MACHINE_CODE_INFO_H
#define EE_MACHINE_CODE_INFO_H

#include "llvm/Support/DataTypes.h"

namespace llvm {

class MachineCodeInfo {
private:
  size_t Size;   // Number of bytes in memory used
  void *Address; // The address of the function in memory

public:
  MachineCodeInfo() : Size(0), Address(0) {}

  void setSize(size_t s) {
    Size = s;
  }

  void setAddress(void *a) {
    Address = a;
  }

  size_t size() const {
    return Size;
  }

  void *address() const {
    return Address;
  }

};

}

#endif

