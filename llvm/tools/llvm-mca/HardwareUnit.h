//===-------------------------- HardwareUnit.h ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines a base class for describing a simulated hardware
/// unit.  These units are used to construct a simulated backend.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_MCA_HARDWAREUNIT_H
#define LLVM_TOOLS_LLVM_MCA_HARDWAREUNIT_H

namespace mca {

class HardwareUnit {
  HardwareUnit(const HardwareUnit &H) = delete;
  HardwareUnit &operator=(const HardwareUnit &H) = delete;

public:
  HardwareUnit() = default;
  virtual ~HardwareUnit();
};

} // namespace mca
#endif // LLVM_TOOLS_LLVM_MCA_HARDWAREUNIT_H
