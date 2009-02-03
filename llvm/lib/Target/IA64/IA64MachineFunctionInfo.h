//===-- IA64MachineFunctionInfo.h - IA64-specific information ---*- C++ -*-===//
//===--                   for MachineFunction                 ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
//===----------------------------------------------------------------------===//
//
// This file declares IA64-specific per-machine-function information.
//
//===----------------------------------------------------------------------===//

#ifndef IA64MACHINEFUNCTIONINFO_H
#define IA64MACHINEFUNCTIONINFO_H

#include "llvm/CodeGen/MachineFunction.h"
//#include "IA64JITInfo.h"

namespace llvm {

class IA64FunctionInfo : public MachineFunctionInfo {

public:
  unsigned outRegsUsed; // how many 'out' registers are used
  // by this machinefunction? (used to compute the appropriate
  // entry in the 'alloc' instruction at the top of the
  // machinefunction)
  IA64FunctionInfo(MachineFunction& MF) { outRegsUsed=0; };

};

} // End llvm namespace

#endif

