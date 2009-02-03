//===-- SPUMachineFunctionInfo.h - Private data used for CellSPU --*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the IBM Cell SPU specific subclass of MachineFunctionInfo.
//
//===----------------------------------------------------------------------===//

#ifndef SPU_MACHINE_FUNCTION_INFO_H
#define SPU_MACHINE_FUNCTION_INFO_H

#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {

/// SPUFunctionInfo - Cell SPU target-specific information for each
/// MachineFunction
class SPUFunctionInfo : public MachineFunctionInfo {
private:
  /// UsesLR - Indicates whether LR is used in the current function.
  ///
  bool UsesLR;

public:
  SPUFunctionInfo(MachineFunction& MF) 
  : UsesLR(false)
  {}

  void setUsesLR(bool U) { UsesLR = U; }
  bool usesLR()          { return UsesLR; }

};

} // end of namespace llvm


#endif

