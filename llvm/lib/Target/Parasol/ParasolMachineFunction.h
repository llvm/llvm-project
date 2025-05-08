//=== ParasolMachineFunctionInfo.h - Private data used for Parasol --------===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//
//
// This file declares the Parasol specific subclass of MachineFunctionInfo.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PARASOL_PARASOLMACHINEFUNCTION_H
#define LLVM_LIB_TARGET_PARASOL_PARASOLMACHINEFUNCTION_H

#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {

/// ParasolFunctionInfo - This class is derived from MachineFunction private
/// Parasol target-specific information for each MachineFunction.
class ParasolFunctionInfo : public MachineFunctionInfo {
private:
  MachineFunction &MF;

public:
  ParasolFunctionInfo(MachineFunction &MF) : MF(MF) {}
};

} // end of namespace llvm

#endif // end LLVM_LIB_TARGET_PARASOL_PARASOLMACHINEFUNCTION_H
