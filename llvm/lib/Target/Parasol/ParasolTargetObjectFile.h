//===-- ParasolTargetObjectFile.h - Parasol Object Info -------------------===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PARASOL_PARASOLTARGETOBJECTFILE_H
#define LLVM_LIB_TARGET_PARASOL_PARASOLTARGETOBJECTFILE_H

#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"

namespace llvm {
class ParasolTargetObjectFile : public TargetLoweringObjectFileELF {
public:
  void Initialize(MCContext &Ctx, const TargetMachine &TM) override;
};
} // end namespace llvm

#endif // end LLVM_LIB_TARGET_PARASOL_PARASOLTARGETOBJECTFILE_H
