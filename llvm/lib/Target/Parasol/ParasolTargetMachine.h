//===-- ParasolTargetMachine.h - Define TargetMachine for Parasol ---------===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//
//
// This file declares the Parasol specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PARASOL_PARASOLTARGETMACHINE_H
#define LLVM_LIB_TARGET_PARASOL_PARASOLTARGETMACHINE_H

#include "ParasolSubtarget.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/Target/TargetMachine.h"
#include <optional>

namespace llvm {
class ParasolTargetMachine : public LLVMTargetMachine {
  std::unique_ptr<TargetLoweringObjectFile> TLOF;
  mutable StringMap<std::unique_ptr<ParasolSubtarget>> SubtargetMap;

public:
  ParasolTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                       StringRef FS, const TargetOptions &Options,
                       std::optional<Reloc::Model> RM,
                       std::optional<CodeModel::Model> CM, CodeGenOptLevel OL,
                       bool JIT);

  const ParasolSubtarget *getSubtargetImpl(const Function &F) const override;
  const ParasolSubtarget *getSubtargetImpl() const = delete;

  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;

  TargetLoweringObjectFile *getObjFileLowering() const override {
    return TLOF.get();
  }
};
} // namespace llvm

#endif // end LLVM_LIB_TARGET_PARASOL_PARASOLTARGETMACHINE_H
