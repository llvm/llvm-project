#ifndef LLVM_LIB_TARGET_SC32_SC32PASSCONFIG_H
#define LLVM_LIB_TARGET_SC32_SC32PASSCONFIG_H

#include "llvm/CodeGen/TargetPassConfig.h"

namespace llvm {

class SC32PassConfig : public TargetPassConfig {
public:
  using TargetPassConfig::TargetPassConfig;

  bool addInstSelector() override;
};

} // namespace llvm

#endif
