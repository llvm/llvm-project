#include "llvm/ABI/ABIInfo.h"
#include <memory>

#ifndef LLVM_ABI_TARGETCODEGENINFO_H
#define LLVM_ABI_TARGETCODEGENINFO_H

namespace llvm::abi {

class TargetCodeGenInfo {
  std::unique_ptr<llvm::abi::ABIInfo> Info;

protected:
  template <typename T> const T getABIInfo() const {
    return static_cast<const T &>(*Info);
  }

public:
  TargetCodeGenInfo(std::unique_ptr<llvm::abi::ABIInfo> Info);
  virtual ~TargetCodeGenInfo();

  const ABIInfo &getABIInfo() const { return *Info; }

  virtual void computeInfo(ABIFunctionInfo &FI) const;
};

std::unique_ptr<TargetCodeGenInfo>
createDefaultTargetCodeGenInfo(TypeBuilder &TB);

std::unique_ptr<TargetCodeGenInfo> createBPFTargetCodeGenInfo(TypeBuilder &TB);

std::unique_ptr<TargetCodeGenInfo>
createX8664TargetCodeGenInfo(TypeBuilder &TB);

std::unique_ptr<TargetCodeGenInfo>
createAArch64TargetCodeGenInfo(TypeBuilder &TB);

std::unique_ptr<TargetCodeGenInfo> createARMTargetCodeGenInfo(TypeBuilder &TB);

std::unique_ptr<TargetCodeGenInfo>
createRISCVTargetCodeGenInfo(TypeBuilder &TB);

std::unique_ptr<TargetCodeGenInfo>
createPPC64TargetCodeGenInfo(TypeBuilder &TB);

std::unique_ptr<TargetCodeGenInfo>
createSystemZTargetCodeGenInfo(TypeBuilder &TB);

std::unique_ptr<TargetCodeGenInfo>
createWebAssemblyTargetCodeGenInfo(TypeBuilder &TB);

std::unique_ptr<TargetCodeGenInfo>
createNVPTXTargetCodeGenInfo(TypeBuilder &TB);

std::unique_ptr<TargetCodeGenInfo>
createAMDGPUTargetCodeGenInfo(TypeBuilder &TB);
} // namespace llvm::abi

#endif
