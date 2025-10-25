//===----- TargetCodeGenInfo.h ------------------------------------ C++ ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "llvm/ABI/ABIInfo.h"
#include "llvm/TargetParser/Triple.h"
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
  TargetCodeGenInfo(std::unique_ptr<llvm::abi::ABIInfo> Info)
      : Info(std::move(Info)) {}

  virtual ~TargetCodeGenInfo() = default;

  const ABIInfo &getABIInfo() const { return *Info; }
};

std::unique_ptr<TargetCodeGenInfo>
createDefaultTargetCodeGenInfo(TypeBuilder &TB);

std::unique_ptr<TargetCodeGenInfo> createBPFTargetCodeGenInfo(TypeBuilder &TB);

/// The AVX ABI level for X86 targets.
enum class X86AVXABILevel {
  None,
  AVX,
  AVX512,
};

std::unique_ptr<TargetCodeGenInfo>
createX8664TargetCodeGenInfo(TypeBuilder &TB, const Triple &Triple,
                             X86AVXABILevel AVXLevel, bool Has64BitPointers,
                             const ABICompatInfo &Compat);
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
