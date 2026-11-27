//===- DeviceAddressSanitizer.h - device address sanitizer ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_DEVICEADDRESSSANITIZER_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_DEVICEADDRESSSANITIZER_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

struct DeviceAddressSanitizerOptions {
  bool Recover = false;
};

struct DeviceAddressSanitizerPass
    : public RequiredPassInfoMixin<DeviceAddressSanitizerPass> {
  DeviceAddressSanitizerPass() = default;
  explicit DeviceAddressSanitizerPass(DeviceAddressSanitizerOptions Options)
      : Options(Options) {}

  LLVM_ABI PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);

private:
  DeviceAddressSanitizerOptions Options;
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_INSTRUMENTATION_DEVICEADDRESSSANITIZER_H
