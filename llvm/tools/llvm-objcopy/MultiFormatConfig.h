//===- MultiFormatConfig.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_OBJCOPY_MULTIFORMATCONFIG_H
#define LLVM_TOOLS_LLVM_OBJCOPY_MULTIFORMATCONFIG_H

#include "llvm/Support/Error.h"

namespace llvm {
namespace objcopy {

struct CommonConfig;
struct ELFConfig;
struct COFFConfig;
struct MachOConfig;
struct WasmConfig;

class MultiFormatConfig {
public:
  virtual ~MultiFormatConfig() {}

  virtual const CommonConfig &getCommonConfig() const = 0;
  virtual Expected<const ELFConfig &> getELFConfig() const = 0;
  virtual Expected<const COFFConfig &> getCOFFConfig() const = 0;
  virtual Expected<const MachOConfig &> getMachOConfig() const = 0;
  virtual Expected<const WasmConfig &> getWasmConfig() const = 0;
};

} // namespace objcopy
} // namespace llvm

#endif // LLVM_TOOLS_LLVM_OBJCOPY_MULTIFORMATCONFIG_H
