//===- ConfigManager.h ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_OBJCOPY_CONFIGMANAGER_H
#define LLVM_TOOLS_LLVM_OBJCOPY_CONFIGMANAGER_H

#include "COFF/COFFConfig.h"
#include "CommonConfig.h"
#include "ELF/ELFConfig.h"
#include "MachO/MachOConfig.h"
#include "MultiFormatConfig.h"
#include "wasm/WasmConfig.h"
#include "llvm/Support/Allocator.h"
#include <vector>

namespace llvm {
namespace objcopy {

// ConfigManager keeps all configurations and prepare
// format-specific options.
struct ConfigManager : public MultiFormatConfig {
  virtual ~ConfigManager() {}

  const CommonConfig &getCommonConfig() const override { return Common; }
  Expected<const ELFConfig &> getELFConfig() const override;
  Expected<const COFFConfig &> getCOFFConfig() const override;
  Expected<const MachOConfig &> getMachOConfig() const override;
  Expected<const WasmConfig &> getWasmConfig() const override;

  // All configs.
  CommonConfig Common;
  ELFConfig ELF;
  COFFConfig COFF;
  MachOConfig MachO;
  WasmConfig Wasm;
};

// Configuration for the overall invocation of this tool. When invoked as
// objcopy, will always contain exactly one CopyConfig. When invoked as strip,
// will contain one or more CopyConfigs.
struct DriverConfig {
  SmallVector<ConfigManager, 1> CopyConfigs;
  BumpPtrAllocator Alloc;
};

// ParseObjcopyOptions returns the config and sets the input arguments. If a
// help flag is set then ParseObjcopyOptions will print the help messege and
// exit. ErrorCallback is used to handle recoverable errors. An Error returned
// by the callback aborts the parsing and is then returned by this function.
Expected<DriverConfig>
parseObjcopyOptions(ArrayRef<const char *> ArgsArr,
                    llvm::function_ref<Error(Error)> ErrorCallback);

// ParseInstallNameToolOptions returns the config and sets the input arguments.
// If a help flag is set then ParseInstallNameToolOptions will print the help
// messege and exit.
Expected<DriverConfig>
parseInstallNameToolOptions(ArrayRef<const char *> ArgsArr);

// ParseBitcodeStripOptions returns the config and sets the input arguments.
// If a help flag is set then ParseBitcodeStripOptions will print the help
// messege and exit.
Expected<DriverConfig> parseBitcodeStripOptions(ArrayRef<const char *> ArgsArr);

// ParseStripOptions returns the config and sets the input arguments. If a
// help flag is set then ParseStripOptions will print the help messege and
// exit. ErrorCallback is used to handle recoverable errors. An Error returned
// by the callback aborts the parsing and is then returned by this function.
Expected<DriverConfig>
parseStripOptions(ArrayRef<const char *> ArgsArr,
                  llvm::function_ref<Error(Error)> ErrorCallback);
} // namespace objcopy
} // namespace llvm

#endif // LLVM_TOOLS_LLVM_OBJCOPY_CONFIGMANAGER_H
