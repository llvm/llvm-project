//===- Transforms/IPO/InstrumentorConfigFile.h ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for the Instrumentor JSON configuration file.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_INSTRUMENTOR_CONFIGFILE_H
#define LLVM_TRANSFORMS_IPO_INSTRUMENTOR_CONFIGFILE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Transforms/IPO/Instrumentor.h"

namespace llvm {
namespace instrumentor {

/// Write the configuration in /p IConf to the file with path \p OutputFile.
void writeConfigToJSON(InstrumentationConfig &IConf, StringRef OutputFile,
                       LLVMContext &Ctx);

/// Read the configuration from the file with path \p InputFile  into /p IConf.
bool readConfigFromJSON(InstrumentationConfig &IConf, StringRef InputFile,
                        LLVMContext &Ctx);

} // end namespace instrumentor
} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_INSTRUMENTOR_CONFIGFILE_H
