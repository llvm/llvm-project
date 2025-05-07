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

void writeConfigToJSON(InstrumentationConfig &IConf, StringRef OutputFile);

bool readConfigFromJSON(InstrumentationConfig &IConf, StringRef InputFile);

} // end namespace instrumentor
} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_INSTRUMENTOR_CONFIGFILE_H
