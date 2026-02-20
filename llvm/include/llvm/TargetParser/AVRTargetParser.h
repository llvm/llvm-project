//===-- AVRTargetParser - Parser for AVR target features ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a target parser to recognise AVR hardware features.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGETPARSER_AVRTARGETPARSER_H
#define LLVM_TARGETPARSER_AVRTARGETPARSER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <string>

namespace llvm {
namespace AVR {

LLVM_ABI Expected<std::string> getFeatureSetFromEFlag(const unsigned EFlag);

} // namespace AVR
} // namespace llvm
#endif
