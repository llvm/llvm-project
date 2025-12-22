//===- tools/dsymutil/SwiftModule.h ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TOOLS_DSYMUTIL_SWIFTMODULE_H
#define LLVM_TOOLS_DSYMUTIL_SWIFTMODULE_H

#include "llvm/Support/Error.h"

llvm::Expected<bool> IsBuiltFromSwiftInterface(llvm::StringRef data);

#endif
