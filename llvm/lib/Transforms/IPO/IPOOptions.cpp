//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IPOOptions.h"
#include "llvm/Option/LibraryOptions.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/ExpandVariadics.h"

#define OPTIONS_STRUCT_DEFS
#include "IPOOptions.inc"

LLVM_REGISTER_LIBRARY_OPTIONS(llvm::IPOOptions)
