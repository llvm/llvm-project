//===- version.go - LLVM version info -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines LLVM version information.
//
//===----------------------------------------------------------------------===//

package llvm

/*
#include "llvm/Config/llvm-config.h"
*/
import "C"

const Version = C.LLVM_VERSION_STRING
