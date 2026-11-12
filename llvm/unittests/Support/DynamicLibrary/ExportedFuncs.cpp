//===- llvm/unittest/Support/DynamicLibrary/ExportedFuncs.cpp -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PipSqueak.h"

#ifndef PIPSQUEAK_TESTA_RETURN
#define PIPSQUEAK_TESTA_RETURN "ProcessCall"
#endif

extern "C" PIPSQUEAK_EXPORT const char *TestA() { return PIPSQUEAK_TESTA_RETURN; }
