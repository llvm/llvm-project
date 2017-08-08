//===- llvm/unittest/Support/DynamicLibrary/DynamicLibraryLib.cpp ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PipSqueak.h"

#ifndef PIPSQUEAK_TESTA_RETURN
#define PIPSQUEAK_TESTA_RETURN "ProcessCall"
#endif

extern "C" PIPSQUEAK_EXPORT const char *TestA() { return PIPSQUEAK_TESTA_RETURN; }
