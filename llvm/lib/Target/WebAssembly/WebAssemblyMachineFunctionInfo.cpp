//=- WebAssemblyMachineFunctionInfo.cpp - WebAssembly Machine Function Info -=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements WebAssembly-specific per-machine-function
/// information.
///
//===----------------------------------------------------------------------===//

#include "WebAssemblyMachineFunctionInfo.h"
using namespace llvm;

WebAssemblyFunctionInfo::~WebAssemblyFunctionInfo() {}
