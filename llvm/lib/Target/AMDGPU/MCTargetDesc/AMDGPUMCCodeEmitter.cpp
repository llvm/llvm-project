//===-- AMDGPUCodeEmitter.cpp - AMDGPU Code Emitter interface -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// CodeEmitter interface for SI codegen.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUMCCodeEmitter.h"

using namespace llvm;

// pin vtable to this file
void AMDGPUMCCodeEmitter::anchor() {}

