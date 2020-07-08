//===- OMPConstants.cpp - Helpers related to OpenMP code generation ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/OpenMP/OMPConstants.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

using namespace llvm;
using namespace omp;

#include "llvm/Frontend/OpenMP/OMP.cpp.inc"
