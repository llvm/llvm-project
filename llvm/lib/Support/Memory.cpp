//===- Memory.cpp - Memory Handling Support ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines some helpful functions for allocating memory and dealing
// with memory mapped files
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Memory.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Valgrind.h"

// Include the platform-specific parts of this class.
#ifdef LLVM_ON_UNIX
#include "Unix/Memory.inc"
#endif
#ifdef _WIN32
#include "Windows/Memory.inc"
#endif
