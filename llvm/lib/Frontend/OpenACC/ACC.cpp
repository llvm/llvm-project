//===- ACC.cpp ------ Collection of helpers for OpenACC -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/OpenACC/ACC.h.inc"

#include "llvm/ADT/StringSwitch.h"

using namespace llvm;
using namespace acc;

#define GEN_DIRECTIVES_IMPL
#include "llvm/Frontend/OpenACC/ACC.inc"
