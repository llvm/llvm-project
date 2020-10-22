//===- CfgTraits.cpp - Traits for generically working on CFGs ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CfgTraits.h"

using namespace llvm;

void CfgInterface::anchor() {}
void CfgPrinter::anchor() {}
