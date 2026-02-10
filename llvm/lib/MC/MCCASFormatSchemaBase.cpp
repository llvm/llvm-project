//===- MCCASFormatSchemaBase.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCCASFormatSchemaBase.h"

using namespace llvm::mccasformats;

char MCFormatSchemaBase::ID = 0;
void MCFormatSchemaBase::anchor() {}
