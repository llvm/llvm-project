//===- TestOps.h - MLIR Test Dialect Operations ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares all ops of TestOps.td. Each op partition has its own header, which
// declares the ops of that partition and forward declares the others.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TESTOPS_H
#define MLIR_TESTOPS_H

#include "TestOpsAttributes.h"
#include "TestOpsConversion.h"
#include "TestOpsCore.h"
#include "TestOpsInterfaces.h"
#include "TestOpsPatterns.h"
#include "TestOpsProperties.h"
#include "TestOpsTraits.h"

#endif // MLIR_TESTOPS_H
