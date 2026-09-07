//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the classes that describe the Fortran types.
///
//===----------------------------------------------------------------------===//

#include "FortranTypes.h"

using namespace lldb_private;
using namespace lldb_private::plugin::fortran;

char FortranType::ID;

FortranType::~FortranType() = default;