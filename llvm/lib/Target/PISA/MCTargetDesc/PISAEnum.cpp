//===-- PISAEnum.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/PISAEnum.h"

namespace llvm {
namespace PISA {

#define GET_BoolOptionTable_IMPL
#define GET_EnumOptionTable_IMPL
#include "PISAGenSearchableTables.inc"

} // namespace PISA
} // namespace llvm
