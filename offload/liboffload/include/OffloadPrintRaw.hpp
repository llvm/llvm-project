//===- OffloadPrintRaw.hpp - Offload raw_ostream printing -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <llvm/Support/raw_ostream.h>

namespace offload::detail {
using print_ostream = llvm::raw_ostream;
}

#include "detail/OffloadPrintGeneric.inc"
