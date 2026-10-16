//===- AccAPI.cpp - OpenACC runtime API entry points -------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Debug.h"
#include "ISO_Fortran_binding.h"
#include "openacc.h"

namespace llvm::acc::impl::f {
#include "AccFImplDecls.inc"
} // namespace llvm::acc::impl::f
#include "AccPrint.h"

using namespace llvm::acc::target::debug;

// Entries call the impl:: declarations above, so keep this include last.
#include "AccFEntries.inc"
