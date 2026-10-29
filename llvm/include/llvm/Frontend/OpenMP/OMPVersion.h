//===-- OMPVersion.h - OpenMP version definition ------------------ C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_OPENMP_OMPVERSION_H
#define LLVM_FRONTEND_OPENMP_OMPVERSION_H

#include "llvm/Frontend/Directive/Version.h"

namespace llvm::omp {
using Version = llvm::directive::Version;
}

#endif // LLVM_FRONTEND_OPENMP_OMPVERSION_H
