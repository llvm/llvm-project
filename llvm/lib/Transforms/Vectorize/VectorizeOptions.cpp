//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VectorizeOptions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Option/LibraryOptions.h"
#include "llvm/Transforms/Vectorize/LoopIdiomVectorize.h"

#define OPTIONS_STRUCT_DEFS
#include "VectorizeOptions.inc"

static llvm::opt::RegisterLibraryOptions<llvm::VectorizeOptions> Registration;
