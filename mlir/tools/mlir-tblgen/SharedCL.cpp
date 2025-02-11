//===- SharedCL.cpp - tblgen command line arguments -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SharedCL.h"

using namespace mlir;
using namespace mlir::tblgen;
namespace cl = llvm::cl;

static llvm::cl::OptionCategory clShared("Options for all -gen-*");

cl::opt<bool> mlir::tblgen::clUseFallbackTypeIDs = cl::opt<bool>(
  "use-fallback-type-ids",
  cl::desc(
      "Don't generate static TypeID decls; fall back to string comparison."),
  cl::init(false), cl::cat(clShared));
