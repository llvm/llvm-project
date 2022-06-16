//===- PluginCAS.cpp --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/CASDB.h"

using namespace llvm;
using namespace llvm::cas;

Expected<std::unique_ptr<CASDB>>
cas::createPluginCAS(StringRef, ArrayRef<std::string>) {
  report_fatal_error("CAS plugins not implemented");
}
