//=----------- ELFLinkGraphBuilder.cpp - ELF LinkGraph builder ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic ELF LinkGraph buliding code.
//
//===----------------------------------------------------------------------===//

#include "ELFLinkGraphBuilder.h"

#define DEBUG_TYPE "jitlink"

namespace llvm {
namespace jitlink {

ELFLinkGraphBuilderBase::~ELFLinkGraphBuilderBase() {}

} // end namespace jitlink
} // end namespace llvm
