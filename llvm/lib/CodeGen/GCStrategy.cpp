//===- GCStrategy.cpp - Garbage Collector Description ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the policy object GCStrategy which describes the
// behavior of a given garbage collector.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GCStrategy.h"

using namespace llvm;

LLVM_INSTANTIATE_REGISTRY(GCRegistry)

GCStrategy::GCStrategy() = default;
