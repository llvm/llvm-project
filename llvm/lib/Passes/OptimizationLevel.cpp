//===- OptimizationLevel.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Passes/OptimizationLevel.h"

using namespace llvm;

const OptimizationLevel OptimizationLevel::O0 = {0};
const OptimizationLevel OptimizationLevel::O1 = {1};
const OptimizationLevel OptimizationLevel::O2 = {2};
const OptimizationLevel OptimizationLevel::O3 = {3};
