//===- TargetTriple.cpp - Linux target triple detection -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "orc-rt/bedrock/ExecutorProcessInfo.h"

namespace orc_rt {

std::string ExecutorProcessInfo::detectTargetTriple() noexcept { return {}; }

} // namespace orc_rt
