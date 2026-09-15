//===- TargetTriple.cpp - Windows target triple detection -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "orc-rt-internal/bedrock/sys/TargetTriple.h"

namespace orc_rt::sys {

std::string detectTargetTriple() noexcept { return {}; }

} // namespace orc_rt::sys
