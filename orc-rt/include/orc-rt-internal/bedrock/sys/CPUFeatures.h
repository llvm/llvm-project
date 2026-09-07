//===- CPUFeatures.h - Host CPU feature detection ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Exactly one implementation is compiled into the runtime, chosen by the
// build: see lib/bedrock/sys/darwin/CPUFeatures.cpp and its siblings.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_INTERNAL_BEDROCK_SYS_CPUFEATURES_H
#define ORC_RT_INTERNAL_BEDROCK_SYS_CPUFEATURES_H

#include <string_view>
#include <vector>

namespace orc_rt::sys {

/// Returns the host's CPU feature names, using LLVM's SubtargetFeatures
/// naming (see orc-rt-internal/bedrock/TargetDetails.h).
std::vector<std::string_view> detectTargetCPUFeatures();

} // namespace orc_rt::sys

#endif // ORC_RT_INTERNAL_BEDROCK_SYS_CPUFEATURES_H
