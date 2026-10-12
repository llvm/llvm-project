//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PROFILEDATA_PROFCORRELATORKIND_H
#define LLVM_PROFILEDATA_PROFCORRELATORKIND_H

namespace llvm {

/// Whether the debug info or the profile metadata sections correlate raw
/// profile data to functions.
enum class ProfCorrelatorKind { DebugInfo, Binary };

} // namespace llvm

#endif // LLVM_PROFILEDATA_PROFCORRELATORKIND_H
