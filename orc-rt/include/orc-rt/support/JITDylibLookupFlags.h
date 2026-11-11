//===-- JITDylibLookupFlags.h -- Flags for JITDylib search ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Flags for JITDylib search.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_SUPPORT_JITDYLIBLOOKUPFLAGS_H
#define ORC_RT_SUPPORT_JITDYLIBLOOKUPFLAGS_H

namespace orc_rt {

/// Describes which of a JITDylib's symbols a lookup may match.
///
/// MatchExportedSymbolsOnly matches only the JITDylib's exported symbols.
/// MatchAllSymbols matches hidden (non-exported) symbols as well.
enum class JITDylibLookupFlags { MatchExportedSymbolsOnly, MatchAllSymbols };

} // namespace orc_rt

#endif // ORC_RT_SUPPORT_JITDYLIBLOOKUPFLAGS_H
