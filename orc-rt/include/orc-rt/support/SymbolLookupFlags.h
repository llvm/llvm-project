//===-- SymbolLookupFlags.h -- Flags for symbol lookups ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Flags for symbol lookups.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_SUPPORT_SYMBOLLOOKUPFLAGS_H
#define ORC_RT_SUPPORT_SYMBOLLOOKUPFLAGS_H

namespace orc_rt {

/// Describes how a symbol is referenced by a lookup.
///
/// A RequiredSymbol must be present. A WeaklyReferencedSymbol may be absent: a
/// missing weakly referenced symbol is resolved to a null address, rather than
/// being treated as missing.
enum class SymbolLookupFlags { RequiredSymbol, WeaklyReferencedSymbol };

} // namespace orc_rt

#endif // ORC_RT_SUPPORT_SYMBOLLOOKUPFLAGS_H
