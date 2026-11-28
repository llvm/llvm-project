//===- HandleLookupInJITDylib.h - Lookups in JITDylibs ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Protocol-agnostic call-controller handler for executor-initiated symbol
// lookups in JITDylibs. See HandleLookupInJITDylibSPS.h for the Simple Packed
// Serialization spec used to bind it.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_HANDLELOOKUPINJITDYLIB_H
#define LLVM_EXECUTIONENGINE_ORC_HANDLELOOKUPINJITDYLIB_H

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/Shared/JITDylibLookupFlags.h"
#include "llvm/ExecutionEngine/Orc/SymbolLookupSet.h"
#include "llvm/Support/Compiler.h"

#include <utility>
#include <vector>

namespace llvm::orc {

/// Maps an executor-supplied key to the JITDylib that it identifies, returning
/// null if the key identifies no JITDylib. May be called concurrently.
using GetJITDylibForKeyFn = unique_function<JITDylibSP(ExecutorAddr Key)>;

/// Handler for executor-initiated lookups in JITDylibs. The executor supplies a
/// key identifying the JITDylib, the JITDylibLookupFlags for the search, and
/// the set of symbols to look up (each with a flag that is true if the symbol
/// is required), and receives one result per element of the set, in the same
/// order.
using HandleLookupInJITDylibFn = unique_function<void(
    unique_function<void(Expected<SymbolLookupResult>)> SendResult,
    ExecutorAddr Key, JITDylibLookupFlags JDLookupFlags,
    const std::vector<std::pair<StringRef, bool>> &Symbols)>;

/// Create a handler for executor-initiated lookups in JITDylibs, callable via
/// the ORC runtime's call-controller mechanism. Bind it to the tag with
/// bindCallControllerHandler, e.g.:
///
/// \code{.cpp}
///   ES.registerCallControllerHandlers(
///       JD, bindCallControllerHandler<sps::HandleLookupInJITDylibSpec>(
///               createLookupInJITDylibHandler(ES, std::move(GetJD))));
/// \endcode
///
/// GetJD maps the key supplied by the executor to a JITDylib. What the key is
/// (e.g. the address of an image header in the executor) is up to whoever
/// maintains that mapping.
///
/// Symbol names are looked up as given (i.e. as linker-mangled names), among
/// all of the JITDylib's symbols or only its exported symbols, as the
/// executor's JITDylibLookupFlags specify. A missing required symbol fails the
/// whole lookup, without materializing anything. A missing weakly referenced
/// symbol is reported as nullopt, so a null address in the result always means
/// a symbol defined at address zero.
LLVM_ABI HandleLookupInJITDylibFn
createLookupInJITDylibHandler(ExecutionSession &ES, GetJITDylibForKeyFn GetJD);

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_HANDLELOOKUPINJITDYLIB_H
