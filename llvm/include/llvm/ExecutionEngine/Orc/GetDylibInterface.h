//===---- GetDylibInterface.h - Get interface for real dylib ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Get symbol interface from a real dynamic library or TAPI file. These
// interfaces can be used to simulate weak linking (ld64 -weak-lx /
// -weak_library) against a library that is absent at runtime.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_GETDYLIBINTERFACE_H
#define LLVM_EXECUTIONENGINE_ORC_GETDYLIBINTERFACE_H

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/Support/Compiler.h"

namespace llvm::orc {

/// Returns a SymbolNameSet containing the exported symbols defined in the
/// given dylib.
LLVM_ABI Expected<SymbolNameSet>
getDylibInterfaceFromDylib(ExecutionSession &ES, Twine Path);

/// Returns a SymbolNameSet containing the exported symbols defined in the
/// relevant slice of the TapiUniversal file.
LLVM_ABI Expected<SymbolNameSet>
getDylibInterfaceFromTapiFile(ExecutionSession &ES, Twine Path);

/// Returns a SymbolNameSet containing the exported symbols defined in the
/// relevant slice of the given file, which may be either a dylib or a tapi
/// file.
LLVM_ABI Expected<SymbolNameSet> getDylibInterface(ExecutionSession &ES,
                                                   Twine Path);

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_GETDYLIBINTERFACE_H
