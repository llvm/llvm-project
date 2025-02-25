//===---- GetTapiInterface.h -- Get interface from TAPI file ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Get symbol interface from TAPI file.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_GETTAPIINTERFACE_H
#define LLVM_EXECUTIONENGINE_ORC_GETTAPIINTERFACE_H

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/Object/TapiUniversal.h"

namespace llvm::orc {

/// Returns a SymbolNameSet containing the exported symbols defined in the
/// relevant slice of the TapiUniversal file.
Expected<SymbolNameSet> getInterfaceFromTapiFile(ExecutionSession &ES,
                                                 object::TapiUniversal &TU);

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_GETTAPIINTERFACE_H
