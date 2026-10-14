//===--- SymbolLookupSetSPS.h - SPS wire format for lookups -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The SPS wire format for symbol lookup sets passed between the controller and
// the executor. The controller serializes ORC's SymbolLookupSet to it (see
// SymbolLookupSet.h), and the executor deserializes it as a
// RemoteSymbolLookupSet (see TargetProcess/RemoteSymbolLookupSet.h).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SHARED_SYMBOLLOOKUPSETSPS_H
#define LLVM_EXECUTIONENGINE_ORC_SHARED_SYMBOLLOOKUPSETSPS_H

#include "llvm/ExecutionEngine/Orc/Shared/SimplePackedSerialization.h"

namespace llvm::orc::shared {

/// A symbol name, and whether the symbol is required (true) or weakly
/// referenced (false).
using SPSSymbolLookupSetElement = SPSTuple<SPSString, bool>;

using SPSSymbolLookupSet = SPSSequence<SPSSymbolLookupSetElement>;

} // namespace llvm::orc::shared

#endif // LLVM_EXECUTIONENGINE_ORC_SHARED_SYMBOLLOOKUPSETSPS_H
