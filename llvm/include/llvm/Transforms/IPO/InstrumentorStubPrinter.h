//===- Transforms/IPO/InstrumentorStubPrinter.h --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A generator of Instrumentor's runtime stub.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_INSTRUMENTOR_STUB_PRINTER_H
#define LLVM_TRANSFORMS_IPO_INSTRUMENTOR_STUB_PRINTER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/IPO/Instrumentor.h"

namespace llvm {
namespace instrumentor {

/// Print a runtime stub file with the enabled instrumentation opportunities.
void printRuntimeStub(const InstrumentationConfig &IConf,
                      StringRef StubRuntimeName, const Module &M);

} // end namespace instrumentor
} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_INSTRUMENTOR_STUB_PRINTER_H
