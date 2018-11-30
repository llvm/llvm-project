//===- llvm/lib/CodeGen/AsmPrinter/BTFDebugStub.cpp -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file is a stub for BTF generation support. The real implementation
/// is at BTFDebug.cpp which will be included if BPF target is built.
///
//===----------------------------------------------------------------------===//

#include "DebugHandlerBase.h"

namespace llvm {

/// Stub class for emitting the BTF .
class BTFDebug : public DebugHandlerBase {
public:
  BTFDebug(AsmPrinter *AP);
};

BTFDebug::BTFDebug(AsmPrinter *AP) : DebugHandlerBase(AP) {}

} // end namespace llvm
