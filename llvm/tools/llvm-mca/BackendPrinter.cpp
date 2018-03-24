//===--------------------- BackendPrinter.cpp -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements the BackendPrinter interface.
///
//===----------------------------------------------------------------------===//

#include "BackendPrinter.h"
#include "View.h"
#include "llvm/CodeGen/TargetSchedule.h"

namespace mca {

using namespace llvm;

void BackendPrinter::printReport(llvm::raw_ostream &OS) const {
  for (const auto &V : Views)
    V->printView(OS);
}
} // namespace mca.
