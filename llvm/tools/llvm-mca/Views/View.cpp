//===----------------------- View.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines the virtual anchor method in View.h to pin the vtable.
///
//===----------------------------------------------------------------------===//

#include "Views/View.h"

namespace llvm {
namespace mca {

void View::anchor() {}

StringRef InstructionView::printInstructionString(const llvm::MCInst &MCI) const {
    InstructionString = "";
    MCIP.printInst(&MCI, 0, "", STI, InstrStream);
    InstrStream.flush();
    // Remove any tabs or spaces at the beginning of the instruction.
    return StringRef(InstructionString).ltrim();
  }
} // namespace mca
} // namespace llvm
