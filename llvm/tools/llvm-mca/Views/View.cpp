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
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSubtargetInfo.h"

namespace llvm {
namespace mca {

void View::anchor() {}

void View::printViewJSON(llvm::raw_ostream &OS) {
  json::Object JO;
  JO.try_emplace(getNameAsString().str(), toJSON());
  OS << formatv("{0:2}", json::Value(std::move(JO))) << "\n";
}


} // namespace mca
} // namespace llvm
