//===--------------------- PipelinePrinter.cpp ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements the PipelinePrinter interface.
///
//===----------------------------------------------------------------------===//

#include "PipelinePrinter.h"
#include "Views/View.h"

namespace llvm {
namespace mca {

json::Object PipelinePrinter::getJSONReportRegion() const {
  json::Object JO;
  for (const auto &V : Views) {
    if (V->isSerializable()) {
      JO.try_emplace(V->getNameAsString().str(), V->toJSON());
    }
  }
  return JO;
}

void PipelinePrinter::printReport(llvm::raw_ostream &OS) const {
  json::Object JO;
  for (const auto &V : Views) {
    V->printView(OS);
  }
}
} // namespace mca.
} // namespace llvm
