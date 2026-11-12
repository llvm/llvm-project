//===- GenInfo.cpp - Generator info -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/GenInfo.h"

#include "mlir/TableGen/GenNameParser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"

using namespace mlir;

static llvm::ManagedStatic<std::vector<GenInfo>> generatorRegistry;

GenRegistration::GenRegistration(StringRef arg, StringRef description,
                                 const GenFunction &function) {
  generatorRegistry->emplace_back(arg, description, function);
}

GenNameParser::GenNameParser(llvm::cl::Option &opt)
    : llvm::cl::parser<const GenInfo *>(opt) {
  for (const auto &kv : *generatorRegistry) {
    addLiteralOption(kv.getGenArgument(), &kv, kv.getGenDescription());
  }
}

void GenNameParser::printOptionInfo(const llvm::cl::Option &o,
                                    size_t globalWidth) const {
  GenNameParser *tp = const_cast<GenNameParser *>(this);
  llvm::array_pod_sort(tp->Values.begin(), tp->Values.end(),
                       [](const GenNameParser::OptionInfo *vT1,
                          const GenNameParser::OptionInfo *vT2) {
                         return vT1->Name.compare(vT2->Name);
                       });
  using llvm::cl::parser;
  parser<const GenInfo *>::printOptionInfo(o, globalWidth);
}
