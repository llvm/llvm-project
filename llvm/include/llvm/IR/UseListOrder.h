//===- llvm/IR/UseListOrder.h - LLVM Use List Order -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file has structures and command-line options for preserving use-list
// order.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_USELISTORDER_H
#define LLVM_IR_USELISTORDER_H

#include "llvm/Support/CommandLine.h"
#include <cstddef>
#include <vector>

namespace llvm {

class Function;
class Value;

/// Structure to hold a use-list order.
struct UseListOrder {
  const Value *V = nullptr;
  const Function *F = nullptr;
  std::vector<unsigned> Shuffle;

  UseListOrder(const Value *V, const Function *F, size_t ShuffleSize)
      : V(V), F(F), Shuffle(ShuffleSize) {}

  UseListOrder() = default;
  UseListOrder(UseListOrder &&) = default;
  UseListOrder &operator=(UseListOrder &&) = default;
};

using UseListOrderStack = std::vector<UseListOrder>;

class PreserveUseListOrderOptionParser
    : public cl::parser<std::optional<bool>> {
public:
  PreserveUseListOrderOptionParser(cl::Option &O)
      : cl::parser<std::optional<bool>>(O) {}

  bool parse(cl::Option &O, StringRef ArgName, StringRef Arg,
             std::optional<bool> &V) {
    if (Arg == "" || Arg == "true" || Arg == "TRUE" || Arg == "True" ||
        Arg == "1") {
      V = true;
      return false;
    }
    if (Arg == "false" || Arg == "FALSE" || Arg == "False" || Arg == "0") {
      V = false;
      return false;
    }
    return O.error("Invalid argument '" + Arg + "',  Try 0 or 1");
  }
};

} // end namespace llvm

#endif // LLVM_IR_USELISTORDER_H
