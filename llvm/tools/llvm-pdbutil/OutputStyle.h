//===- OutputStyle.h ------------------------------------------ *- C++ --*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMPDBDUMP_OUTPUTSTYLE_H
#define LLVM_TOOLS_LLVMPDBDUMP_OUTPUTSTYLE_H

#include "llvm/Support/Error.h"

namespace llvm {
namespace pdb {
class PDBFile;

class OutputStyle {
public:
  virtual ~OutputStyle() {}

  virtual Error dump() = 0;
};
}
}

#endif
