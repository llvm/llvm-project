//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the DXContainer-specific dumper for llvm-objdump.
///
//===----------------------------------------------------------------------===//

#include "llvm-objdump.h"
#include "llvm/Object/DXContainer.h"

using namespace llvm;

namespace {
class DXContainerDumper : public objdump::Dumper {
public:
  DXContainerDumper(const object::DXContainerObjectFile &Obj)
      : objdump::Dumper(Obj) {}
};
} // namespace

std::unique_ptr<objdump::Dumper> llvm::objdump::createDXContainerDumper(
    const object::DXContainerObjectFile &Obj) {
  return std::make_unique<DXContainerDumper>(Obj);
}
