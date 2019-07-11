//===-- PPCXCOFFObjectWriter.cpp - PowerPC XCOFF Writer -------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PPCMCTargetDesc.h"
#include "llvm/MC/MCXCOFFObjectWriter.h"

using namespace llvm;

namespace {
class PPCXCOFFObjectWriter : public MCXCOFFObjectTargetWriter {

public:
  PPCXCOFFObjectWriter(bool Is64Bit);
};
} // end anonymous namespace

PPCXCOFFObjectWriter::PPCXCOFFObjectWriter(bool Is64Bit)
    : MCXCOFFObjectTargetWriter(Is64Bit) {}

std::unique_ptr<MCObjectTargetWriter>
llvm::createPPCXCOFFObjectWriter(bool Is64Bit) {
  return llvm::make_unique<PPCXCOFFObjectWriter>(Is64Bit);
}
