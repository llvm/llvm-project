//===- DebugUnknownSubsection.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_DEBUGUNKNOWNSUBSECTION_H
#define LLVM_DEBUGINFO_CODEVIEW_DEBUGUNKNOWNSUBSECTION_H

#include "llvm/DebugInfo/CodeView/DebugSubsection.h"
#include "llvm/Support/BinaryStreamRef.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
namespace codeview {

class LLVM_CLASS_ABI DebugUnknownSubsectionRef final : public DebugSubsectionRef {
public:
  DebugUnknownSubsectionRef(DebugSubsectionKind Kind, BinaryStreamRef Data)
      : DebugSubsectionRef(Kind), Data(Data) {}

  BinaryStreamRef getData() const { return Data; }

private:
  BinaryStreamRef Data;
};
}
}

#endif
