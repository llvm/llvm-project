//===- ModuleDebugUnknownFragment.h -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_MODULEDEBUGUNKNOWNFRAGMENT_H
#define LLVM_DEBUGINFO_CODEVIEW_MODULEDEBUGUNKNOWNFRAGMENT_H

#include "llvm/DebugInfo/CodeView/ModuleDebugFragment.h"
#include "llvm/Support/BinaryStreamRef.h"

namespace llvm {
namespace codeview {

class ModuleDebugUnknownFragmentRef final : public ModuleDebugFragmentRef {
public:
  ModuleDebugUnknownFragmentRef(ModuleDebugFragmentKind Kind,
                                BinaryStreamRef Data)
      : ModuleDebugFragmentRef(Kind), Data(Data) {}

  BinaryStreamRef getData() const { return Data; }

private:
  BinaryStreamRef Data;
};
}
}

#endif
