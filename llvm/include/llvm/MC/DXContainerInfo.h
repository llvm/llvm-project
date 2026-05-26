//===----- llvm/MC/DXContainerInfo.h - DXContainer Info ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_DXCONTAINERINFO_H
#define LLVM_MC_DXCONTAINERINFO_H

#include "llvm/BinaryFormat/DXContainer.h"

namespace llvm {

class raw_ostream;

namespace mcdxbc {

struct DebugName {
  dxbc::DebugNameHeader Parameters;
  StringRef Filename;

  DebugName() { Parameters.Flags = 0; }
  DebugName(dxbc::DebugNameHeader &Parameters, StringRef Filename)
      : Parameters(Parameters), Filename(Filename) {}

  void setFilename(StringRef DebugFilename);
  void write(raw_ostream &OS) const;
};

struct CompilerVersion {
  dxbc::CompilerVersionHeader Parameters;
  StringRef CommitSha;
  StringRef CustomVersionString;

  CompilerVersion();

  void setCommitSha(StringRef CommitSha);
  void setVersionString(StringRef VersionString);
  void write(raw_ostream &OS) const;

private:
  void updateContentSize();
};

} // namespace mcdxbc
} // namespace llvm

#endif // LLVM_MC_DXCONTAINERINFO_H
