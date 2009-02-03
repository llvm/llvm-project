//===--- Tool.cpp - The LLVM Compiler Driver --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open
// Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Tool base class - implementation details.
//
//===----------------------------------------------------------------------===//

#include "llvm/CompilerDriver/Tool.h"

#include "llvm/System/Path.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;
using namespace llvmc;

extern cl::opt<std::string> OutputFilename;

namespace {
  sys::Path MakeTempFile(const sys::Path& TempDir, const std::string& BaseName,
                         const std::string& Suffix) {
    sys::Path Out;

    // Make sure we don't end up with path names like '/file.o' if the
    // TempDir is empty.
    if (TempDir.empty()) {
      Out.set(BaseName);
    }
    else {
      Out = TempDir;
      Out.appendComponent(BaseName);
    }
    Out.appendSuffix(Suffix);
    // NOTE: makeUnique always *creates* a unique temporary file,
    // which is good, since there will be no races. However, some
    // tools do not like it when the output file already exists, so
    // they have to be placated with -f or something like that.
    Out.makeUnique(true, NULL);
    return Out;
  }
}

sys::Path Tool::OutFilename(const sys::Path& In,
                            const sys::Path& TempDir,
                            bool StopCompilation,
                            const char* OutputSuffix) const {
  sys::Path Out;

  if (StopCompilation) {
    if (!OutputFilename.empty()) {
      Out.set(OutputFilename);
    }
    else if (IsJoin()) {
      Out.set("a");
      Out.appendSuffix(OutputSuffix);
    }
    else {
      Out.set(In.getBasename());
      Out.appendSuffix(OutputSuffix);
    }
  }
  else {
    if (IsJoin())
      Out = MakeTempFile(TempDir, "tmp", OutputSuffix);
    else
      Out = MakeTempFile(TempDir, In.getBasename(), OutputSuffix);
  }
  return Out;
}
