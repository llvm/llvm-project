//===- PDBTypeServerHandler.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_PDBTYPESERVERHANDLER_H
#define LLVM_DEBUGINFO_PDB_PDBTYPESERVERHANDLER_H

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/CodeView/TypeServerHandler.h"
#include "llvm/DebugInfo/PDB/Native/NativeSession.h"
#include "llvm/DebugInfo/PDB/PDBTypes.h"

#include <memory>
#include <string>

namespace llvm {
namespace pdb {
class NativeSession;

class PDBTypeServerHandler : public codeview::TypeServerHandler {
public:
  PDBTypeServerHandler(bool RevisitAlways = false);

  void addSearchPath(StringRef Path);
  Expected<bool> handle(codeview::TypeServer2Record &TS,
                        codeview::TypeVisitorCallbacks &Callbacks) override;

private:
  Expected<bool> handleInternal(PDBFile &File,
                                codeview::TypeVisitorCallbacks &Callbacks);

  bool RevisitAlways;
  std::unique_ptr<NativeSession> Session;
  SmallVector<SmallString<64>, 4> SearchPaths;
};
}
}

#endif
