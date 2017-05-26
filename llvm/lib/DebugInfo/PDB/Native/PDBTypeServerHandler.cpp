//===- PDBTypeServerHandler.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Handles CodeView LF_TYPESERVER2 records by attempting to locate a matching
// PDB file, then loading the PDB file and visiting all types from the
// referenced PDB using the original supplied visitor.
//
// The net effect of this is that when visiting a PDB containing a TypeServer
// record, the TypeServer record is "replaced" with all of the records in
// the referenced PDB file.  If a single instance of PDBTypeServerHandler
// encounters the same TypeServer multiple times (for example reusing one
// PDBTypeServerHandler across multiple visitations of distinct object files or
// PDB files), PDBTypeServerHandler will optionally revisit all the records
// again, or simply consume the record and do nothing.
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/PDBTypeServerHandler.h"

#include "llvm/DebugInfo/CodeView/CVTypeVisitor.h"
#include "llvm/DebugInfo/CodeView/CodeViewError.h"
#include "llvm/DebugInfo/PDB/GenericError.h"
#include "llvm/DebugInfo/PDB/Native/InfoStream.h"
#include "llvm/DebugInfo/PDB/Native/NativeSession.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/DebugInfo/PDB/Native/TpiStream.h"
#include "llvm/DebugInfo/PDB/PDB.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::pdb;

static void ignoreErrors(Error EC) {
  llvm::handleAllErrors(std::move(EC), [&](ErrorInfoBase &EIB) {});
}

PDBTypeServerHandler::PDBTypeServerHandler(bool RevisitAlways)
    : RevisitAlways(RevisitAlways) {}

void PDBTypeServerHandler::addSearchPath(StringRef Path) {
  if (Path.empty() || !sys::fs::is_directory(Path))
    return;

  SearchPaths.insert(Path);
}

Expected<bool>
PDBTypeServerHandler::handleInternal(PDBFile &File,
                                     TypeVisitorCallbacks &Callbacks) {
  auto ExpectedTpi = File.getPDBTpiStream();
  if (!ExpectedTpi)
    return ExpectedTpi.takeError();

  // For handling a type server, we should be using whatever the callback array
  // was
  // that is being used for the original file.  We shouldn't allow the visitor
  // to
  // arbitrarily stick a deserializer in there.
  if (auto EC = codeview::visitTypeStream(ExpectedTpi->typeArray(), Callbacks,
                                          VDS_BytesExternal))
    return std::move(EC);

  return true;
}

Expected<bool> PDBTypeServerHandler::handle(TypeServer2Record &TS,
                                            TypeVisitorCallbacks &Callbacks) {
  if (Session) {
    // If we've already handled this TypeServer and we only want to handle each
    // TypeServer once, consume the record without doing anything.
    if (!RevisitAlways)
      return true;

    return handleInternal(Session->getPDBFile(), Callbacks);
  }

  StringRef File = sys::path::filename(TS.Name);
  if (File.empty())
    return make_error<CodeViewError>(
        cv_error_code::corrupt_record,
        "TypeServer2Record does not contain filename!");

  for (auto &Path : SearchPaths) {
    SmallString<64> PathStr = Path.getKey();
    sys::path::append(PathStr, File);
    if (!sys::fs::exists(PathStr))
      continue;

    std::unique_ptr<IPDBSession> ThisSession;
    if (auto EC = loadDataForPDB(PDB_ReaderType::Native, PathStr, ThisSession)) {
      // It is not an error if this PDB fails to load, it just means that it
      // doesn't match and we should continue searching.
      ignoreErrors(std::move(EC));
      continue;
    }

    std::unique_ptr<NativeSession> NS(
        static_cast<NativeSession *>(ThisSession.release()));
    PDBFile &File = NS->getPDBFile();
    auto ExpectedInfo = File.getPDBInfoStream();
    // All PDB Files should have an Info stream.
    if (!ExpectedInfo)
      return ExpectedInfo.takeError();

    // Just because a file with a matching name was found and it was an actual
    // PDB file doesn't mean it matches.  For it to match the InfoStream's GUID
    // must match the GUID specified in the TypeServer2 record.
    ArrayRef<uint8_t> GuidBytes(ExpectedInfo->getGuid().Guid);
    StringRef GuidStr(reinterpret_cast<const char *>(GuidBytes.begin()),
                      GuidBytes.size());
    if (GuidStr != TS.Guid)
      continue;

    Session = std::move(NS);
    return handleInternal(File, Callbacks);
  }

  // We couldn't find a matching PDB, so let it be handled by someone else.
  return false;
}
