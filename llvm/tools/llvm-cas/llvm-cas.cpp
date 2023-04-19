//===- llvm-cas.cpp - CAS tool --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Optional.h"
#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/BuiltinUnifiedCASDatabases.h"
#include "llvm/CAS/CASFileSystem.h"
#include "llvm/CAS/CachingOnDiskFileSystem.h"
#include "llvm/CAS/HierarchicalTreeBuilder.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/CAS/TreeSchema.h"
#include "llvm/CAS/Utils.h"
#include "llvm/RemoteCachingService/RemoteCachingService.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrefixMapper.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <system_error>

using namespace llvm;
using namespace llvm::cas;

static cl::opt<bool> AllTrees("all-trees",
                              cl::desc("Print all trees, not just empty ones, for ls-tree-recursive"));
static cl::list<std::string> PrefixMapPaths(
    "prefix-map",
    cl::desc("prefix map for file system ingestion, -prefix-map BEFORE=AFTER"));

static int dump(ObjectStore &CAS);
static int listTree(ObjectStore &CAS, const CASID &ID);
static int listTreeRecursively(ObjectStore &CAS, const CASID &ID);
static int listObjectReferences(ObjectStore &CAS, const CASID &ID);
static int catBlob(ObjectStore &CAS, const CASID &ID);
static int catNodeData(ObjectStore &CAS, const CASID &ID);
static int printKind(ObjectStore &CAS, const CASID &ID);
static int makeBlob(ObjectStore &CAS, StringRef DataPath);
static int makeNode(ObjectStore &CAS, ArrayRef<std::string> References,
                    StringRef DataPath);
static int diffGraphs(ObjectStore &CAS, const CASID &LHS, const CASID &RHS);
static int traverseGraph(ObjectStore &CAS, const CASID &ID);
static int ingestFileSystem(ObjectStore &CAS, std::optional<StringRef> CASPath,
                            StringRef Path);
static int mergeTrees(ObjectStore &CAS, ArrayRef<std::string> Objects);
static int getCASIDForFile(ObjectStore &CAS, const CASID &ID, StringRef Path);
static int import(ObjectStore &CAS, ObjectStore &UpstreamCAS,
                  ArrayRef<std::string> Objects);
static int putCacheKey(ObjectStore &CAS, ActionCache &AC,
                       ArrayRef<std::string> Objects);
static int validateObject(ObjectStore &CAS, const CASID &ID);

int main(int Argc, char **Argv) {
  InitLLVM X(Argc, Argv);
  RegisterGRPCCAS Y;

  cl::list<std::string> Objects(cl::Positional, cl::desc("<object>..."));
  cl::opt<std::string> CASPath("cas", cl::desc("Path to CAS on disk."),
                               cl::value_desc("path"));
  cl::opt<std::string> UpstreamCASPath(
      "upstream-cas", cl::desc("Path to another CAS."), cl::value_desc("path"));
  cl::opt<std::string> DataPath("data",
                                cl::desc("Path to data or '-' for stdin."),
                                cl::value_desc("path"));

  enum CommandKind {
    Invalid,
    Dump,
    PrintKind,
    CatBlob,
    CatNodeData,
    DiffGraphs,
    TraverseGraph,
    MakeBlob,
    MakeNode,
    ListTree,
    ListTreeRecursive,
    ListObjectReferences,
    IngestFileSystem,
    MergeTrees,
    GetCASIDForFile,
    Import,
    PutCacheKey,
    Validate,
  };
  cl::opt<CommandKind> Command(
      cl::desc("choose command action:"),
      cl::values(
          clEnumValN(Dump, "dump", "dump internal contents"),
          clEnumValN(PrintKind, "print-kind", "print kind"),
          clEnumValN(CatBlob, "cat-blob", "cat blob"),
          clEnumValN(CatNodeData, "cat-node-data", "cat node data"),
          clEnumValN(DiffGraphs, "diff-graphs", "diff graphs"),
          clEnumValN(TraverseGraph, "traverse-graph", "traverse graph"),
          clEnumValN(MakeBlob, "make-blob", "make blob"),
          clEnumValN(MakeNode, "make-node", "make node"),
          clEnumValN(ListTree, "ls-tree", "list tree"),
          clEnumValN(ListTreeRecursive, "ls-tree-recursive",
                     "list tree recursive"),
          clEnumValN(ListObjectReferences, "ls-node-refs", "list node refs"),
          clEnumValN(IngestFileSystem, "ingest", "ingest file system"),
          clEnumValN(MergeTrees, "merge", "merge paths/cas-ids"),
          clEnumValN(GetCASIDForFile, "get-cas-id", "get cas id for file"),
          clEnumValN(Import, "import", "import objects from another CAS"),
          clEnumValN(PutCacheKey, "put-cache-key",
                     "set a value for a cache key"),
          clEnumValN(Validate, "validate", "validate the object for CASID")),
      cl::init(CommandKind::Invalid));

  cl::ParseCommandLineOptions(Argc, Argv, "llvm-cas CAS tool\n");
  ExitOnError ExitOnErr("llvm-cas: ");

  if (Command == CommandKind::Invalid)
    ExitOnErr(createStringError(inconvertibleErrorCode(),
                                "no command action is specified"));

  // FIXME: Consider creating an in-memory CAS.
  if (CASPath.empty())
    ExitOnErr(
        createStringError(inconvertibleErrorCode(), "missing --cas=<path>"));

  std::unique_ptr<ObjectStore> CAS;
  std::unique_ptr<ActionCache> AC;
  std::optional<StringRef> CASFilePath;
  if (sys::path::is_absolute(CASPath)) {
    CASFilePath = CASPath;
    std::tie(CAS, AC) = ExitOnErr(createOnDiskUnifiedCASDatabases(CASPath));
  } else {
    CAS = ExitOnErr(createCASFromIdentifier(CASPath));
  }
  assert(CAS);

  std::unique_ptr<ObjectStore> UpstreamCAS;
  if (!UpstreamCASPath.empty())
    UpstreamCAS = ExitOnErr(createCASFromIdentifier(UpstreamCASPath));

  if (Command == Dump)
    return dump(*CAS);

  if (Command == MakeBlob)
    return makeBlob(*CAS, DataPath);

  if (Command == MakeNode)
    return makeNode(*CAS, Objects, DataPath);

  if (Command == DiffGraphs) {
    ExitOnError CommandErr("llvm-cas: diff-graphs");

    if (Objects.size() != 2)
      CommandErr(
          createStringError(inconvertibleErrorCode(), "expected 2 objects"));

    CASID LHS = ExitOnErr(CAS->parseID(Objects[0]));
    CASID RHS = ExitOnErr(CAS->parseID(Objects[1]));
    return diffGraphs(*CAS, LHS, RHS);
  }

  if (Command == IngestFileSystem)
    return ingestFileSystem(*CAS, CASFilePath, DataPath);

  if (Command == MergeTrees)
    return mergeTrees(*CAS, Objects);

  if (Objects.empty())
    ExitOnErr(createStringError(inconvertibleErrorCode(),
                                "missing <object> to operate on"));

  if (Command == Import) {
    if (!UpstreamCAS)
      ExitOnErr(createStringError(inconvertibleErrorCode(),
                                  "missing '-upstream-cas'"));
    return import(*CAS, *UpstreamCAS, Objects);
  }

  if (Command == PutCacheKey) {
    if (!AC)
      ExitOnErr(createStringError(inconvertibleErrorCode(),
                                  "no action-cache available"));
    return putCacheKey(*CAS, *AC, Objects);
  }

  // Remaining commands need exactly one CAS object.
  if (Objects.size() > 1)
    ExitOnErr(createStringError(inconvertibleErrorCode(),
                                "too many <object>s, expected 1"));
  CASID ID = ExitOnErr(CAS->parseID(Objects.front()));

  if (Command == TraverseGraph)
    return traverseGraph(*CAS, ID);

  if (Command == ListTree)
    return listTree(*CAS, ID);

  if (Command == ListTreeRecursive)
    return listTreeRecursively(*CAS, ID);

  if (Command == ListObjectReferences)
    return listObjectReferences(*CAS, ID);

  if (Command == CatNodeData)
    return catNodeData(*CAS, ID);

  if (Command == PrintKind)
    return printKind(*CAS, ID);

  if (Command == GetCASIDForFile)
    return getCASIDForFile(*CAS, ID, DataPath);

  if (Command == Validate)
    return validateObject(*CAS, ID);

  assert(Command == CatBlob);
  return catBlob(*CAS, ID);
}

int listTree(ObjectStore &CAS, const CASID &ID) {
  ExitOnError ExitOnErr("llvm-cas: ls-tree: ");

  TreeSchema Schema(CAS);
  ObjectProxy TreeN = ExitOnErr(CAS.getProxy(ID));
  TreeProxy Tree = ExitOnErr(Schema.load(TreeN));
  ExitOnErr(Tree.forEachEntry([&](const NamedTreeEntry &Entry) {
    Entry.print(llvm::outs(), CAS);
    return Error::success();
  }));

  return 0;
}

int listTreeRecursively(ObjectStore &CAS, const CASID &ID) {
  ExitOnError ExitOnErr("llvm-cas: ls-tree-recursively: ");
  TreeSchema Schema(CAS);
  ObjectProxy TreeN = ExitOnErr(CAS.getProxy(ID));
  ExitOnErr(Schema.walkFileTreeRecursively(
      CAS, TreeN.getRef(),
      [&](const NamedTreeEntry &Entry, Optional<TreeProxy> Tree) -> Error {
        if (Entry.getKind() != TreeEntry::Tree) {
          Entry.print(llvm::outs(), CAS);
          return Error::success();
        }
        if (Tree->empty() || AllTrees)
          Entry.print(llvm::outs(), CAS);
        return Error::success();
      }));

  return 0;
}

int catBlob(ObjectStore &CAS, const CASID &ID) { return catNodeData(CAS, ID); }

static Expected<std::unique_ptr<MemoryBuffer>>
openBuffer(StringRef DataPath) {
  if (DataPath.empty())
    return createStringError(inconvertibleErrorCode(), "--data missing");
  return errorOrToExpected(
      DataPath == "-" ? llvm::MemoryBuffer::getSTDIN()
                      : llvm::MemoryBuffer::getFile(DataPath));
}

int dump(ObjectStore &CAS) {
  ExitOnError ExitOnErr("llvm-cas: dump: ");
  CAS.print(llvm::outs());
  return 0;
}

int makeBlob(ObjectStore &CAS, StringRef DataPath) {
  ExitOnError ExitOnErr("llvm-cas: make-blob: ");
  std::unique_ptr<MemoryBuffer> Buffer =
      ExitOnErr(openBuffer(DataPath));

  ObjectProxy Blob =
      ExitOnErr(CAS.createProxy(std::nullopt, Buffer->getBuffer()));
  llvm::outs() << Blob.getID() << "\n";
  return 0;
}

int catNodeData(ObjectStore &CAS, const CASID &ID) {
  ExitOnError ExitOnErr("llvm-cas: cat-node-data: ");
  llvm::outs() << ExitOnErr(CAS.getProxy(ID)).getData();
  return 0;
}

static StringRef getKindString(ObjectStore &CAS, ObjectProxy Object) {
  TreeSchema Schema(CAS);
  if (Schema.isNode(Object))
    return "tree";
  return "object";
}

int printKind(ObjectStore &CAS, const CASID &ID) {
  ExitOnError ExitOnErr("llvm-cas: print-kind: ");
  ObjectProxy Object = ExitOnErr(CAS.getProxy(ID));

  llvm::outs() << getKindString(CAS, Object) << "\n";
  return 0;
}

int listObjectReferences(ObjectStore &CAS, const CASID &ID) {
  ExitOnError ExitOnErr("llvm-cas: ls-node-refs: ");

  ObjectProxy Object = ExitOnErr(CAS.getProxy(ID));
  ExitOnErr(Object.forEachReference([&](ObjectRef Ref) -> Error {
    llvm::outs() << CAS.getID(Ref) << "\n";
    return Error::success();
  }));

  return 0;
}

static int makeNode(ObjectStore &CAS, ArrayRef<std::string> Objects,
                    StringRef DataPath) {
  std::unique_ptr<MemoryBuffer> Data =
      ExitOnError("llvm-cas: make-node: data: ")(openBuffer(DataPath));

  SmallVector<ObjectRef> IDs;
  for (StringRef Object : Objects) {
    ExitOnError ObjectErr("llvm-cas: make-node: ref: ");
    Optional<ObjectRef> ID = CAS.getReference(ObjectErr(CAS.parseID(Object)));
    if (!ID)
      ObjectErr(createStringError(inconvertibleErrorCode(),
                                  "unknown object '" + Object + "'"));
    IDs.push_back(*ID);
  }

  ExitOnError ExitOnErr("llvm-cas: make-node: ");
  ObjectProxy Object = ExitOnErr(CAS.createProxy(IDs, Data->getBuffer()));
  llvm::outs() << Object.getID() << "\n";
  return 0;
}

namespace {
struct GraphInfo {
  SmallVector<cas::CASID> PostOrder;
  DenseSet<cas::CASID> Seen;
};
} // namespace

static GraphInfo traverseObjectGraph(ObjectStore &CAS, const CASID &TopLevel) {
  ExitOnError ExitOnErr("llvm-cas: traverse-node-graph: ");
  GraphInfo Info;

  SmallVector<std::pair<CASID, bool>> Worklist;
  auto push = [&](CASID ID) {
    if (Info.Seen.insert(ID).second)
      Worklist.push_back({ID, false});
  };
  push(TopLevel);
  while (!Worklist.empty()) {
    if (Worklist.back().second) {
      Info.PostOrder.push_back(Worklist.pop_back_val().first);
      continue;
    }
    Worklist.back().second = true;
    CASID ID = Worklist.back().first;
    ObjectProxy Object = ExitOnErr(CAS.getProxy(ID));

    TreeSchema Schema(CAS);
    if (Schema.isNode(Object)) {
      TreeProxy Tree = ExitOnErr(Schema.load(Object));
      ExitOnErr(Tree.forEachEntry([&](const NamedTreeEntry &Entry) {
        push(CAS.getID(Entry.getRef()));
        return Error::success();
      }));
      continue;
    }

    ExitOnErr(Object.forEachReference([&](ObjectRef Ref) {
      push(CAS.getID(Ref));
      return Error::success();
    }));
  }

  return Info;
}

static void printDiffs(ObjectStore &CAS, const GraphInfo &Baseline,
                       const GraphInfo &New, StringRef NewName) {
  ExitOnError ExitOnErr("llvm-cas: diff-graphs: ");

  for (cas::CASID ID : New.PostOrder) {
    if (Baseline.Seen.count(ID))
      continue;

    StringRef KindString;
    ObjectProxy Object = ExitOnErr(CAS.getProxy(ID));
    KindString = getKindString(CAS, Object);

    outs() << llvm::formatv("{0}{1,-4} {2}\n", NewName, KindString, ID);
  }
}

int diffGraphs(ObjectStore &CAS, const CASID &LHS, const CASID &RHS) {
  if (LHS == RHS)
    return 0;

  ExitOnError ExitOnErr("llvm-cas: diff-graphs: ");
  GraphInfo LHSInfo = traverseObjectGraph(CAS, LHS);
  GraphInfo RHSInfo = traverseObjectGraph(CAS, RHS);

  printDiffs(CAS, RHSInfo, LHSInfo, "- ");
  printDiffs(CAS, LHSInfo, RHSInfo, "+ ");
  return 0;
}

int traverseGraph(ObjectStore &CAS, const CASID &ID) {
  ExitOnError ExitOnErr("llvm-cas: traverse-graph: ");
  GraphInfo Info = traverseObjectGraph(CAS, ID);
  printDiffs(CAS, GraphInfo{}, Info, "");
  return 0;
}

static Error
recursiveAccess(CachingOnDiskFileSystem &FS, StringRef Path,
                llvm::DenseSet<llvm::sys::fs::UniqueID> &SeenDirectories) {
  auto ST = FS.status(Path);

  // Ignore missing entries, which can be a symlink to a missing file, which is
  // not an error in the filesystem itself.
  // FIXME: add status(follow=false) to VFS instead, which would let us detect
  // this case directly.
  if (ST.getError() == llvm::errc::no_such_file_or_directory)
    return Error::success();

  if (!ST)
    return createFileError(Path, ST.getError());

  // Check that this is the first time we see the directory to prevent infinite
  // recursion into symlinks. The status() above will ensure all symlinks are
  // ingested.
  // FIXME: add status(follow=false) to VFS instead, and then only traverse
  // a directory and not a symlink to a directory.
  if (ST->isDirectory() && SeenDirectories.insert(ST->getUniqueID()).second) {
    std::error_code EC;
    for (llvm::vfs::directory_iterator I = FS.dir_begin(Path, EC), IE;
         !EC && I != IE; I.increment(EC)) {
      auto Err = recursiveAccess(FS, I->path(), SeenDirectories);
      if (Err)
        return Err;
    }
  }

  return Error::success();
}

static Expected<ObjectProxy> ingestFileSystemImpl(ObjectStore &CAS,
                                                  StringRef Path) {
  auto FS = createCachingOnDiskFileSystem(CAS);
  if (!FS)
    return FS.takeError();

  TreePathPrefixMapper Mapper(*FS);
  SmallVector<llvm::MappedPrefix> Split;
  if (!PrefixMapPaths.empty()) {
    MappedPrefix::transformJoinedIfValid(PrefixMapPaths, Split);
    Mapper.addRange(Split);
    Mapper.sort();
  }

  (*FS)->trackNewAccesses();

  llvm::DenseSet<llvm::sys::fs::UniqueID> SeenDirectories;
  if (Error E = recursiveAccess(**FS, Path, SeenDirectories))
    return std::move(E);

  return (*FS)->createTreeFromNewAccesses(
      [&](const llvm::vfs::CachedDirectoryEntry &Entry,
          SmallVectorImpl<char> &Storage) {
        return Mapper.mapDirEntry(Entry, Storage);
      });
}

/// Check that we are not attempting to ingest the CAS into itself, which can
/// accidentally create a weird or large cas.
Error checkCASIngestPath(StringRef CASPath, StringRef DataPath) {
  SmallString<128> RealCAS, RealData;
  if (std::error_code EC = sys::fs::real_path(StringRef(CASPath), RealCAS))
    return createFileError(CASPath, EC);
  if (std::error_code EC = sys::fs::real_path(StringRef(DataPath), RealData))
    return createFileError(CASPath, EC);
  if (RealCAS.startswith(RealData) &&
      (RealCAS.size() == RealData.size() ||
       sys::path::is_separator(RealCAS[RealData.size()])))
    return createStringError(inconvertibleErrorCode(),
                             "-cas is inside -data directory, which would "
                             "ingest the cas into itself");
  return Error::success();
}

int ingestFileSystem(ObjectStore &CAS, std::optional<StringRef> CASPath,
                     StringRef Path) {
  ExitOnError ExitOnErr("llvm-cas: ingest: ");
  if (Path.empty())
    ExitOnErr(
        createStringError(inconvertibleErrorCode(), "missing --data=<path>"));
  if (CASPath)
    ExitOnErr(checkCASIngestPath(*CASPath, Path));
  auto Ref = ExitOnErr(ingestFileSystemImpl(CAS, Path));
  outs() << Ref.getID() << "\n";
  return 0;
}

static int mergeTrees(ObjectStore &CAS, ArrayRef<std::string> Objects) {
  ExitOnError ExitOnErr("llvm-cas: merge: ");

  HierarchicalTreeBuilder Builder;
  for (const auto &Object : Objects) {
    auto ID = CAS.parseID(Object);
    if (ID) {
      if (Optional<ObjectRef> Ref = CAS.getReference(*ID))
        Builder.pushTreeContent(*Ref, "");
      else
        ExitOnErr(createStringError(inconvertibleErrorCode(),
                                    "unknown node with id: " + ID->toString()));
    } else {
      consumeError(ID.takeError());
      auto Ref = ExitOnErr(ingestFileSystemImpl(CAS, Object));
      Builder.pushTreeContent(Ref.getRef(), "");
    }
  }

  auto Ref = ExitOnErr(Builder.create(CAS));
  outs() << Ref.getID() << "\n";
  return 0;
}

int getCASIDForFile(ObjectStore &CAS, const CASID &ID, StringRef Path) {
  ExitOnError ExitOnErr("llvm-cas: get-cas-id: ");
  auto FS = createCASFileSystem(CAS, ID);
  if (!FS)
    ExitOnErr(FS.takeError());

  auto FileRef = (*FS)->getObjectRefForFileContent(Path);
  if (!FileRef)
    ExitOnErr(errorCodeToError(
        std::make_error_code(std::errc::no_such_file_or_directory)));

  CASID FileID = CAS.getID(**FileRef);
  outs() << FileID << "\n";
  return 0;
}

static ObjectRef importNode(ObjectStore &CAS, ObjectStore &UpstreamCAS,
                            const CASID &ID) {
  ExitOnError ExitOnErr("llvm-cas: import: ");

  std::optional<ObjectRef> PrimaryRef = CAS.getReference(ID);
  if (PrimaryRef)
    return *PrimaryRef; // object is present.

  ObjectProxy UpstreamObj = ExitOnErr(UpstreamCAS.getProxy(ID));
  SmallVector<ObjectRef> Refs;
  ExitOnErr(UpstreamObj.forEachReference([&](ObjectRef UpstreamRef) -> Error {
    ObjectRef Ref =
        importNode(CAS, UpstreamCAS, UpstreamCAS.getID(UpstreamRef));
    Refs.push_back(Ref);
    return Error::success();
  }));
  return ExitOnErr(CAS.storeFromString(Refs, UpstreamObj.getData()));
}

static int import(ObjectStore &CAS, ObjectStore &UpstreamCAS,
                  ArrayRef<std::string> Objects) {
  ExitOnError ExitOnErr("llvm-cas: import: ");

  for (StringRef Object : Objects) {
    CASID ID = ExitOnErr(CAS.parseID(Object));
    importNode(CAS, UpstreamCAS, ID);
  }
  return 0;
}

static int putCacheKey(ObjectStore &CAS, ActionCache &AC,
                       ArrayRef<std::string> Objects) {
  ExitOnError ExitOnErr("llvm-cas: put-cache-key: ");

  if (Objects.size() % 2 != 0)
    ExitOnErr(createStringError(inconvertibleErrorCode(),
                                "expected pairs of inputs"));
  while (!Objects.empty()) {
    CASID Key = ExitOnErr(CAS.parseID(Objects[0]));
    CASID Result = ExitOnErr(CAS.parseID(Objects[1]));
    Objects = Objects.drop_front(2);
    ExitOnErr(AC.put(Key, Result));
  }
  return 0;
}

int validateObject(ObjectStore &CAS, const CASID &ID) {
  ExitOnError ExitOnErr("llvm-cas: validate: ");
  ExitOnErr(CAS.validate(ID));
  outs() << ID << ": validated successfully\n";
  return 0;
}
