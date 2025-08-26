//===- llvm-cas.cpp - CAS tool --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/BuiltinUnifiedCASDatabases.h"
#include "llvm/CAS/CASFileSystem.h"
#include "llvm/CAS/CachingOnDiskFileSystem.h"
#include "llvm/CAS/HierarchicalTreeBuilder.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/CAS/TreeSchema.h"
#include "llvm/CAS/UnifiedOnDiskCache.h"
#include "llvm/CAS/Utils.h"
#include "llvm/RemoteCachingService/RemoteCachingService.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrefixMapper.h"
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
static cl::opt<bool>
    CASIDFile("casid-file",
              cl::desc("Input is CASID file, just ingest CASID."));
static cl::list<std::string> Inputs(cl::Positional, cl::desc("Input object"));

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
                            ArrayRef<std::string> Paths);
static int mergeTrees(ObjectStore &CAS, ArrayRef<std::string> Objects);
static int getCASIDForFile(ObjectStore &CAS, const CASID &ID,
                           ArrayRef<std::string> Path);
static int import(ObjectStore &CAS, ObjectStore &UpstreamCAS,
                  ArrayRef<std::string> Objects);
static int putCacheKey(ObjectStore &CAS, ActionCache &AC,
                       ArrayRef<std::string> Objects);
static int getCacheResult(ObjectStore &CAS, ActionCache &AC, const CASID &ID);
static int validateObject(ObjectStore &CAS, const CASID &ID);
static int validate(ObjectStore &CAS, ActionCache &AC, bool CheckHash);
static int validateIfNeeded(StringRef Path, StringRef PluginPath,
                            ArrayRef<std::string> PluginOpts, bool CheckHash,
                            bool Force, bool AllowRecovery, bool InProcess,
                            const char *Argv0);
static int ingestCasIDFile(cas::ObjectStore &CAS, ArrayRef<std::string> CASIDs);
static int prune(cas::ObjectStore &CAS);

int main(int Argc, char **Argv) {
  InitLLVM X(Argc, Argv);
  RegisterGRPCCAS Y;

  cl::opt<std::string> CASPath("cas", cl::desc("Path to CAS on disk."),
                               cl::value_desc("path"));
  cl::opt<std::string> CASPluginPath("fcas-plugin-path",
                                     cl::desc("Path to plugin CAS library"),
                                     cl::value_desc("path"));
  cl::list<std::string> CASPluginOpts("fcas-plugin-option",
                                      cl::desc("Plugin CAS Options"));
  cl::opt<std::string> UpstreamCASPath(
      "upstream-cas", cl::desc("Path to another CAS."), cl::value_desc("path"));
  cl::opt<std::string> DataPath("data",
                                cl::desc("Path to data or '-' for stdin."),
                                cl::value_desc("path"));
  cl::opt<bool> CheckHash("check-hash",
                          cl::desc("check all hashes during validation"));
  cl::opt<bool> AllowRecovery("allow-recovery",
                              cl::desc("allow recovery of cas data"));
  cl::opt<bool> Force("force",
                      cl::desc("force validation even if unnecessary"));
  cl::opt<bool> InProcess("in-process", cl::desc("validate in-process"));

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
    GetCacheResult,
    Validate,
    ValidateObject,
    ValidateIfNeeded,
    Prune,
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
          clEnumValN(GetCacheResult, "get-cache-result",
                     "get the result value from a cache key"),
          clEnumValN(Validate, "validate", "validate ObjectStore"),
          clEnumValN(ValidateObject, "validate-object",
                     "validate the object for CASID"),
          clEnumValN(ValidateIfNeeded, "validate-if-needed",
                     "validate cas contents if needed"),
          clEnumValN(Prune, "prune", "prune local cas storage")),
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

  if (Command == ValidateIfNeeded)
    return validateIfNeeded(CASPath, CASPluginPath, CASPluginOpts, CheckHash,
                            Force, AllowRecovery, InProcess, Argv[0]);

  std::shared_ptr<ObjectStore> CAS;
  std::shared_ptr<ActionCache> AC;
  std::optional<StringRef> CASFilePath;
  if (sys::path::is_absolute(CASPath)) {
    CASFilePath = CASPath;
    if (!CASPluginPath.empty()) {
      SmallVector<std::pair<std::string, std::string>> PluginOptions;
      for (const auto &PluginOpt : CASPluginOpts) {
        auto [Name, Val] = StringRef(PluginOpt).split('=');
        PluginOptions.push_back({std::string(Name), std::string(Val)});
      }
      std::tie(CAS, AC) = ExitOnErr(
          createPluginCASDatabases(CASPluginPath, CASPath, PluginOptions));
    } else {
      std::tie(CAS, AC) = ExitOnErr(createOnDiskUnifiedCASDatabases(CASPath));
    }
  } else {
    CAS = ExitOnErr(createCASFromIdentifier(CASPath));
  }
  assert(CAS);

  std::shared_ptr<ObjectStore> UpstreamCAS;
  if (!UpstreamCASPath.empty())
    UpstreamCAS = ExitOnErr(createCASFromIdentifier(UpstreamCASPath));

  if (Command == Dump)
    return dump(*CAS);

  if (Command == Validate)
    return validate(*CAS, *AC, CheckHash);

  if (Command == MakeBlob)
    return makeBlob(*CAS, DataPath);

  if (Command == MakeNode)
    return makeNode(*CAS, Inputs, DataPath);

  if (Command == DiffGraphs) {
    ExitOnError CommandErr("llvm-cas: diff-graphs");

    if (Inputs.size() != 2)
      CommandErr(
          createStringError(inconvertibleErrorCode(), "expected 2 objects"));

    CASID LHS = ExitOnErr(CAS->parseID(Inputs[0]));
    CASID RHS = ExitOnErr(CAS->parseID(Inputs[1]));
    return diffGraphs(*CAS, LHS, RHS);
  }

  if (Command == IngestFileSystem)
    return ingestFileSystem(*CAS, CASFilePath, Inputs);

  if (Command == MergeTrees)
    return mergeTrees(*CAS, Inputs);

  if (Command == Prune)
    return prune(*CAS);

  if (Inputs.empty())
    ExitOnErr(createStringError(inconvertibleErrorCode(),
                                "missing <object> to operate on"));

  if (Command == Import) {
    if (!UpstreamCAS)
      ExitOnErr(createStringError(inconvertibleErrorCode(),
                                  "missing '-upstream-cas'"));

    return import(*UpstreamCAS, *CAS, Inputs);
  }

  if (Command == PutCacheKey || Command == GetCacheResult) {
    if (!AC)
      ExitOnErr(createStringError(inconvertibleErrorCode(),
                                  "no action-cache available"));
  }

  if (Command == PutCacheKey)
    return putCacheKey(*CAS, *AC, Inputs);

  // Remaining commands need exactly one CAS object.
  if (Inputs.size() > 1)
    ExitOnErr(createStringError(inconvertibleErrorCode(),
                                "too many <object>s, expected 1"));
  CASID ID = ExitOnErr(CAS->parseID(Inputs.front()));

  if (Command == GetCacheResult)
    return getCacheResult(*CAS, *AC, ID);

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

  if (Command == ValidateObject)
    return validateObject(*CAS, ID);

  assert(Command == CatBlob);
  return catBlob(*CAS, ID);
}

int ingestCasIDFile(cas::ObjectStore &CAS, ArrayRef<std::string> CASIDs) {
  ExitOnError ExitOnErr;
  StringMap<ObjectRef> Files;
  SmallVector<ObjectProxy> SummaryIDs;
  for (StringRef IF : CASIDs) {
    auto ObjBuffer = ExitOnErr(errorOrToExpected(MemoryBuffer::getFile(IF)));
    auto ID = ExitOnErr(readCASIDBuffer(CAS, ObjBuffer->getMemBufferRef()));
    auto Ref = ExitOnErr(CAS.getProxy(ID)).getRef();
    assert(!Files.count(IF));
    Files.try_emplace(IF, Ref);
  }
  HierarchicalTreeBuilder Builder;
  for (auto &File : Files) {
    Builder.push(File.second, TreeEntry::Regular, File.first());
  }
  ObjectProxy SummaryRef = ExitOnErr(Builder.create(CAS));
  SummaryIDs.emplace_back(SummaryRef);
  outs() << SummaryRef.getID() << "\n";
  return 0;
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
      [&](const NamedTreeEntry &Entry, std::optional<TreeProxy> Tree) -> Error {
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
  std::unique_ptr<MemoryBuffer> Buffer = ExitOnErr(openBuffer(DataPath));

  ObjectProxy Blob = ExitOnErr(CAS.createProxy({}, Buffer->getBuffer()));
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
    std::optional<ObjectRef> ID =
        CAS.getReference(ObjectErr(CAS.parseID(Object)));
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
                                                  ArrayRef<std::string> Paths) {
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
  for (auto &Path : Paths)
    if (Error E = recursiveAccess(**FS, Path, SeenDirectories))
      return E;

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
    return createFileError(DataPath, EC);
  if (RealCAS.starts_with(RealData) &&
      (RealCAS.size() == RealData.size() ||
       sys::path::is_separator(RealCAS[RealData.size()])))
    return createStringError(inconvertibleErrorCode(),
                             "-cas is inside -data directory, which would "
                             "ingest the cas into itself");
  return Error::success();
}

int ingestFileSystem(ObjectStore &CAS, std::optional<StringRef> CASPath,
                     ArrayRef<std::string> Paths) {
  ExitOnError ExitOnErr("llvm-cas: ingest: ");
  if (CASIDFile)
    return ingestCasIDFile(CAS, Inputs);
  if (CASPath)
    for (auto File : Inputs)
      ExitOnErr(checkCASIngestPath(*CASPath, File));

  auto Ref = ExitOnErr(ingestFileSystemImpl(CAS, Paths));
  outs() << Ref.getID() << "\n";

  return 0;
}

static int mergeTrees(ObjectStore &CAS, ArrayRef<std::string> Objects) {
  ExitOnError ExitOnErr("llvm-cas: merge: ");

  HierarchicalTreeBuilder Builder;
  for (const auto &Object : Objects) {
    auto ID = CAS.parseID(Object);
    if (ID) {
      if (std::optional<ObjectRef> Ref = CAS.getReference(*ID))
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

int getCASIDForFile(ObjectStore &CAS, const CASID &ID,
                    ArrayRef<std::string> Path) {
  ExitOnError ExitOnErr("llvm-cas: get-cas-id: ");
  auto FS = createCASFileSystem(CAS, ID);
  if (!FS)
    ExitOnErr(FS.takeError());

  auto *CASFS = cast<CASBackedFileSystem>(FS->get());
  auto FileRef = CASFS->getObjectRefForFileContent(Path.front());
  if (!FileRef)
    ExitOnErr(errorCodeToError(
        std::make_error_code(std::errc::no_such_file_or_directory)));

  CASID FileID = CAS.getID(*FileRef);
  outs() << FileID << "\n";
  return 0;
}

static int import(ObjectStore &FromCAS, ObjectStore &ToCAS,
                  ArrayRef<std::string> Objects) {
  ExitOnError ExitOnErr("llvm-cas: import: ");

  for (StringRef Object : Objects) {
    CASID ID = ExitOnErr(FromCAS.parseID(Object));
    auto Ref = FromCAS.getReference(ID);
    if (!Ref) {
      ExitOnErr(createStringError(inconvertibleErrorCode(),
                                  "input not found: " + ID.toString()));
      return 1;
    }

    auto Imported = ExitOnErr(ToCAS.importObject(FromCAS, *Ref));
    llvm::outs() << ToCAS.getID(Imported).toString() << "\n";
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

static int getCacheResult(ObjectStore &CAS, ActionCache &AC, const CASID &ID) {
  ExitOnError ExitOnErr("llvm-cas: get-cache-result: ");

  auto Result = ExitOnErr(AC.get(ID));
  if (!Result) {
    outs() << "result not found\n";
    return 1;
  }
  outs() << *Result << "\n";
  return 0;
}

int validateObject(ObjectStore &CAS, const CASID &ID) {
  ExitOnError ExitOnErr("llvm-cas: validate-object: ");
  ExitOnErr(CAS.validateObject(ID));
  outs() << ID << ": validated successfully\n";
  return 0;
}

int validate(ObjectStore &CAS, ActionCache &AC, bool CheckHash) {
  ExitOnError ExitOnErr("llvm-cas: validate: ");
  ExitOnErr(CAS.validate(CheckHash));
  ExitOnErr(AC.validate());
  outs() << "validated successfully\n";
  return 0;
}

int validateIfNeeded(StringRef Path, StringRef PluginPath,
                     ArrayRef<std::string> PluginOpts, bool CheckHash,
                     bool Force, bool AllowRecovery, bool InProcess,
                     const char *Argv0) {
  ExitOnError ExitOnErr("llvm-cas: validate-if-needed: ");
  std::string ExecStorage;
  std::optional<StringRef> Exec;
  if (!InProcess) {
    ExecStorage = sys::fs::getMainExecutable(Argv0, (void *)validateIfNeeded);
    Exec = ExecStorage;
  }
  ValidationResult Result;
  if (PluginPath.empty()) {
    Result = ExitOnErr(validateOnDiskUnifiedCASDatabasesIfNeeded(
        Path, CheckHash, AllowRecovery, Force, Exec));
  } else {
    // FIXME: add a hook for plugin validation
    Result = ValidationResult::Skipped;
  }
  switch (Result) {
  case ValidationResult::Valid:
    outs() << "validated successfully\n";
    break;
  case ValidationResult::Recovered:
    outs() << "recovered from invalid data\n";
    break;
  case ValidationResult::Skipped:
    outs() << "validation skipped\n";
    break;
  }
  return 0;
}

static int prune(cas::ObjectStore &CAS) {
  ExitOnError ExitOnErr("llvm-cas: prune: ");
  ExitOnErr(CAS.pruneStorageData());
  return 0;
}
