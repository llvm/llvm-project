//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file A utility for operating on LLVM CAS.
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringMap.h"
#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/BuiltinUnifiedCASDatabases.h"
#include "llvm/CAS/CASFileSystem.h"
#include "llvm/CAS/CachingOnDiskFileSystem.h"
#include "llvm/CAS/HierarchicalTreeBuilder.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/CAS/TreeSchema.h"
#include "llvm/CAS/Utils.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
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

namespace {
enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "Options.inc"
#undef OPTION
};

#define OPTTABLE_STR_TABLE_CODE
#include "Options.inc"
#undef OPTTABLE_STR_TABLE_CODE

#define OPTTABLE_PREFIXES_TABLE_CODE
#include "Options.inc"
#undef OPTTABLE_PREFIXES_TABLE_CODE

using namespace llvm::opt;
static constexpr opt::OptTable::Info InfoTable[] = {
#define OPTION(...) LLVM_CONSTRUCT_OPT_INFO(__VA_ARGS__),
#include "Options.inc"
#undef OPTION
};

class LLVMCASOptTable : public opt::GenericOptTable {
public:
  LLVMCASOptTable()
      : opt::GenericOptTable(OptionStrTable, OptionPrefixesTable, InfoTable) {}
};

enum class CommandKind {
  Invalid,
  Dump,
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

struct CommandOptions {
  CommandKind Command = CommandKind::Invalid;
  std::vector<std::string> Inputs;
  std::string CASPath;
  std::string CASPluginPath;
  std::vector<std::string> CASPluginOpts;
  std::string UpstreamCASPath;
  std::string DataPath;
  std::vector<std::string> PrefixMapPaths;
  bool CheckHash;
  bool AllowRecovery;
  bool Force;
  bool InProcess;
  bool AllTrees;
  bool CASIDFile;

  static CommandKind getCommandKind(opt::Arg &A) {
    switch (A.getOption().getID()) {
    case OPT_cas_dump:
      return CommandKind::Dump;
    case OPT_cat_node_data:
      return CommandKind::CatNodeData;
    case OPT_diff_graph:
      return CommandKind::DiffGraphs;
    case OPT_traverse_graph:
      return CommandKind::TraverseGraph;
    case OPT_make_blob:
      return CommandKind::MakeBlob;
    case OPT_make_node:
      return CommandKind::MakeNode;
    case OPT_ls_tree:
      return CommandKind::ListTree;
    case OPT_ls_tree_recursive:
      return CommandKind::ListTreeRecursive;
    case OPT_ls_node_refs:
      return CommandKind::ListObjectReferences;
    case OPT_ingest:
      return CommandKind::IngestFileSystem;
    case OPT_merge_tree:
      return CommandKind::MergeTrees;
    case OPT_get_cas_id:
      return CommandKind::GetCASIDForFile;
    case OPT_import:
      return CommandKind::Import;
    case OPT_put_cache_key:
      return CommandKind::PutCacheKey;
    case OPT_get_cache_result:
      return CommandKind::GetCacheResult;
    case OPT_validate:
      return CommandKind::Validate;
    case OPT_validate_object:
      return CommandKind::ValidateObject;
    case OPT_validate_if_needed:
      return CommandKind::ValidateIfNeeded;
    case OPT_prune:
      return CommandKind::Prune;
    }
    return CommandKind::Invalid;
  }

  // Command requires input.
  static bool requiresInput(CommandKind Kind) {
    return Kind != CommandKind::ValidateIfNeeded &&
           Kind != CommandKind::Validate && Kind != CommandKind::MakeBlob &&
           Kind != CommandKind::MakeNode && Kind != CommandKind::Dump &&
           Kind != CommandKind::Prune && Kind != CommandKind::MergeTrees;
  }
};
} // namespace

static int dump(ObjectStore &CAS);
static int listTree(ObjectStore &CAS, const CASID &ID);
static int listTreeRecursively(ObjectStore &CAS, const CASID &ID,
                               bool AllTrees);
static int listObjectReferences(ObjectStore &CAS, const CASID &ID);
static int catNodeData(ObjectStore &CAS, const CASID &ID);
static int makeBlob(ObjectStore &CAS, StringRef DataPath);
static int makeNode(ObjectStore &CAS, ArrayRef<std::string> References,
                    StringRef DataPath);
static int diffGraphs(ObjectStore &CAS, const CASID &LHS, const CASID &RHS);
static int traverseGraph(ObjectStore &CAS, const CASID &ID);
static int ingestFileSystem(ObjectStore &CAS, std::optional<StringRef> CASPath,
                            ArrayRef<std::string> Paths,
                            ArrayRef<std::string> PrefixMapPaths,
                            bool CASIDFile);
static int mergeTrees(ObjectStore &CAS, ArrayRef<std::string> Objects);
static int getCASIDForFile(ObjectStore &CAS, const CASID &ID,
                           ArrayRef<std::string> Path);
static int import(ObjectStore &FromCAS, ObjectStore &ToCAS,
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

static Expected<CommandOptions> parseOptions(int Argc, char **Argv) {
  BumpPtrAllocator Alloc;
  StringSaver Saver(Alloc);
  SmallVector<const char *> ExpanedArgs;
  if (!cl::expandResponseFiles(Argc, Argv, nullptr, Saver, ExpanedArgs))
    return createStringError("cannot expand response file");

  LLVMCASOptTable T;
  unsigned MI, MC;
  opt::InputArgList Args = T.ParseArgs(ExpanedArgs, MI, MC);

  for (auto *Arg : Args.filtered(OPT_UNKNOWN)) {
    llvm::errs() << "ignoring unknown option: " << Arg->getSpelling() << '\n';
  }

  if (Args.hasArg(OPT_help)) {
    T.printHelp(
        outs(),
        (std::string(Argv[0]) + " [action] [options] <input files>").c_str(),
        "llvm-cas tool that performs CAS actions.", false);
    exit(0);
  }

  CommandOptions Opts;
  for (auto *A : Args.filtered(OPT_grp_action))
    Opts.Command = CommandOptions::getCommandKind(*A);

  if (Opts.Command == CommandKind::Invalid)
    return createStringError("no command action is specified");

  for (auto *File : Args.filtered(OPT_INPUT))
    Opts.Inputs.push_back(File->getValue());
  Opts.CASPath = Args.getLastArgValue(OPT_cas_path);
  Opts.CASPluginPath = Args.getLastArgValue(OPT_cas_plugin_path);
  Opts.CASPluginOpts = Args.getAllArgValues(OPT_cas_plugin_option);
  Opts.UpstreamCASPath = Args.getLastArgValue(OPT_upstream_cas);
  Opts.DataPath = Args.getLastArgValue(OPT_data);
  Opts.PrefixMapPaths = Args.getAllArgValues(OPT_prefix_map);
  Opts.CheckHash = Args.hasArg(OPT_check_hash);
  Opts.AllowRecovery = Args.hasArg(OPT_allow_recovery);
  Opts.Force = Args.hasArg(OPT_force);
  Opts.InProcess = Args.hasArg(OPT_in_process);
  Opts.AllTrees = Args.hasArg(OPT_all_trees);
  Opts.CASIDFile = Args.hasArg(OPT_casid_file);

  // Validate options.
  if (Opts.CASPath.empty())
    return createStringError("missing --cas <path>");

  if (Opts.Inputs.empty() && CommandOptions::requiresInput(Opts.Command))
    return createStringError("missing <input> to operate on");

  return Opts;
}

int main(int Argc, char **Argv) {
  InitLLVM X(Argc, Argv);

  ExitOnError ExitOnErr;
  auto Opts = ExitOnErr(parseOptions(Argc, Argv));

  if (Opts.Command == CommandKind::ValidateIfNeeded)
    return validateIfNeeded(Opts.CASPath, Opts.CASPluginPath,
                            Opts.CASPluginOpts, Opts.CheckHash, Opts.Force,
                            Opts.AllowRecovery, Opts.InProcess, Argv[0]);

  std::shared_ptr<ObjectStore> CAS;
  std::shared_ptr<ActionCache> AC;
  std::optional<StringRef> CASFilePath;
  if (sys::path::is_absolute(Opts.CASPath)) {
    CASFilePath = Opts.CASPath;
    if (!Opts.CASPluginPath.empty()) {
      SmallVector<std::pair<std::string, std::string>> PluginOptions;
      for (const auto &PluginOpt : Opts.CASPluginOpts) {
        auto [Name, Val] = StringRef(PluginOpt).split('=');
        PluginOptions.push_back({std::string(Name), std::string(Val)});
      }
      std::tie(CAS, AC) = ExitOnErr(createPluginCASDatabases(
          Opts.CASPluginPath, Opts.CASPath, PluginOptions));
    } else {
      std::tie(CAS, AC) =
          ExitOnErr(createOnDiskUnifiedCASDatabases(Opts.CASPath));
    }
  } else {
    CAS = ExitOnErr(createCASFromIdentifier(Opts.CASPath));
  }
  assert(CAS);

  if (Opts.Command == CommandKind::Dump)
    return dump(*CAS);

  if (Opts.Command == CommandKind::Validate)
    return validate(*CAS, *AC, Opts.CheckHash);

  if (Opts.Command == CommandKind::MakeBlob)
    return makeBlob(*CAS, Opts.DataPath);

  if (Opts.Command == CommandKind::MakeNode)
    return makeNode(*CAS, Opts.Inputs, Opts.DataPath);

  if (Opts.Command == CommandKind::DiffGraphs) {
    ExitOnError CommandErr("llvm-cas: diff-graphs");

    if (Opts.Inputs.size() != 2)
      CommandErr(createStringError("expected 2 objects"));

    CASID LHS = ExitOnErr(CAS->parseID(Opts.Inputs[0]));
    CASID RHS = ExitOnErr(CAS->parseID(Opts.Inputs[1]));
    return diffGraphs(*CAS, LHS, RHS);
  }

  if (Opts.Command == CommandKind::IngestFileSystem)
    return ingestFileSystem(*CAS, CASFilePath, Opts.Inputs, Opts.PrefixMapPaths,
                            Opts.CASIDFile);

  if (Opts.Command == CommandKind::MergeTrees)
    return mergeTrees(*CAS, Opts.Inputs);

  if (Opts.Command == CommandKind::Prune)
    return prune(*CAS);

  if (Opts.Command == CommandKind::Import) {
    if (Opts.UpstreamCASPath.empty())
      ExitOnErr(createStringError("missing '-upstream-cas'"));

    auto UpstreamCAS = ExitOnErr(createCASFromIdentifier(Opts.UpstreamCASPath));

    return import(*UpstreamCAS, *CAS, Opts.Inputs);
  }

  if (Opts.Command == CommandKind::PutCacheKey ||
      Opts.Command == CommandKind::GetCacheResult) {
    if (!AC)
      ExitOnErr(createStringError("no action-cache available"));
  }

  if (Opts.Command == CommandKind::PutCacheKey)
    return putCacheKey(*CAS, *AC, Opts.Inputs);

  // Remaining commands need exactly one CAS object.
  if (Opts.Inputs.size() > 1)
    ExitOnErr(createStringError("too many <object>s, expected 1"));
  CASID ID = ExitOnErr(CAS->parseID(Opts.Inputs.front()));

  if (Opts.Command == CommandKind::GetCacheResult)
    return getCacheResult(*CAS, *AC, ID);

  if (Opts.Command == CommandKind::TraverseGraph)
    return traverseGraph(*CAS, ID);

  if (Opts.Command == CommandKind::ListTree)
    return listTree(*CAS, ID);

  if (Opts.Command == CommandKind::ListTreeRecursive)
    return listTreeRecursively(*CAS, ID, Opts.AllTrees);

  if (Opts.Command == CommandKind::ListObjectReferences)
    return listObjectReferences(*CAS, ID);

  if (Opts.Command == CommandKind::CatNodeData)
    return catNodeData(*CAS, ID);

  if (Opts.Command == CommandKind::GetCASIDForFile)
    return getCASIDForFile(*CAS, ID, Opts.DataPath);

  assert(Opts.Command == CommandKind::ValidateObject);
  return validateObject(*CAS, ID);
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

int listTreeRecursively(ObjectStore &CAS, const CASID &ID, bool AllTrees) {
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

static Expected<std::unique_ptr<MemoryBuffer>> openBuffer(StringRef DataPath) {
  if (DataPath.empty())
    return createStringError("--data missing");
  return errorOrToExpected(DataPath == "-"
                               ? llvm::MemoryBuffer::getSTDIN()
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
      ObjectErr(createStringError("unknown object '" + Object + "'"));
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

static Expected<ObjectProxy>
ingestFileSystemImpl(ObjectStore &CAS, ArrayRef<std::string> Paths,
                     ArrayRef<std::string> PrefixMapPaths) {
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
static Error checkCASIngestPath(StringRef CASPath, StringRef DataPath) {
  SmallString<128> RealCAS, RealData;
  if (std::error_code EC = sys::fs::real_path(StringRef(CASPath), RealCAS))
    return createFileError(CASPath, EC);
  if (std::error_code EC = sys::fs::real_path(StringRef(DataPath), RealData))
    return createFileError(DataPath, EC);
  if (RealCAS.starts_with(RealData) &&
      (RealCAS.size() == RealData.size() ||
       sys::path::is_separator(RealCAS[RealData.size()])))
    return createStringError("-cas is inside -data directory, which would "
                             "ingest the cas into itself");
  return Error::success();
}

static int ingestFileSystem(ObjectStore &CAS, std::optional<StringRef> CASPath,
                            ArrayRef<std::string> Paths,
                            ArrayRef<std::string> PrefixMapPaths,
                            bool CASIDFile) {
  ExitOnError ExitOnErr("llvm-cas: ingest: ");
  if (CASIDFile)
    return ingestCasIDFile(CAS, Paths);
  if (CASPath)
    for (auto File : Paths)
      ExitOnErr(checkCASIngestPath(*CASPath, File));

  auto Ref = ExitOnErr(ingestFileSystemImpl(CAS, Paths, PrefixMapPaths));
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
        ExitOnErr(createStringError("unknown node with id: " + ID->toString()));
    } else {
      consumeError(ID.takeError());
      auto Ref = ExitOnErr(ingestFileSystemImpl(CAS, Object, {}));
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
    if (!Ref)
      ExitOnErr(createStringError("input not found: " + ID.toString()));

    auto Imported = ExitOnErr(ToCAS.importObject(FromCAS, *Ref));
    llvm::outs() << ToCAS.getID(Imported).toString() << "\n";
  }
  return 0;
}

static int putCacheKey(ObjectStore &CAS, ActionCache &AC,
                       ArrayRef<std::string> Objects) {
  ExitOnError ExitOnErr("llvm-cas: put-cache-key: ");

  if (Objects.size() % 2 != 0)
    ExitOnErr(createStringError("expected pairs of inputs"));
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
