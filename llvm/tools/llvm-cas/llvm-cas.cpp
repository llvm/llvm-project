//===- llvm-cas.cpp - CAS tool --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/BuiltinUnifiedCASDatabases.h"
#include "llvm/CAS/CASRegistry.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/CAS/TreeSchema.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <system_error>

using namespace llvm;
using namespace llvm::cas;

static cl::opt<bool> AllTrees(
    "all-trees",
    cl::desc("Print all trees, not just empty ones, for ls-tree-recursive"));
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
static int import(ObjectStore &CAS, ObjectStore &UpstreamCAS,
                  ArrayRef<std::string> Objects);
static int putCacheKey(ObjectStore &CAS, ActionCache &AC,
                       ArrayRef<std::string> Objects);
static int getCacheResult(ObjectStore &CAS, ActionCache &AC, const CASID &ID);

int main(int Argc, char **Argv) {
  InitLLVM X(Argc, Argv);

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
    Import,
    PutCacheKey,
    GetCacheResult,
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
          clEnumValN(Import, "import", "import objects from another CAS"),
          clEnumValN(PutCacheKey, "put-cache-key",
                     "set a value for a cache key"),
          clEnumValN(GetCacheResult, "get-cache-result",
                     "get the result value from a cache key")),
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

  std::shared_ptr<ObjectStore> CAS;
  std::shared_ptr<ActionCache> AC;
  if (isRegisteredCASIdentifier(CASPath))
    std::tie(CAS, AC) = ExitOnErr(createCASFromIdentifier(CASPath));
  else
    std::tie(CAS, AC) = ExitOnErr(createOnDiskUnifiedCASDatabases(CASPath));

  std::shared_ptr<ObjectStore> UpstreamCAS;
  if (!UpstreamCASPath.empty())
    UpstreamCAS =
        std::move(ExitOnErr(createCASFromIdentifier(UpstreamCASPath)).first);

  if (Command == Dump)
    return dump(*CAS);

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

  if (Inputs.empty())
    ExitOnErr(createStringError(inconvertibleErrorCode(),
                                "missing <object> to operate on"));

  if (Command == Import) {
    if (!UpstreamCAS)
      ExitOnErr(createStringError(inconvertibleErrorCode(),
                                  "missing '-upstream-cas'"));
    return import(*CAS, *UpstreamCAS, Inputs);
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

static Expected<std::unique_ptr<MemoryBuffer>> openBuffer(StringRef DataPath) {
  if (DataPath.empty())
    return createStringError(inconvertibleErrorCode(), "--data missing");
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
