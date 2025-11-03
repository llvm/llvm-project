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

#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/BuiltinUnifiedCASDatabases.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/CAS/UnifiedOnDiskCache.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <system_error>

using namespace llvm;
using namespace llvm::cas;

static cl::list<std::string> Inputs(cl::Positional, cl::desc("Input object"));

static int dump(ObjectStore &CAS);
static int listObjectReferences(ObjectStore &CAS, const CASID &ID);
static int catNodeData(ObjectStore &CAS, const CASID &ID);
static int makeBlob(ObjectStore &CAS, StringRef DataPath);
static int makeNode(ObjectStore &CAS, ArrayRef<std::string> References,
                    StringRef DataPath);
static int import(ObjectStore &FromCAS, ObjectStore &ToCAS,
                  ArrayRef<std::string> Objects);
static int putCacheKey(ObjectStore &CAS, ActionCache &AC,
                       ArrayRef<std::string> Objects);
static int getCacheResult(ObjectStore &CAS, ActionCache &AC, const CASID &ID);
static int validateObject(ObjectStore &CAS, const CASID &ID);
static int validate(ObjectStore &CAS, ActionCache &AC, bool CheckHash);
static int validateIfNeeded(StringRef Path, bool CheckHash, bool Force,
                            bool AllowRecovery, bool InProcess,
                            const char *Argv0);
static int prune(cas::ObjectStore &CAS);

int main(int Argc, char **Argv) {
  InitLLVM X(Argc, Argv);
  cl::opt<std::string> CASPath("cas", cl::desc("Path to CAS on disk."),
                               cl::value_desc("path"));
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
    CatNodeData,
    MakeBlob,
    MakeNode,
    ListObjectReferences,
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
          clEnumValN(CatNodeData, "cat-node-data", "cat node data"),
          clEnumValN(MakeBlob, "make-blob", "make blob"),
          clEnumValN(MakeNode, "make-node", "make node"),
          clEnumValN(ListObjectReferences, "ls-node-refs", "list node refs"),
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
    return validateIfNeeded(CASPath, CheckHash, Force, AllowRecovery, InProcess,
                            Argv[0]);

  auto [CAS, AC] = ExitOnErr(createOnDiskUnifiedCASDatabases(CASPath));
  assert(CAS);

  if (Command == Dump)
    return dump(*CAS);

  if (Command == Validate)
    return validate(*CAS, *AC, CheckHash);

  if (Command == MakeBlob)
    return makeBlob(*CAS, DataPath);

  if (Command == MakeNode)
    return makeNode(*CAS, Inputs, DataPath);

  if (Command == Prune)
    return prune(*CAS);

  if (Inputs.empty())
    ExitOnErr(createStringError(inconvertibleErrorCode(),
                                "missing <object> to operate on"));

  if (Command == Import) {
    if (UpstreamCASPath.empty())
      ExitOnErr(createStringError(inconvertibleErrorCode(),
                                  "missing '-upstream-cas'"));

    auto [UpstreamCAS, _] =
        ExitOnErr(createOnDiskUnifiedCASDatabases(UpstreamCASPath));
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

  if (Command == ListObjectReferences)
    return listObjectReferences(*CAS, ID);

  if (Command == CatNodeData)
    return catNodeData(*CAS, ID);

  assert(Command == ValidateObject);
  return validateObject(*CAS, ID);
}

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

  ObjectProxy Blob = ExitOnErr(CAS.createProxy({}, Buffer->getBuffer()));
  llvm::outs() << Blob.getID() << "\n";
  return 0;
}

int catNodeData(ObjectStore &CAS, const CASID &ID) {
  ExitOnError ExitOnErr("llvm-cas: cat-node-data: ");
  llvm::outs() << ExitOnErr(CAS.getProxy(ID)).getData();
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

int validateIfNeeded(StringRef Path, bool CheckHash, bool Force,
                     bool AllowRecovery, bool InProcess, const char *Argv0) {
  ExitOnError ExitOnErr("llvm-cas: validate-if-needed: ");
  std::string ExecStorage;
  std::optional<StringRef> Exec;
  if (!InProcess) {
    ExecStorage = sys::fs::getMainExecutable(Argv0, (void *)validateIfNeeded);
    Exec = ExecStorage;
  }
  ValidationResult Result = ExitOnErr(validateOnDiskUnifiedCASDatabasesIfNeeded(
      Path, CheckHash, AllowRecovery, Force, Exec));
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
