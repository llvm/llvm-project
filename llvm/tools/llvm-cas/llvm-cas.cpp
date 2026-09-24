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

#include "llvm/ADT/StringExtras.h"
#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/BuiltinUnifiedCASDatabases.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::cas;

namespace {
enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "Options.inc"
#undef OPTION
};

using namespace llvm::opt;
#define OPTTABLE_CODE
#include "Options.inc"

class LLVMCASOptTable : public opt::OptTable {
public:
  LLVMCASOptTable() : opt::OptTable(optionTables()) {}
};

enum class CommandKind {
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

struct CommandOptions {
  CommandKind Command = CommandKind::Invalid;
  std::vector<std::string> Inputs;
  std::string CASPath;
  std::string CASPluginPath;
  SmallVector<std::pair<std::string, std::string>> CASPluginOpts;
  std::string UpstreamCASPath;
  std::string DataPath;
  bool CheckHash;
  bool AllowRecovery;
  bool Force;
  bool InProcess;

  static CommandKind getCommandKind(opt::Arg &A) {
    switch (A.getOption().getID()) {
    case OPT_cas_dump:
      return CommandKind::Dump;
    case OPT_cat_node_data:
      return CommandKind::CatNodeData;
    case OPT_make_blob:
      return CommandKind::MakeBlob;
    case OPT_make_node:
      return CommandKind::MakeNode;
    case OPT_ls_node_refs:
      return CommandKind::ListObjectReferences;
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
           Kind != CommandKind::Prune;
  }
};
} // namespace

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
static int validateIfNeeded(const CommandOptions &Opts, const char *Argv0);
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
  for (StringRef PluginOpt : Args.getAllArgValues(OPT_cas_plugin_option)) {
    auto [Name, Value] = PluginOpt.split('=');
    Opts.CASPluginOpts.emplace_back(Name, Value);
  }
  Opts.UpstreamCASPath = Args.getLastArgValue(OPT_upstream_cas);
  Opts.DataPath = Args.getLastArgValue(OPT_data);
  Opts.CheckHash = Args.hasArg(OPT_check_hash);
  Opts.AllowRecovery = Args.hasArg(OPT_allow_recovery);
  Opts.Force = Args.hasArg(OPT_force);
  Opts.InProcess = Args.hasArg(OPT_in_process);

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
    return validateIfNeeded(Opts, Argv[0]);

  std::shared_ptr<ObjectStore> CAS;
  std::shared_ptr<ActionCache> AC;
  if (!Opts.CASPluginPath.empty())
    std::tie(CAS, AC) = ExitOnErr(createPluginCASDatabases(
        Opts.CASPluginPath, Opts.CASPath, Opts.CASPluginOpts));
  else
    std::tie(CAS, AC) =
        ExitOnErr(createOnDiskUnifiedCASDatabases(Opts.CASPath));
  assert(CAS);

  if (Opts.Command == CommandKind::Dump)
    return dump(*CAS);

  if (Opts.Command == CommandKind::Validate)
    return validate(*CAS, *AC, Opts.CheckHash);

  if (Opts.Command == CommandKind::MakeBlob)
    return makeBlob(*CAS, Opts.DataPath);

  if (Opts.Command == CommandKind::MakeNode)
    return makeNode(*CAS, Opts.Inputs, Opts.DataPath);

  if (Opts.Command == CommandKind::Prune)
    return prune(*CAS);

  if (Opts.Command == CommandKind::Import) {
    if (Opts.UpstreamCASPath.empty())
      ExitOnErr(createStringError("missing '-upstream-cas'"));

    auto [UpstreamCAS, _] =
        ExitOnErr(createOnDiskUnifiedCASDatabases(Opts.UpstreamCASPath));
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

  if (Opts.Command == CommandKind::ListObjectReferences)
    return listObjectReferences(*CAS, ID);

  if (Opts.Command == CommandKind::CatNodeData)
    return catNodeData(*CAS, ID);

  assert(Opts.Command == CommandKind::ValidateObject);
  return validateObject(*CAS, ID);
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

/// Validates the CAS in this process and prints the result.
static Error validateInProcess(const CommandOptions &Opts) {
  ValidationResult Result;
  if (Error E = (Opts.CASPluginPath.empty()
                     ? validateOnDiskUnifiedCASDatabasesIfNeeded(
                           Opts.CASPath, Opts.CheckHash, Opts.Force)
                     : validatePluginCASDatabasesIfNeeded(
                           Opts.CASPluginPath, Opts.CASPath, Opts.CASPluginOpts,
                           Opts.CheckHash, Opts.Force))
                    .moveInto(Result))
    return E;
  outs() << (Result == ValidationResult::Skipped ? "validation skipped\n"
                                                 : "validated successfully\n");
  return Error::success();
}

/// Validates the CAS by re-executing llvm-cas with --in-process, which
/// protects against crashes during validation. The output of the child process
/// is forwarded.
///
/// \returns false if validation failed or crashed.
static Expected<bool> validateOutOfProcess(const CommandOptions &Opts,
                                           const char *Argv0) {
  std::string Exec =
      sys::fs::getMainExecutable(Argv0, (void *)validateOutOfProcess);
  SmallVector<std::string> PluginOpts;
  for (const auto &[Name, Value] : Opts.CASPluginOpts)
    PluginOpts.push_back(Name + "=" + Value);

  SmallVector<StringRef> Args{Exec, "--cas", Opts.CASPath};
  if (!Opts.CASPluginPath.empty()) {
    Args.append({"--fcas-plugin-path", Opts.CASPluginPath});
    for (StringRef PluginOpt : PluginOpts)
      Args.append({"--fcas-plugin-option", PluginOpt});
  }
  Args.append({"--validate-if-needed", "--in-process"});
  if (Opts.CheckHash)
    Args.push_back("--check-hash");
  if (Opts.Force)
    Args.push_back("--force");

  outs().flush();
  std::string ErrMsg;
  int Result = sys::ExecuteAndWait(Exec, Args, /*Env=*/std::nullopt,
                                   /*Redirects=*/{}, /*SecondsToWait=*/120,
                                   /*MemoryLimit=*/0, &ErrMsg);
  if (Result == -1)
    return createStringError("failed to exec " + join(Args, " ") + ": " +
                             ErrMsg);
  if (Result == -2)
    errs() << "llvm-cas: validate-if-needed: validation crashed: " << ErrMsg
           << "\n";
  return Result == 0;
}

int validateIfNeeded(const CommandOptions &Opts, const char *Argv0) {
  ExitOnError ExitOnErr("llvm-cas: validate-if-needed: ");
  if (Opts.InProcess) {
    Error E = validateInProcess(Opts);
    if (!E)
      return 0;
    if (!Opts.AllowRecovery)
      ExitOnErr(std::move(E));
    errs() << "llvm-cas: validate-if-needed: " << toString(std::move(E))
           << "\n";
  } else {
    if (ExitOnErr(validateOutOfProcess(Opts, Argv0)))
      return 0;
    if (!Opts.AllowRecovery)
      ExitOnErr(createStringError("cas contents invalid"));
  }

  ValidationResult Result = ExitOnErr(
      Opts.CASPluginPath.empty()
          ? recoverOnDiskUnifiedCASDatabases(Opts.CASPath)
          : recoverPluginCASDatabases(Opts.CASPluginPath, Opts.CASPath,
                                      Opts.CASPluginOpts));
  outs() << (Result == ValidationResult::Skipped
                 ? "recovery skipped\n"
                 : "recovered from invalid data\n");
  return 0;
}

static int prune(cas::ObjectStore &CAS) {
  ExitOnError ExitOnErr("llvm-cas: prune: ");
  ExitOnErr(CAS.pruneStorageData());
  return 0;
}
