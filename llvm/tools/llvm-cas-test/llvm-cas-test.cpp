//===- llvm-cas-test.cpp - CAS stress tester ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/BuiltinUnifiedCASDatabases.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/RandomNumberGenerator.h"
#include "llvm/Support/ThreadPool.h"

using namespace llvm;
using namespace llvm::cas;

enum CommandKind {
  StressTest,
  GenerateData,
  CheckLockFiles
};

static cl::opt<CommandKind>
    Command(cl::desc("choose command action:"),
            cl::values(clEnumValN(StressTest, "stress-test", "stress test CAS"),
                       clEnumValN(GenerateData, "gen-data",
                                  "fill CAS with random data"),
                       clEnumValN(CheckLockFiles, "check-lock-files",
                                  "check lock files")),
            cl::init(CommandKind::StressTest));

// CAS configuration.
static cl::opt<std::string>
    CASPath("cas", cl::desc("CAS path on disk for testing"), cl::Required);
static cl::opt<bool> Verbose("v", cl::desc("verbose output"));
static cl::opt<bool>
    ForceKill("force-kill",
              cl::desc("force kill subprocess to test termination"));
static cl::opt<bool> KeepLog("keep-log",
                             cl::desc("keep log and do not rotate the log"));

// CAS stress test parameters.
static cl::opt<unsigned>
    OptNumShards("num-shards", cl::desc("number of shards"), cl::init(0));
static cl::opt<unsigned> OptTreeDepth("tree-depth", cl::desc("tree depth"),
                                      cl::init(0));
static cl::opt<unsigned> OptNumChildren("num-children",
                                        cl::desc("number of child nodes"),
                                        cl::init(0));
static cl::opt<unsigned> OptDataLength("data-length", cl::desc("data length"),
                                       cl::init(0));
static cl::opt<unsigned> OptPrecentFile(
    "precent-file",
    cl::desc("percentage of nodes that is long enough to be file based"),
    cl::init(0));
// Default size to be 100MB.
static cl::opt<uint64_t>
    SizeLimit("size-limit", cl::desc("CAS size limit (in MB)"), cl::init(100));
// Default timeout 180s.
static cl::opt<uint64_t>
    Timeout("timeout", cl::desc("test timeout (in seconds)"), cl::init(180));

enum CASFuzzingSettings : uint8_t {
  Default = 0,
  Fork = 1,                   // CAS Data filling happens in subprocesses.
  CheckTermination = 1 << 1, // Try kill the subprocess when it fills the data.

  Last = UINT8_MAX, // Enum is randomly generated, use MAX to cover all inputs.
  LLVM_MARK_AS_BITMASK_ENUM(Last)
};

struct Config {
  CASFuzzingSettings Settings = Default;
  uint8_t NumShards;
  uint8_t NumChildren;
  uint8_t TreeDepth;
  uint16_t DataLength;
  uint16_t PrecentFile;

  static constexpr unsigned MaxShards = 20;
  static constexpr unsigned MaxChildren = 32;
  static constexpr unsigned MaxDepth = 8;
  static constexpr unsigned MaxDataLength = 1024 * 4;

  void constrainParameters() {
    // reduce the size of parameter if they are too big.
    NumShards = NumShards % MaxShards;
    NumChildren = NumChildren % MaxChildren;
    TreeDepth = TreeDepth % MaxDepth;
    DataLength = DataLength % MaxDataLength;
    PrecentFile = PrecentFile % 100;

    if (ForceKill) {
      Settings |= Fork;
      Settings |= CheckTermination;
    }
  }

  bool extendToFile(uint8_t Seed) const {
    return ((float)Seed / (float)UINT8_MAX) > ((float)PrecentFile / 100.0f);
  }

  void init() {
    NumShards = OptNumShards ? OptNumShards : MaxShards;
    NumChildren = OptNumChildren ? OptNumChildren : MaxChildren;
    TreeDepth = OptTreeDepth ? OptTreeDepth : MaxDepth;
    DataLength = OptDataLength ? OptDataLength : MaxDataLength;
    PrecentFile = OptPrecentFile;
  }

  void appendCommandLineOpts(std::vector<std::string> &Cmd) {
    Cmd.push_back("--num-shards=" + utostr(NumShards));
    Cmd.push_back("--num-children=" + utostr(NumChildren));
    Cmd.push_back("--tree-depth=" + utostr(TreeDepth));
    Cmd.push_back("--data-length=" + utostr(DataLength));
    Cmd.push_back("--precent-file=" + utostr(PrecentFile));
  }

  void dump() {
    llvm::errs() << "## Configuration:"
                 << " Fork: " << (bool)(Settings & Fork)
                 << " Kill: " << (bool)(Settings & CheckTermination)
                 << " NumShards: " << (unsigned)NumShards
                 << " TreeDepth: " << (unsigned)TreeDepth
                 << " NumChildren: " << (unsigned)NumChildren
                 << " DataLength: " << (unsigned)DataLength
                 << " PrecentFile: " << (unsigned)PrecentFile << "\n";
  }
};

// fill the CAS with random data of specified tree depth and children numbers.
static void fillData(ObjectStore &CAS, ActionCache &AC, const Config &Conf) {
  ExitOnError ExitOnErr("llvm-cas-test fill data: ");
  DefaultThreadPool ThreadPool(hardware_concurrency());
  for (size_t I = 0; I != Conf.NumShards; ++I) {
    ThreadPool.async([&] {
      std::vector<ObjectRef> Refs;
      for (unsigned Depth = 0; Depth < Conf.TreeDepth; ++Depth) {
        unsigned NumNodes = (Conf.TreeDepth - Depth + 1) * Conf.NumChildren + 1;
        std::vector<ObjectRef> Created;
        Created.reserve(NumNodes);
        ArrayRef<ObjectRef> PreviouslyCreated(Refs);
        for (unsigned I = 0; I < NumNodes; ++I) {
          std::vector<char> Data(Conf.DataLength);
          getRandomBytes(Data.data(), Data.size());
          // Use the first byte that generated to decide if we should make it
          // 64KB bigger and force that into a file based storage.
          if (Conf.extendToFile(Data[0]))
            Data.resize(64LL * 1024LL + Conf.DataLength);

          if (Depth == 0) {
            auto Ref = ExitOnErr(CAS.store({}, Data));
            Created.push_back(Ref);
          } else {
            auto Parent = PreviouslyCreated.slice(I, Conf.NumChildren);
            auto Ref = ExitOnErr(CAS.store(Parent, Data));
            Created.push_back(Ref);
          }
        }
        // Put a self mapping in action cache to avoid cache poisoning.
        if (!Created.empty())
          ExitOnErr(
              AC.put(CAS.getID(Created.back()), CAS.getID(Created.back())));
        Refs.swap(Created);
      }
    });
  }
  ThreadPool.wait();
}

static int genData() {
  ExitOnError ExitOnErr("llvm-cas-test --gen-data: ");

  Config Conf;
  Conf.init();

  auto DB = ExitOnErr(cas::createOnDiskUnifiedCASDatabases(CASPath));
  fillData(*DB.first, *DB.second, Conf);

  return 0;
}

static int runOneTest(const char *Argv0) {
  ExitOnError ExitOnErr("llvm-cas-test: ");

  Config Conf;
  getRandomBytes(&Conf, sizeof(Conf));
  Conf.constrainParameters();

  if (Verbose)
    Conf.dump();

  // Start with fresh log if --keep-log is not used.
  if (!KeepLog) {
    static constexpr StringLiteral LogFile = "v1.log";
    SmallString<256> LogPath(CASPath);
    llvm::sys::path::append(LogPath, LogFile);
    llvm::sys::fs::rename(LogPath, LogPath + ".old");
  }

  auto DB = ExitOnErr(cas::createOnDiskUnifiedCASDatabases(CASPath));
  auto &CAS = *DB.first;
  auto &AC = *DB.second;

  // Size limit in MB.
  ExitOnErr(CAS.setSizeLimit(SizeLimit * 1024 * 1024));
  if (Conf.Settings & Fork) {
    // fill data using sub processes.
    std::string MainExe = sys::fs::getMainExecutable(Argv0, &CASPath);
    std::vector<std::string> Args = {MainExe, "--gen-data", "--cas=" + CASPath};
    Conf.appendCommandLineOpts(Args);

    std::vector<StringRef> Cmd;
    for_each(Args, [&Cmd](const std::string &Arg) { Cmd.push_back(Arg); });

    std::vector<sys::ProcessInfo> Subprocesses;
    for (int I = 0; I < Conf.NumShards; ++I) {
      auto SP = sys::ExecuteNoWait(MainExe, Cmd, std::nullopt);
      if (SP.Pid != 0)
        Subprocesses.push_back(SP);
    }

    std::optional<unsigned> Timeout;
    // Wait 1 second and killed the process if CheckTermination.
    if (Conf.Settings & CheckTermination)
      Timeout = 1;

    auto HasError = any_of(Subprocesses, [&](auto &P) {
      std::string ErrMsg;
      auto WP = sys::Wait(P, Timeout, /*ErrMsg=*/&ErrMsg);
      if (WP.ReturnCode == 0)
        return false;
      if ((Conf.Settings & CheckTermination) && WP.ReturnCode == -2 &&
          StringRef(ErrMsg).starts_with("Child timed out")) {
        if (Verbose)
          llvm::errs() << "subprocess killed successfully\n";
        return false;
      }
      llvm::errs() << "subprocess failed with error code (" << WP.ReturnCode
                   << "): " << ErrMsg << "\n";
      return true;
    });
    if (HasError) {
      llvm::errs() << "end of stress test due to an error in subprocess\n";
      return 1;
    }
  } else {
    // in-process fill data.
    fillData(CAS, AC, Conf);
  }
  if (Verbose)
    llvm::errs() << "Finished filling data, start validating\n";

  // validate and prune in the end.
  ExitOnErr(CAS.validate(true));
  if (Verbose)
    llvm::errs() << "Finished validating, start pruning storage if needed\n";

  ExitOnErr(CAS.pruneStorageData());
  if (Verbose)
    llvm::errs() << "Finished pruning, end of iteration\n";

  return 0;
}

static int stressTest(const char *Argv0) {
  auto Start = std::chrono::steady_clock::now();
  std::chrono::seconds Duration(Timeout);

  while (std::chrono::steady_clock::now() - Start < Duration) {
    if (int Res = runOneTest(Argv0))
      return Res;
  }

  return 0;
}

static int checkLockFiles() {
  ExitOnError ExitOnErr("llvm-cas-test: check-lock-files: ");

  SmallString<128> DataPoolPath(CASPath);
  sys::path::append(DataPoolPath, "v1.1/v9.data");

  auto OpenCASAndGetDataPoolSize = [&]() -> Expected<uint64_t> {
    auto Result = createOnDiskUnifiedCASDatabases(CASPath);
    if (!Result)
      return Result.takeError();

    sys::fs::file_status DataStat;
    if (std::error_code EC = sys::fs::status(DataPoolPath, DataStat))
      ExitOnErr(createFileError(DataPoolPath, EC));
    return DataStat.getSize();
  };

  // Get the normal size of an open CAS data pool to compare against later.
  uint64_t OpenSize = ExitOnErr(OpenCASAndGetDataPoolSize());

  DefaultThreadPool Pool;
  for (int I = 0; I < 1000; ++I) {
    Pool.async([&, I] {
      uint64_t DataPoolSize = ExitOnErr(OpenCASAndGetDataPoolSize());
      if (DataPoolSize < OpenSize)
        ExitOnErr(createStringError(
            inconvertibleErrorCode(),
            StringRef("CAS data file size (" + std::to_string(DataPoolSize) +
                      ") is smaller than expected (" +
                      std::to_string(OpenSize) + ") in iteration " +
                      std::to_string(I))));
    });
  }

  Pool.wait();
  return 0;
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "llvm-cas-test CAS testing tool\n");

  switch (Command) {
  case GenerateData:
    return genData();
  case StressTest:
    return stressTest(argv[0]);
  case CheckLockFiles:
    return checkLockFiles();
  }
}
