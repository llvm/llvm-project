//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/OnDiskCASLogger.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::cas;
using namespace llvm::cas::ondisk;
using namespace llvm::sys;

#ifndef _WIN32 // windows doesn't support logging yet.

static void writeToLog(OnDiskCASLogger *Logger, int NumOpens, int NumEntries) {
  StringRef Path = "/fake_cas/index";
  uint8_t Hash[32] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                      11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                      22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
  void *Region = &Logger;

  for (int J = 0; J < NumOpens; ++J) {
    Logger->logMappedFileRegionArenaCreate(Path, 0, Region, 100, 7);
    Logger->logMappedFileRegionArenaResizeFile(Path, 7, 100);
    Logger->logMappedFileRegionArenaAllocate(Region, 0, 50);
    Logger->logMappedFileRegionArenaOom(Path, 100, 51, 50);
    for (int K = 0; K < NumEntries; ++K) {
      Logger->logMappedFileRegionArenaAllocate(Region, K, 10);
      Logger->logHashMappedTrieHandleCreateRecord(Region, K, Hash);
      Logger->logMappedFileRegionArenaAllocate(Region, K, 20);
      Logger->logSubtrieHandleCreate(Region, K, 1, 2);
      Logger->logSubtrieHandleCmpXchg(Region, K, J, -3, -1, -2);
    }
    Logger->logTempFileCreate(Path);
    Logger->logTempFileKeep(Path, Path, std::error_code());
    Logger->logTempFileRemove(Path, std::error_code());
    Logger->logMappedFileRegionArenaClose(Path);
    Logger->logUnifiedOnDiskCacheCollectGarbage(Path);
  }
}

static Error checkLog(StringRef Dir) {
  // Read back the log and check formatting.
  auto LogBuf = llvm::MemoryBuffer::getFile(Twine(Dir) + "/v1.log");
  if (std::error_code EC = LogBuf.getError())
    return createStringError(EC, "failed to read log");

  if ((*LogBuf)->getBuffer().empty())
    return createStringError("empty log file");

  for (line_iterator I(**LogBuf, /*SkipBlanks=*/false, '\0'); !I.is_at_eof();
       ++I) {
    auto MakeErr = [&](StringRef Reason) -> Error {
      return createStringError(Twine("invalid log at line ") +
                               std::to_string(I.line_number()) + ":\n" + *I);
    };
    StringRef Line = *I;
    StringRef TimeS, TimeMS, Pid, Tid;
    int Num;
    std::tie(TimeS, Line) = Line.split('.');
    if (TimeS.empty())
      return MakeErr("missing '.' for time");
    if (TimeS.getAsInteger(10, Num))
      return MakeErr("invalid time s");
    std::tie(TimeMS, Line) = Line.split(' ');
    if (TimeMS.empty())
      return MakeErr("missing ' ' after time ms");
    if (TimeMS.getAsInteger(10, Num))
      return MakeErr("invalid time ms");
    std::tie(Pid, Line) = Line.split(' ');
    if (Pid.empty())
      return MakeErr("missing ' ' after pid");
    if (Pid.getAsInteger(10, Num))
      return MakeErr("invalid pid");
    std::tie(Tid, Line) = Line.split(':');
    if (Tid.empty())
      return MakeErr("missing ':' after tid");
    if (Tid.getAsInteger(10, Num))
      return MakeErr("invalid tid");
    if (Line.empty())
      return MakeErr("nothing after ':'");
  }

  return Error::success();
}

TEST(OnDiskCASLoggerTest, MultiThread) {
  unittest::TempDir Dir("OnDiskCASLoggerTest_MultiThread", /*Unique=*/true);
  llvm::DefaultThreadPool Pool(llvm::hardware_concurrency());

  std::unique_ptr<OnDiskCASLogger> SharedLogger;
  ASSERT_THAT_ERROR(OnDiskCASLogger::open(Dir.path(), /*LogAllocations=*/true)
                        .moveInto(SharedLogger),
                    Succeeded());

  for (int I = 0; I < 10; ++I) {
    Pool.async([I, &SharedLogger, &Dir] {
      std::unique_ptr<OnDiskCASLogger> OwnedLogger;
      OnDiskCASLogger *Logger;
      // Mix using a shared instance and opening new instances in the same log.
      if (I % 2 == 0) {
        Logger = SharedLogger.get();
      } else {
        ASSERT_THAT_ERROR(
            OnDiskCASLogger::open(Dir.path(), /*LogAllocations=*/true)
                .moveInto(OwnedLogger),
            Succeeded());
        Logger = OwnedLogger.get();
      }

      writeToLog(Logger, /*NumOpens=*/10, /*NumEntries=*/100);
    });
  }
  Pool.wait();

  ASSERT_THAT_ERROR(checkLog(Dir.path()), Succeeded());
}

static cl::opt<std::string> CASLogDir("cas-log-dir");
// From TestMain.cpp.
extern const char *TestMainArgv0;

TEST(OnDiskCASLoggerTest, MultiProcess) {
  if (!CASLogDir.empty()) {
    // Child process.
    std::unique_ptr<OnDiskCASLogger> Logger;
    ASSERT_THAT_ERROR(OnDiskCASLogger::open(CASLogDir, /*LogAllocations=*/true)
                          .moveInto(Logger),
                      Succeeded());

    for (int I = 0; I < 10; ++I) {
      writeToLog(Logger.get(), /*NumOpens=*/5, /*NumEntries=*/50);
    }
    exit(0);
  }

  // Parent process.
  unittest::TempDir Dir("OnDiskCASLoggerTest_MultiProcess", /*Unique=*/true);

  std::string Executable =
      sys::fs::getMainExecutable(TestMainArgv0, &CASLogDir);
  StringRef Argv[] = {Executable,
                      "--gtest_filter=OnDiskCASLoggerTest.MultiProcess",
                      "-cas-log-dir", Dir.path()};

  std::string Error;
  SmallVector<ProcessInfo> PIs;
  for (int I = 0; I < 5; ++I) {
    bool ExecutionFailed;
    auto PI = ExecuteNoWait(Executable, Argv, ArrayRef<StringRef>{}, {}, 0,
                            &Error, &ExecutionFailed);
    ASSERT_FALSE(ExecutionFailed) << Error;
    PIs.push_back(std::move(PI));
  }

  for (auto &PI : PIs) {
    // Note: this is typically <1 second, but account for slow CI systems.
    auto Result = Wait(PI, /*Timeout=*/15, &Error);
    ASSERT_TRUE(Error.empty()) << Error;
    ASSERT_EQ(Result.Pid, PI.Pid);
    ASSERT_EQ(Result.ReturnCode, 0);
  }

  ASSERT_THAT_ERROR(checkLog(Dir.path()), Succeeded());
}

#endif // _WIN32
