//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Program.h"
#include "llvm/CAS/MappedFileRegionArena.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/ExponentialBackoff.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#if defined(__APPLE__)
#include <crt_externs.h>
#elif !defined(_MSC_VER)
// Forward declare environ in case it's not provided by stdlib.h.
extern char **environ;
#endif

using namespace llvm;
using namespace llvm::cas;

#if LLVM_ENABLE_ONDISK_CAS

extern const char *TestMainArgv0;
static char ProgramID = 0;

class CASProgramTest : public testing::Test {
  std::vector<StringRef> EnvTable;
  std::vector<std::string> EnvStorage;

protected:
  void SetUp() override {
    auto EnvP = [] {
#if defined(_WIN32)
      _wgetenv(L"TMP"); // Populate _wenviron, initially is null
      return _wenviron;
#elif defined(__APPLE__)
      return *_NSGetEnviron();
#else
      return environ;
#endif
    }();
    ASSERT_TRUE(EnvP);

    auto prepareEnvVar = [this](decltype(*EnvP) Var) -> StringRef {
#if defined(_WIN32)
      // On Windows convert UTF16 encoded variable to UTF8
      auto Len = wcslen(Var);
      ArrayRef<char> Ref{reinterpret_cast<char const *>(Var),
                         Len * sizeof(*Var)};
      EnvStorage.emplace_back();
      auto convStatus = llvm::convertUTF16ToUTF8String(Ref, EnvStorage.back());
      EXPECT_TRUE(convStatus);
      return EnvStorage.back();
#else
      (void)this;
      return StringRef(Var);
#endif
    };

    while (*EnvP != nullptr) {
      auto S = prepareEnvVar(*EnvP);
      if (!StringRef(S).starts_with("GTEST_"))
        EnvTable.emplace_back(S);
      ++EnvP;
    }
  }

  void TearDown() override {
    EnvTable.clear();
    EnvStorage.clear();
  }

  void addEnvVar(StringRef Var) { EnvTable.emplace_back(Var); }

  ArrayRef<StringRef> getEnviron() const { return EnvTable; }
};

static Error emptyConstructor(MappedFileRegionArena &) {
  return Error::success();
}

TEST_F(CASProgramTest, MappedFileRegionArenaTest) {
  auto TestAllocator = [](StringRef Path) {
    std::optional<MappedFileRegionArena> Alloc;
    ASSERT_THAT_ERROR(
        MappedFileRegionArena::create(Path, /*Capacity=*/10 * 1024 * 1024,
                                      /*HeaderOffset=*/0, /*Logger=*/nullptr,
                                      emptyConstructor)
            .moveInto(Alloc),
        Succeeded());

    std::vector<unsigned *> AllocatedPtr;
    AllocatedPtr.resize(100);
    DefaultThreadPool Threads;
    for (unsigned I = 0; I < 100; ++I) {
      Threads.async(
          [&](unsigned Idx) {
            // Allocate a buffer that is larger than needed so allocator hits
            // additional pages for test coverage.
            unsigned *P = (unsigned *)cantFail(Alloc->allocate(100));
            *P = Idx;
            AllocatedPtr[Idx] = P;
          },
          I);
    }

    Threads.wait();
    for (unsigned I = 0; I < 100; ++I)
      EXPECT_EQ(*AllocatedPtr[I], I);
  };

  if (const char *File = getenv("LLVM_CAS_TEST_MAPPED_FILE_REGION")) {
    TestAllocator(File);
    exit(0);
  }

  SmallString<128> FilePath;
  sys::fs::createUniqueDirectory("MappedFileRegionArena", FilePath);
  sys::path::append(FilePath, "allocation-file");

  std::string Executable =
      sys::fs::getMainExecutable(TestMainArgv0, &ProgramID);
  StringRef Argv[] = {
      Executable, "--gtest_filter=CASProgramTest.MappedFileRegionArenaTest"};

  // Add LLVM_PROGRAM_TEST_LOCKED_FILE to the environment of the child.
  std::string EnvVar = "LLVM_CAS_TEST_MAPPED_FILE_REGION=";
  EnvVar += FilePath.str();
  addEnvVar(EnvVar);

  std::string Error;
  bool ExecutionFailed;
  sys::ProcessInfo PI = sys::ExecuteNoWait(Executable, Argv, getEnviron(), {},
                                           0, &Error, &ExecutionFailed);
  TestAllocator(FilePath);

  ASSERT_FALSE(ExecutionFailed) << Error;
  ASSERT_NE(PI.Pid, sys::ProcessInfo::InvalidPid) << "Invalid process id";
  PI = llvm::sys::Wait(PI, /*SecondsToWait=*/5, &Error);
  ASSERT_TRUE(PI.ReturnCode == 0);
  ASSERT_TRUE(Error.empty());

  // Clean up after both processes finish testing.
  sys::fs::remove(FilePath);
  sys::fs::remove_directories(sys::path::parent_path(FilePath));
}

TEST_F(CASProgramTest, MappedFileRegionArenaSizeTest) {
  using namespace std::chrono_literals;
  if (const char *File = getenv("LLVM_CAS_TEST_MAPPED_FILE_REGION")) {
    ExponentialBackoff Backoff(5s);
    do {
      if (sys::fs::exists(File)) {
        break;
      }
    } while (Backoff.waitForNextAttempt());

    std::optional<MappedFileRegionArena> Alloc;
    ASSERT_THAT_ERROR(MappedFileRegionArena::create(File, /*Capacity=*/1024,
                                                    /*HeaderOffset=*/0,
                                                    /*Logger=*/nullptr,
                                                    emptyConstructor)
                          .moveInto(Alloc),
                      Succeeded());
    ASSERT_TRUE(Alloc->capacity() == 2048);

    Alloc.reset();
    ASSERT_THAT_ERROR(MappedFileRegionArena::create(File, /*Capacity=*/4096,
                                                    /*HeaderOffset=*/0,
                                                    /*Logger=*/nullptr,
                                                    emptyConstructor)
                          .moveInto(Alloc),
                      Succeeded());
    ASSERT_TRUE(Alloc->capacity() == 2048);
    Alloc.reset();

    ASSERT_THAT_ERROR(
        MappedFileRegionArena::create(File, /*Capacity=*/2048,
                                      /*HeaderOffset=*/32,
                                      /*Logger=*/nullptr, emptyConstructor)
            .moveInto(Alloc),
        FailedWithMessage(
            "specified header offset (32) does not match existing config (0)"));

    ASSERT_THAT_ERROR(MappedFileRegionArena::create(File, /*Capacity=*/2048,
                                                    /*HeaderOffset=*/0,
                                                    /*Logger=*/nullptr,
                                                    emptyConstructor)
                          .moveInto(Alloc),
                      Succeeded());

    exit(0);
  }

  SmallString<128> FilePath;
  sys::fs::createUniqueDirectory("MappedFileRegionArena", FilePath);
  sys::path::append(FilePath, "allocation-file");

  std::string Executable =
      sys::fs::getMainExecutable(TestMainArgv0, &ProgramID);
  StringRef Argv[] = {
      Executable,
      "--gtest_filter=CASProgramTest.MappedFileRegionArenaSizeTest"};

  // Add LLVM_PROGRAM_TEST_LOCKED_FILE to the environment of the child.
  std::string EnvVar = "LLVM_CAS_TEST_MAPPED_FILE_REGION=";
  EnvVar += FilePath.str();
  addEnvVar(EnvVar);

  std::optional<MappedFileRegionArena> Alloc;
  ASSERT_THAT_ERROR(MappedFileRegionArena::create(FilePath, /*Capacity=*/2048,
                                                  /*HeaderOffset=*/0,
                                                  /*Logger=*/nullptr,
                                                  emptyConstructor)
                        .moveInto(Alloc),
                    Succeeded());

  std::string Error;
  bool ExecutionFailed;
  sys::ProcessInfo PI = sys::ExecuteNoWait(Executable, Argv, getEnviron(), {},
                                           0, &Error, &ExecutionFailed);

  ASSERT_FALSE(ExecutionFailed) << Error;
  ASSERT_NE(PI.Pid, sys::ProcessInfo::InvalidPid) << "Invalid process id";
  PI = llvm::sys::Wait(PI, /*SecondsToWait=*/100, &Error);
  ASSERT_TRUE(PI.ReturnCode == 0);
  ASSERT_TRUE(Error.empty());

  // Size is still the requested 2048.
  ASSERT_TRUE(Alloc->capacity() == 2048);

  // Clean up after both processes finish testing.
  sys::fs::remove(FilePath);
  sys::fs::remove_directories(sys::path::parent_path(FilePath));
}

#endif // LLVM_ENABLE_ONDISK_CAS
