//===- MappedFileRegionBumpPtrTest.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Program.h"
#include "llvm/CAS/MappedFileRegionBumpPtr.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ThreadPool.h"
#include "gtest/gtest.h"
#if defined(__APPLE__)
#include <crt_externs.h>
#elif !defined(_MSC_VER)
// Forward declare environ in case it's not provided by stdlib.h.
extern char **environ;
#endif

using namespace llvm;
using namespace llvm::cas;

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
      auto convStatus = convertUTF16ToUTF8String(Ref, EnvStorage.back());
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

#if LLVM_ENABLE_ONDISK_CAS

TEST_F(CASProgramTest, MappedFileRegionBumpPtrTest) {
  auto TestAllocator = [](StringRef Path) {
    auto NewFileConstructor = [&](MappedFileRegionBumpPtr &Alloc) -> Error {
      Alloc.initializeBumpPtr(0);
      return Error::success();
    };

    auto Alloc = MappedFileRegionBumpPtr::create(
        Path, /*Capacity=*/10 * 1024 * 1024,
        /*BumpPtrOffset=*/0, NewFileConstructor);
    if (!Alloc)
      ASSERT_TRUE(false);

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
  sys::fs::createUniqueDirectory("MappedFileRegionBumpPtr", FilePath);
  sys::path::append(FilePath, "allocation-file");

  std::string Executable =
      sys::fs::getMainExecutable(TestMainArgv0, &ProgramID);
  StringRef Argv[] = {
      Executable, "--gtest_filter=CASProgramTest.MappedFileRegionBumpPtrTest"};

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
  ASSERT_TRUE(Error.empty());
  ASSERT_NE(PI.Pid, sys::ProcessInfo::InvalidPid) << "Invalid process id";
  llvm::sys::Wait(PI, /*SecondsToWait=*/5, &Error);
  ASSERT_TRUE(Error.empty());

  // Clean up after both processes finish testing.
  sys::fs::remove(FilePath);
  sys::fs::remove_directories(sys::path::parent_path(FilePath));
}

#endif // LLVM_ENABLE_ONDISK_CAS
