//===- unittest/Support/ProcessTest.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Process.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Error.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <optional>

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef LLVM_ON_UNIX
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

using namespace llvm;
using namespace sys;

TEST(ProcessTest, GetProcessIdTest) {
  const Process::Pid pid = Process::getProcessId();

#ifdef _WIN32
  EXPECT_EQ((DWORD)pid, ::GetCurrentProcessId());
#else
  EXPECT_EQ(pid, ::getpid());
#endif
}

TEST(ProcessTest, GetRandomNumberTest) {
  const unsigned r1 = Process::GetRandomNumber();
  const unsigned r2 = Process::GetRandomNumber();
  // It should be extremely unlikely that both r1 and r2 are 0.
  EXPECT_NE((r1 | r2), 0u);
}

TEST(ProcessTest, GetRandomNumberKeepsRandState) {
  // GetRandomNumber must not reseed or advance the C library's rand() state,
  // which belongs to the program.
  ::srand(42);
  const int Expected1 = ::rand();
  const int Expected2 = ::rand();
  ::srand(42);
  (void)Process::GetRandomNumber();
  EXPECT_EQ(::rand(), Expected1);
  (void)Process::GetRandomNumber();
  EXPECT_EQ(::rand(), Expected2);
}

#ifdef LLVM_ON_UNIX
TEST(ProcessTest, GetRandomNumberDiffersAfterFork) {
  // A child process must not draw the same numbers as its parent, or they
  // would, for example, pick the same temporary file names.
  (void)Process::GetRandomNumber();
  int Pipe[2];
  ASSERT_EQ(::pipe(Pipe), 0);
  const pid_t Child = ::fork();
  ASSERT_NE(Child, -1);
  if (Child == 0) {
    const unsigned Numbers[2] = {Process::GetRandomNumber(),
                                 Process::GetRandomNumber()};
    const ssize_t Written = ::write(Pipe[1], Numbers, sizeof(Numbers));
    ::_exit(Written == sizeof(Numbers) ? 0 : 1);
  }
  ::close(Pipe[1]);
  unsigned ChildNumbers[2];
  const ssize_t Read = ::read(Pipe[0], ChildNumbers, sizeof(ChildNumbers));
  ::close(Pipe[0]);
  int Status;
  ASSERT_EQ(::waitpid(Child, &Status, 0), Child);
  ASSERT_EQ(Read, static_cast<ssize_t>(sizeof(ChildNumbers)));
  const unsigned ParentNumbers[2] = {Process::GetRandomNumber(),
                                     Process::GetRandomNumber()};
  // Two numbers make an accidental match very unlikely.
  EXPECT_FALSE(ChildNumbers[0] == ParentNumbers[0] &&
               ChildNumbers[1] == ParentNumbers[1]);
}
#endif

#ifdef _MSC_VER
#define setenv(name, var, ignore) _putenv_s(name, var)
#endif

#if HAVE_SETENV || _MSC_VER
TEST(ProcessTest, Basic) {
  setenv("__LLVM_TEST_ENVIRON_VAR__", "abc", true);
  std::optional<std::string> val(Process::GetEnv("__LLVM_TEST_ENVIRON_VAR__"));
  EXPECT_TRUE(val.has_value());
  EXPECT_STREQ("abc", val->c_str());
}

TEST(ProcessTest, None) {
  std::optional<std::string> val(
      Process::GetEnv("__LLVM_TEST_ENVIRON_NO_SUCH_VAR__"));
  EXPECT_FALSE(val.has_value());
}
#endif

#ifdef _WIN32

TEST(ProcessTest, EmptyVal) {
  SetEnvironmentVariableA("__LLVM_TEST_ENVIRON_VAR__", "");
  std::optional<std::string> val(Process::GetEnv("__LLVM_TEST_ENVIRON_VAR__"));
  EXPECT_TRUE(val.has_value());
  EXPECT_STREQ("", val->c_str());
}

TEST(ProcessTest, Wchar) {
  SetEnvironmentVariableW(L"__LLVM_TEST_ENVIRON_VAR__", L"abcdefghijklmnopqrs");
  std::optional<std::string> val(Process::GetEnv("__LLVM_TEST_ENVIRON_VAR__"));
  EXPECT_TRUE(val.has_value());
  EXPECT_STREQ("abcdefghijklmnopqrs", val->c_str());
}
#endif

class PageSizeTest : public testing::Test {
  Triple Host;

protected:
  PageSizeTest() : Host(Triple::normalize(sys::getProcessTriple())) {}

  bool isSupported() const {
    // For now just on X86-64 and Aarch64. This can be expanded in the future.
    return (Host.getArch() == Triple::x86_64 ||
            Host.getArch() == Triple::aarch64) &&
           Host.getOS() == Triple::Linux;
  }

  bool pageSizeAsExpected(unsigned PageSize) const {
    switch (Host.getArch()) {
    case Triple::x86_64:
      return PageSize == 4096;
    case Triple::aarch64:
      // supported granule sizes are 4k, 16k and 64k
      return PageSize == 4096 || PageSize == 16384 || PageSize == 65536;
    default:
      llvm_unreachable("unexpected arch!");
    }
  }
};

TEST_F(PageSizeTest, PageSize) {
  if (!isSupported())
    GTEST_SKIP();

  llvm::Expected<unsigned> Result = llvm::sys::Process::getPageSize();
  ASSERT_FALSE(!Result);
  ASSERT_TRUE(pageSizeAsExpected(*Result));
}

} // end anonymous namespace
