//===- unittest/Support/ProgramTest.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Program.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include "gtest/gtest.h"
#include <stdlib.h>
#include <thread>
#if defined(__APPLE__)
# include <crt_externs.h>
#elif !defined(_MSC_VER)
// Forward declare environ in case it's not provided by stdlib.h.
extern char **environ;
#endif

#if defined(LLVM_ON_UNIX)
#include <unistd.h>
void sleep_for(unsigned int seconds) {
  sleep(seconds);
}
#elif defined(_WIN32)
#include <windows.h>
void sleep_for(unsigned int seconds) {
  Sleep(seconds * 1000);
}
#else
#error sleep_for is not implemented on your platform.
#endif

#define ASSERT_NO_ERROR(x)                                                     \
  if (std::error_code ASSERT_NO_ERROR_ec = x) {                                \
    SmallString<128> MessageStorage;                                           \
    raw_svector_ostream Message(MessageStorage);                               \
    Message << #x ": did not return errc::success.\n"                          \
            << "error number: " << ASSERT_NO_ERROR_ec.value() << "\n"          \
            << "error message: " << ASSERT_NO_ERROR_ec.message() << "\n";      \
    GTEST_FATAL_FAILURE_(MessageStorage.c_str());                              \
  } else {                                                                     \
  }
// From TestMain.cpp.
extern const char *TestMainArgv0;

namespace {

using namespace llvm;
using namespace sys;

static cl::opt<std::string>
ProgramTestStringArg1("program-test-string-arg1");
static cl::opt<std::string>
ProgramTestStringArg2("program-test-string-arg2");

class ProgramEnvTest : public testing::Test {
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
      if (!StringRef(S).startswith("GTEST_"))
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

#ifdef _WIN32
void checkSeparators(StringRef Path) {
  char UndesiredSeparator = sys::path::get_separator()[0] == '/' ? '\\' : '/';
  ASSERT_EQ(Path.find(UndesiredSeparator), StringRef::npos);
}

TEST_F(ProgramEnvTest, CreateProcessLongPath) {
  if (getenv("LLVM_PROGRAM_TEST_LONG_PATH"))
    exit(0);

  // getMainExecutable returns an absolute path; prepend the long-path prefix.
  SmallString<128> MyAbsExe(
      sys::fs::getMainExecutable(TestMainArgv0, &ProgramTestStringArg1));
  checkSeparators(MyAbsExe);
  // Force a path with backslashes, when we are going to prepend the \\?\
  // prefix.
  sys::path::native(MyAbsExe, sys::path::Style::windows_backslash);
  std::string MyExe;
  if (!StringRef(MyAbsExe).startswith("\\\\?\\"))
    MyExe.append("\\\\?\\");
  MyExe.append(std::string(MyAbsExe.begin(), MyAbsExe.end()));

  StringRef ArgV[] = {MyExe,
                      "--gtest_filter=ProgramEnvTest.CreateProcessLongPath"};

  // Add LLVM_PROGRAM_TEST_LONG_PATH to the environment of the child.
  addEnvVar("LLVM_PROGRAM_TEST_LONG_PATH=1");

  // Redirect stdout to a long path.
  SmallString<128> TestDirectory;
  ASSERT_NO_ERROR(
    fs::createUniqueDirectory("program-redirect-test", TestDirectory));
  SmallString<256> LongPath(TestDirectory);
  LongPath.push_back('\\');
  // MAX_PATH = 260
  LongPath.append(260 - TestDirectory.size(), 'a');

  std::string Error;
  bool ExecutionFailed;
  std::optional<StringRef> Redirects[] = {std::nullopt, LongPath.str(),
                                          std::nullopt};
  int RC = ExecuteAndWait(MyExe, ArgV, getEnviron(), Redirects,
    /*secondsToWait=*/ 10, /*memoryLimit=*/ 0, &Error,
    &ExecutionFailed);
  EXPECT_FALSE(ExecutionFailed) << Error;
  EXPECT_EQ(0, RC);

  // Remove the long stdout.
  ASSERT_NO_ERROR(fs::remove(Twine(LongPath)));
  ASSERT_NO_ERROR(fs::remove(Twine(TestDirectory)));
}
#endif

TEST_F(ProgramEnvTest, CreateProcessTrailingSlash) {
  if (getenv("LLVM_PROGRAM_TEST_CHILD")) {
    if (ProgramTestStringArg1 == "has\\\\ trailing\\" &&
        ProgramTestStringArg2 == "has\\\\ trailing\\") {
      exit(0);  // Success!  The arguments were passed and parsed.
    }
    exit(1);
  }

  std::string my_exe =
      sys::fs::getMainExecutable(TestMainArgv0, &ProgramTestStringArg1);
  StringRef argv[] = {
      my_exe,
      "--gtest_filter=ProgramEnvTest.CreateProcessTrailingSlash",
      "-program-test-string-arg1",
      "has\\\\ trailing\\",
      "-program-test-string-arg2",
      "has\\\\ trailing\\"};

  // Add LLVM_PROGRAM_TEST_CHILD to the environment of the child.
  addEnvVar("LLVM_PROGRAM_TEST_CHILD=1");

  std::string error;
  bool ExecutionFailed;
  // Redirect stdout and stdin to NUL, but let stderr through.
#ifdef _WIN32
  StringRef nul("NUL");
#else
  StringRef nul("/dev/null");
#endif
  std::optional<StringRef> redirects[] = {nul, nul, None};
  int rc = ExecuteAndWait(my_exe, argv, getEnviron(), redirects,
                          /*secondsToWait=*/ 10, /*memoryLimit=*/ 0, &error,
                          &ExecutionFailed);
  EXPECT_FALSE(ExecutionFailed) << error;
  EXPECT_EQ(0, rc);
}

TEST_F(ProgramEnvTest, TestExecuteNoWait) {
  using namespace llvm::sys;

  if (getenv("LLVM_PROGRAM_TEST_EXECUTE_NO_WAIT")) {
    sleep_for(/*seconds*/ 1);
    exit(0);
  }

  std::string Executable =
      sys::fs::getMainExecutable(TestMainArgv0, &ProgramTestStringArg1);
  StringRef argv[] = {Executable,
                      "--gtest_filter=ProgramEnvTest.TestExecuteNoWait"};

  // Add LLVM_PROGRAM_TEST_EXECUTE_NO_WAIT to the environment of the child.
  addEnvVar("LLVM_PROGRAM_TEST_EXECUTE_NO_WAIT=1");

  std::string Error;
  bool ExecutionFailed;
  ProcessInfo PI1 = ExecuteNoWait(Executable, argv, getEnviron(), {}, 0, &Error,
                                  &ExecutionFailed);
  ASSERT_FALSE(ExecutionFailed) << Error;
  ASSERT_NE(PI1.Pid, ProcessInfo::InvalidPid) << "Invalid process id";

  unsigned LoopCount = 0;

  // Test that Wait() with WaitUntilTerminates=true works. In this case,
  // LoopCount should only be incremented once.
  while (true) {
    ++LoopCount;
    ProcessInfo WaitResult = llvm::sys::Wait(PI1, 0, true, &Error);
    ASSERT_TRUE(Error.empty());
    if (WaitResult.Pid == PI1.Pid)
      break;
  }

  EXPECT_EQ(LoopCount, 1u) << "LoopCount should be 1";

  ProcessInfo PI2 = ExecuteNoWait(Executable, argv, getEnviron(), {}, 0, &Error,
                                  &ExecutionFailed);
  ASSERT_FALSE(ExecutionFailed) << Error;
  ASSERT_NE(PI2.Pid, ProcessInfo::InvalidPid) << "Invalid process id";

  // Test that Wait() with SecondsToWait=0 performs a non-blocking wait. In this
  // cse, LoopCount should be greater than 1 (more than one increment occurs).
  while (true) {
    ++LoopCount;
    ProcessInfo WaitResult = llvm::sys::Wait(PI2, 0, false, &Error);
    ASSERT_TRUE(Error.empty());
    if (WaitResult.Pid == PI2.Pid)
      break;
  }

  ASSERT_GT(LoopCount, 1u) << "LoopCount should be >1";
}

TEST_F(ProgramEnvTest, TestExecuteAndWaitTimeout) {
  using namespace llvm::sys;

  if (getenv("LLVM_PROGRAM_TEST_TIMEOUT")) {
    sleep_for(/*seconds*/ 10);
    exit(0);
  }

  std::string Executable =
      sys::fs::getMainExecutable(TestMainArgv0, &ProgramTestStringArg1);
  StringRef argv[] = {
      Executable, "--gtest_filter=ProgramEnvTest.TestExecuteAndWaitTimeout"};

  // Add LLVM_PROGRAM_TEST_TIMEOUT to the environment of the child.
 addEnvVar("LLVM_PROGRAM_TEST_TIMEOUT=1");

  std::string Error;
  bool ExecutionFailed;
  int RetCode =
      ExecuteAndWait(Executable, argv, getEnviron(), {}, /*secondsToWait=*/1, 0,
                     &Error, &ExecutionFailed);
  ASSERT_EQ(-2, RetCode);
}

TEST(ProgramTest, TestExecuteNegative) {
  std::string Executable = "i_dont_exist";
  StringRef argv[] = {Executable};

  {
    std::string Error;
    bool ExecutionFailed;
    int RetCode = ExecuteAndWait(Executable, argv, llvm::None, {}, 0, 0, &Error,
                                 &ExecutionFailed);
    ASSERT_LT(RetCode, 0) << "On error ExecuteAndWait should return 0 or "
                             "positive value indicating the result code";
    ASSERT_TRUE(ExecutionFailed);
    ASSERT_FALSE(Error.empty());
  }

  {
    std::string Error;
    bool ExecutionFailed;
    ProcessInfo PI = ExecuteNoWait(Executable, argv, llvm::None, {}, 0, &Error,
                                   &ExecutionFailed);
    ASSERT_EQ(PI.Pid, ProcessInfo::InvalidPid)
        << "On error ExecuteNoWait should return an invalid ProcessInfo";
    ASSERT_TRUE(ExecutionFailed);
    ASSERT_FALSE(Error.empty());
  }

}

#ifdef _WIN32
const char utf16le_text[] =
    "\x6c\x00\x69\x00\x6e\x00\x67\x00\xfc\x00\x69\x00\xe7\x00\x61\x00";
const char utf16be_text[] =
    "\x00\x6c\x00\x69\x00\x6e\x00\x67\x00\xfc\x00\x69\x00\xe7\x00\x61";
#endif
const char utf8_text[] = "\x6c\x69\x6e\x67\xc3\xbc\x69\xc3\xa7\x61";

TEST(ProgramTest, TestWriteWithSystemEncoding) {
  SmallString<128> TestDirectory;
  ASSERT_NO_ERROR(fs::createUniqueDirectory("program-test", TestDirectory));
  errs() << "Test Directory: " << TestDirectory << '\n';
  errs().flush();
  SmallString<128> file_pathname(TestDirectory);
  path::append(file_pathname, "international-file.txt");
  // Only on Windows we should encode in UTF16. For other systems, use UTF8
  ASSERT_NO_ERROR(sys::writeFileWithEncoding(file_pathname.c_str(), utf8_text,
                                             sys::WEM_UTF16));
  int fd = 0;
  ASSERT_NO_ERROR(fs::openFileForRead(file_pathname.c_str(), fd));
#if defined(_WIN32)
  char buf[18];
  ASSERT_EQ(::read(fd, buf, 18), 18);
  const char *utf16_text;
  if (strncmp(buf, "\xfe\xff", 2) == 0) { // UTF16-BE
    utf16_text = utf16be_text;
  } else if (strncmp(buf, "\xff\xfe", 2) == 0) { // UTF16-LE
    utf16_text = utf16le_text;
  } else {
    FAIL() << "Invalid BOM in UTF-16 file";
  }
  ASSERT_EQ(strncmp(&buf[2], utf16_text, 16), 0);
#else
  char buf[10];
  ASSERT_EQ(::read(fd, buf, 10), 10);
  ASSERT_EQ(strncmp(buf, utf8_text, 10), 0);
#endif
  ::close(fd);
  ASSERT_NO_ERROR(fs::remove(file_pathname.str()));
  ASSERT_NO_ERROR(fs::remove(TestDirectory.str()));
}

TEST_F(ProgramEnvTest, TestExecuteAndWaitStatistics) {
  using namespace llvm::sys;

  if (getenv("LLVM_PROGRAM_TEST_STATISTICS"))
    exit(0);

  std::string Executable =
      sys::fs::getMainExecutable(TestMainArgv0, &ProgramTestStringArg1);
  StringRef argv[] = {
      Executable, "--gtest_filter=ProgramEnvTest.TestExecuteAndWaitStatistics"};

  // Add LLVM_PROGRAM_TEST_STATISTICS to the environment of the child.
  addEnvVar("LLVM_PROGRAM_TEST_STATISTICS=1");

  std::string Error;
  bool ExecutionFailed;
  std::optional<ProcessStatistics> ProcStat;
  int RetCode = ExecuteAndWait(Executable, argv, getEnviron(), {}, 0, 0, &Error,
                               &ExecutionFailed, &ProcStat);
  ASSERT_EQ(0, RetCode);
  ASSERT_TRUE(ProcStat);
  ASSERT_GE(ProcStat->UserTime, std::chrono::microseconds(0));
  ASSERT_GE(ProcStat->TotalTime, ProcStat->UserTime);
}

TEST_F(ProgramEnvTest, TestLockFile) {
  using namespace llvm::sys;

  if (const char *LockedFile = getenv("LLVM_PROGRAM_TEST_LOCKED_FILE")) {
    // Child process.
    int FD2;
    ASSERT_NO_ERROR(fs::openFileForReadWrite(LockedFile, FD2,
                                             fs::CD_OpenExisting, fs::OF_None));

    std::error_code ErrC = fs::tryLockFile(FD2, std::chrono::seconds(5));
    ASSERT_NO_ERROR(ErrC);
    ASSERT_NO_ERROR(fs::unlockFile(FD2));
    close(FD2);
    exit(0);
  }

  // Create file that will be locked.
  SmallString<64> LockedFile;
  int FD1;
  ASSERT_NO_ERROR(
      fs::createTemporaryFile("TestLockFile", "temp", FD1, LockedFile));

  std::string Executable =
      sys::fs::getMainExecutable(TestMainArgv0, &ProgramTestStringArg1);
  StringRef argv[] = {Executable, "--gtest_filter=ProgramEnvTest.TestLockFile"};

  // Add LLVM_PROGRAM_TEST_LOCKED_FILE to the environment of the child.
  std::string EnvVar = "LLVM_PROGRAM_TEST_LOCKED_FILE=";
  EnvVar += LockedFile.str();
  addEnvVar(EnvVar);

  // Lock the file.
  ASSERT_NO_ERROR(fs::tryLockFile(FD1));

  std::string Error;
  bool ExecutionFailed;
  ProcessInfo PI2 = ExecuteNoWait(Executable, argv, getEnviron(), {}, 0, &Error,
                                  &ExecutionFailed);
  ASSERT_FALSE(ExecutionFailed) << Error;
  ASSERT_TRUE(Error.empty());
  ASSERT_NE(PI2.Pid, ProcessInfo::InvalidPid) << "Invalid process id";

  // Wait some time to give the child process a chance to start.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  ASSERT_NO_ERROR(fs::unlockFile(FD1));
  ProcessInfo WaitResult = llvm::sys::Wait(PI2, 5 /* seconds */, true, &Error);
  ASSERT_TRUE(Error.empty());
  ASSERT_EQ(0, WaitResult.ReturnCode);
  ASSERT_EQ(WaitResult.Pid, PI2.Pid);
  sys::fs::remove(LockedFile);
}

TEST_F(ProgramEnvTest, TestExecuteWithNoStacktraceHandler) {
  using namespace llvm::sys;

  if (getenv("LLVM_PROGRAM_TEST_NO_STACKTRACE_HANDLER")) {
    sys::PrintStackTrace(errs());
    exit(0);
  }

  std::string Executable =
      sys::fs::getMainExecutable(TestMainArgv0, &ProgramTestStringArg1);
  StringRef argv[] = {
      Executable,
      "--gtest_filter=ProgramEnvTest.TestExecuteWithNoStacktraceHandler"};

  addEnvVar("LLVM_PROGRAM_TEST_NO_STACKTRACE_HANDLER=1");

  std::string Error;
  bool ExecutionFailed;
  int RetCode = ExecuteAndWait(Executable, argv, getEnviron(), {}, 0, 0, &Error,
                               &ExecutionFailed);
  EXPECT_FALSE(ExecutionFailed) << Error;
  ASSERT_EQ(0, RetCode);
}

} // end anonymous namespace
