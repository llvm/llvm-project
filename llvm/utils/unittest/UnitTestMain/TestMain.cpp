//===--- utils/unittest/UnitTestMain/TestMain.cpp - unittest driver -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Signals.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#if defined(_WIN32)
# include <windows.h>
# if defined(_MSC_VER)
#   include <crtdbg.h>
# endif
#endif

const char *TestMainArgv0;

int main(int argc, char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0],
                                          true /* Disable crash reporting */);

  // Initialize both gmock and gtest.
  testing::InitGoogleMock(&argc, argv);

  llvm::cl::ParseCommandLineOptions(argc, argv);

  // Make it easy for a test to re-execute itself by saving argv[0].
  TestMainArgv0 = argv[0];

# if defined(_WIN32)
  // Disable all of the possible ways Windows conspires to make automated
  // testing impossible.
  ::SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
#   if defined(_MSC_VER)
    ::_set_error_mode(_OUT_TO_STDERR);
    _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG);
    _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#   endif
# endif

  return RUN_ALL_TESTS();
}
