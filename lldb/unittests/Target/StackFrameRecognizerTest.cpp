//===-- StackFrameRecognizerTest.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/StackFrameRecognizer.h"
#include "Plugins/Platform/Linux/PlatformLinux.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-forward.h"
#include "lldb/lldb-private-enumerations.h"
#include "lldb/lldb-private.h"
#include "llvm/Support/FormatVariadic.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb_private::repro;
using namespace lldb;

namespace {
class StackFrameRecognizerTest : public ::testing::Test {
public:
  void SetUp() override {
    FileSystem::Initialize();
    HostInfo::Initialize();

    // Pretend Linux is the host platform.
    platform_linux::PlatformLinux::Initialize();
    ArchSpec arch("powerpc64-pc-linux");
    Platform::SetHostPlatform(
        platform_linux::PlatformLinux::CreateInstance(true, &arch));
  }

  void TearDown() override {
    platform_linux::PlatformLinux::Terminate();
    HostInfo::Terminate();
    FileSystem::Terminate();
  }
};

class DummyStackFrameRecognizer : public StackFrameRecognizer {
public:
  std::string GetName() override { return "Dummy StackFrame Recognizer"; }
};

void RegisterDummyStackFrameRecognizer(StackFrameRecognizerManager &manager) {
  RegularExpressionSP module_regex_sp = nullptr;
  RegularExpressionSP symbol_regex_sp(new RegularExpression("boom"));

  StackFrameRecognizerSP dummy_recognizer_sp(new DummyStackFrameRecognizer());

  manager.AddRecognizer(dummy_recognizer_sp, module_regex_sp, symbol_regex_sp,
                        false);
}

} // namespace

TEST_F(StackFrameRecognizerTest, NullModuleRegex) {
  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);

  StackFrameRecognizerManager manager;

  RegisterDummyStackFrameRecognizer(manager);

  bool any_printed = false;
  manager.ForEach([&any_printed](uint32_t recognizer_id, std::string name,
                                 std::string function,
                                 llvm::ArrayRef<ConstString> symbols,
                                 bool regexp) { any_printed = true; });

  EXPECT_TRUE(any_printed);
}
