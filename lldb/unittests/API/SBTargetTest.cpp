//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/LLDB.h"

#include "SBTestingSupport/SBTestUtilities.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

class SBTargetTest : public SBDebuggerFixture {
protected:
  void SetUp() override {
    SBDebuggerFixture::SetUp();
    m_target = CreateEmptyTarget();
    ASSERT_TRUE(m_target.IsValid());
  }

  SBTarget m_target;
  SBFileSpecList m_no_files;
};

TEST_F(SBTargetTest, BreakpointsByNamesKeepTheLanguage) {
  const char *names[] = {"main"};
  SBBreakpoint breakpoint = m_target.BreakpointCreateByNames(
      names, 1, eFunctionNameTypeAuto, eLanguageTypeC_plus_plus, m_no_files,
      m_no_files);
  char language[16] = {};
  breakpoint.SerializeToStructuredData()
      .GetValueForKey("Breakpoint")
      .GetValueForKey("BKPTResolver")
      .GetValueForKey("Options")
      .GetValueForKey("Language")
      .GetStringValue(language, sizeof(language));
  EXPECT_STREQ(language, "c++");
}
