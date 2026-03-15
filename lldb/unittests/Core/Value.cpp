//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Value.h"
#include "Plugins/Platform/MacOSX/PlatformMacOSX.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/Symbol/ClangTestUtils.h"

#include "lldb/Utility/DataExtractor.h"

#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb_private::clang_utils;

TEST(ValueTest, GetValueAsData) {
  SubsystemRAII<FileSystem, HostInfo, PlatformMacOSX> subsystems;
  auto holder = std::make_unique<clang_utils::TypeSystemClangHolder>("test");
  auto *clang = holder->GetAST();

  Value v(Scalar(42));
  DataExtractor extractor;

  // no compiler type
  Status status = v.GetValueAsData(nullptr, extractor, nullptr);
  ASSERT_TRUE(status.Fail());

  // with compiler type
  v.SetCompilerType(clang->GetBasicType(lldb::BasicType::eBasicTypeChar));

  status = v.GetValueAsData(nullptr, extractor, nullptr);
  ASSERT_TRUE(status.Success());
}
