//===-- ProtocolTypesTest.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Protocol/ProtocolTypes.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_dap;
using namespace lldb_dap::protocol;

template <typename T> static llvm::Expected<T> roundtrip(const T &input) {
  llvm::json::Value value = toJSON(input);
  llvm::json::Path::Root root;
  T output;
  if (!fromJSON(value, output, root))
    return root.getError();
  return output;
}

TEST(ProtocolTypesTest, ExceptionBreakpointsFilter) {
  ExceptionBreakpointsFilter filter;
  filter.filter = "testFilter";
  filter.label = "Test Filter";
  filter.description = "This is a test filter";
  filter.defaultState = true;
  filter.supportsCondition = true;
  filter.conditionDescription = "Condition for test filter";

  llvm::Expected<ExceptionBreakpointsFilter> deserialized_filter =
      roundtrip(filter);
  ASSERT_THAT_EXPECTED(deserialized_filter, llvm::Succeeded());

  EXPECT_EQ(filter.filter, deserialized_filter->filter);
  EXPECT_EQ(filter.label, deserialized_filter->label);
  EXPECT_EQ(filter.description, deserialized_filter->description);
  EXPECT_EQ(filter.defaultState, deserialized_filter->defaultState);
  EXPECT_EQ(filter.supportsCondition, deserialized_filter->supportsCondition);
  EXPECT_EQ(filter.conditionDescription,
            deserialized_filter->conditionDescription);
}

TEST(ProtocolTypesTest, Source) {
  Source source;
  source.name = "testName";
  source.path = "/path/to/source";
  source.sourceReference = 12345;
  source.presentationHint = ePresentationHintEmphasize;

  llvm::Expected<Source> deserialized_source = roundtrip(source);
  ASSERT_THAT_EXPECTED(deserialized_source, llvm::Succeeded());

  EXPECT_EQ(source.name, deserialized_source->name);
  EXPECT_EQ(source.path, deserialized_source->path);
  EXPECT_EQ(source.sourceReference, deserialized_source->sourceReference);
  EXPECT_EQ(source.presentationHint, deserialized_source->presentationHint);
}
