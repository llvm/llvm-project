//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/AcceleratorGDBRemotePackets.h"
#include "TestingSupport/TestUtilities.h"
#include "llvm/Support/JSON.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace llvm;

TEST(AcceleratorGDBRemotePacketsTest, SymbolValue) {
  SymbolValue sv;
  sv.name = "my_symbol";
  sv.value = 0xDEADBEEF;

  Expected<SymbolValue> deserialized = roundtripJSON(sv);
  ASSERT_THAT_EXPECTED(deserialized, Succeeded());
  EXPECT_EQ(sv.name, deserialized->name);
  EXPECT_EQ(sv.value, deserialized->value);
}

TEST(AcceleratorGDBRemotePacketsTest, SymbolValueNullopt) {
  SymbolValue sv;
  sv.name = "missing";
  sv.value = std::nullopt;

  Expected<SymbolValue> deserialized = roundtripJSON(sv);
  ASSERT_THAT_EXPECTED(deserialized, Succeeded());
  EXPECT_EQ(sv.name, deserialized->name);
  EXPECT_EQ(std::nullopt, deserialized->value);
}

TEST(AcceleratorGDBRemotePacketsTest, AcceleratorBreakpointByName) {
  AcceleratorBreakpointByName bp;
  bp.function_name = "main";
  bp.shlib = "libfoo.so";

  Expected<AcceleratorBreakpointByName> deserialized = roundtripJSON(bp);
  ASSERT_THAT_EXPECTED(deserialized, Succeeded());
  EXPECT_EQ(bp.function_name, deserialized->function_name);
  EXPECT_EQ(bp.shlib, deserialized->shlib);
}

TEST(AcceleratorGDBRemotePacketsTest, AcceleratorBreakpointByNameNoShlib) {
  AcceleratorBreakpointByName bp;
  bp.function_name = "main";

  Expected<AcceleratorBreakpointByName> deserialized = roundtripJSON(bp);
  ASSERT_THAT_EXPECTED(deserialized, Succeeded());
  EXPECT_EQ(bp.function_name, deserialized->function_name);
  EXPECT_EQ(std::nullopt, deserialized->shlib);
}

TEST(AcceleratorGDBRemotePacketsTest, AcceleratorBreakpointByAddress) {
  AcceleratorBreakpointByAddress bp;
  bp.load_address = 0x400000;

  Expected<AcceleratorBreakpointByAddress> deserialized = roundtripJSON(bp);
  ASSERT_THAT_EXPECTED(deserialized, Succeeded());
  EXPECT_EQ(bp.load_address, deserialized->load_address);
}

TEST(AcceleratorGDBRemotePacketsTest, AcceleratorBreakpointInfoByName) {
  AcceleratorBreakpointInfo info;
  info.identifier = 42;
  info.by_name = AcceleratorBreakpointByName{std::nullopt, "main"};
  info.symbol_names = {"sym1", "sym2"};

  Expected<AcceleratorBreakpointInfo> deserialized = roundtripJSON(info);
  ASSERT_THAT_EXPECTED(deserialized, Succeeded());
  EXPECT_EQ(info.identifier, deserialized->identifier);
  ASSERT_TRUE(deserialized->by_name.has_value());
  EXPECT_EQ("main", deserialized->by_name->function_name);
  EXPECT_EQ(std::nullopt, deserialized->by_address);
  EXPECT_EQ(info.symbol_names, deserialized->symbol_names);
}

TEST(AcceleratorGDBRemotePacketsTest, AcceleratorBreakpointInfoByAddress) {
  AcceleratorBreakpointInfo info;
  info.identifier = 7;
  info.by_address = AcceleratorBreakpointByAddress{0x1000};

  Expected<AcceleratorBreakpointInfo> deserialized = roundtripJSON(info);
  ASSERT_THAT_EXPECTED(deserialized, Succeeded());
  EXPECT_EQ(info.identifier, deserialized->identifier);
  EXPECT_EQ(std::nullopt, deserialized->by_name);
  ASSERT_TRUE(deserialized->by_address.has_value());
  EXPECT_EQ(0x1000u, deserialized->by_address->load_address);
}

TEST(AcceleratorGDBRemotePacketsTest, AcceleratorBreakpointHitArgs) {
  AcceleratorBreakpointHitArgs args("mock");
  args.breakpoint.identifier = 1;
  args.breakpoint.by_name = AcceleratorBreakpointByName{std::nullopt, "main"};
  args.symbol_values = {{"sym1", 0x2000}, {"sym2", std::nullopt}};

  Expected<AcceleratorBreakpointHitArgs> deserialized = roundtripJSON(args);
  ASSERT_THAT_EXPECTED(deserialized, Succeeded());
  EXPECT_EQ("mock", deserialized->plugin_name);
  EXPECT_EQ(1, deserialized->breakpoint.identifier);
  ASSERT_EQ(2u, deserialized->symbol_values.size());
  EXPECT_EQ("sym1", deserialized->symbol_values[0].name);
  EXPECT_EQ(0x2000u, deserialized->symbol_values[0].value);
  EXPECT_EQ("sym2", deserialized->symbol_values[1].name);
  EXPECT_EQ(std::nullopt, deserialized->symbol_values[1].value);
}

TEST(AcceleratorGDBRemotePacketsTest, AcceleratorBreakpointHitArgsGetSymbol) {
  AcceleratorBreakpointHitArgs args("mock");
  args.symbol_values = {{"sym1", 0x1000}, {"sym2", 0x2000}};

  EXPECT_EQ(0x1000u, args.GetSymbolValue("sym1"));
  EXPECT_EQ(0x2000u, args.GetSymbolValue("sym2"));
  EXPECT_EQ(std::nullopt, args.GetSymbolValue("missing"));
}

TEST(AcceleratorGDBRemotePacketsTest, AcceleratorActions) {
  AcceleratorActions actions("mock", 1);
  actions.session_name = "Mock Session";
  AcceleratorBreakpointInfo bp;
  bp.identifier = 1;
  bp.by_name = AcceleratorBreakpointByName{std::nullopt, "main"};
  actions.breakpoints.push_back(std::move(bp));

  Expected<AcceleratorActions> deserialized = roundtripJSON(actions);
  ASSERT_THAT_EXPECTED(deserialized, Succeeded());
  EXPECT_EQ("mock", deserialized->plugin_name);
  EXPECT_EQ("Mock Session", deserialized->session_name);
  EXPECT_EQ(1, deserialized->identifier);
  ASSERT_EQ(1u, deserialized->breakpoints.size());
  EXPECT_EQ(1, deserialized->breakpoints[0].identifier);
  EXPECT_EQ("main", deserialized->breakpoints[0].by_name->function_name);
}

TEST(AcceleratorGDBRemotePacketsTest, AcceleratorActionsEmpty) {
  AcceleratorActions actions("test", 0);

  Expected<AcceleratorActions> deserialized = roundtripJSON(actions);
  ASSERT_THAT_EXPECTED(deserialized, Succeeded());
  EXPECT_EQ("test", deserialized->plugin_name);
  EXPECT_TRUE(deserialized->breakpoints.empty());
}

TEST(AcceleratorGDBRemotePacketsTest,
     AcceleratorBreakpointHitResponseNoActions) {
  AcceleratorBreakpointHitResponse response;
  response.disable_bp = true;
  response.auto_resume_native = false;

  Expected<AcceleratorBreakpointHitResponse> deserialized =
      roundtripJSON(response);
  ASSERT_THAT_EXPECTED(deserialized, Succeeded());
  EXPECT_TRUE(deserialized->disable_bp);
  EXPECT_FALSE(deserialized->auto_resume_native);
  EXPECT_FALSE(deserialized->actions.has_value());
}

TEST(AcceleratorGDBRemotePacketsTest,
     AcceleratorBreakpointHitResponseWithActions) {
  AcceleratorBreakpointHitResponse response;
  response.disable_bp = true;
  response.auto_resume_native = false;

  AcceleratorActions actions("mock", 2);
  AcceleratorBreakpointInfo bp;
  bp.identifier = 2;
  bp.by_name = AcceleratorBreakpointByName{std::nullopt, "exit"};
  actions.breakpoints.push_back(std::move(bp));
  response.actions = std::move(actions);

  Expected<AcceleratorBreakpointHitResponse> deserialized =
      roundtripJSON(response);
  ASSERT_THAT_EXPECTED(deserialized, Succeeded());
  EXPECT_TRUE(deserialized->disable_bp);
  EXPECT_FALSE(deserialized->auto_resume_native);
  ASSERT_TRUE(deserialized->actions.has_value());
  EXPECT_EQ("mock", deserialized->actions->plugin_name);
  EXPECT_EQ(2, deserialized->actions->identifier);
  ASSERT_EQ(1u, deserialized->actions->breakpoints.size());
  EXPECT_EQ("exit",
            deserialized->actions->breakpoints[0].by_name->function_name);
}
