//===-- LuaTests.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/ScriptInterpreter/Lua/Lua.h"
#include "gtest/gtest.h"

using namespace lldb_private;

extern "C" int luaopen_lldb(lua_State *L) { return 0; }

TEST(LuaTest, RunValid) {
  Lua lua;
  llvm::Error error = lua.Run("foo = 1");
  EXPECT_FALSE(static_cast<bool>(error));
}

TEST(LuaTest, RunInvalid) {
  Lua lua;
  llvm::Error error = lua.Run("nil = foo");
  EXPECT_TRUE(static_cast<bool>(error));
  EXPECT_EQ(llvm::toString(std::move(error)),
            "[string \"buffer\"]:1: unexpected symbol near 'nil'\n");
}
